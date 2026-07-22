"""Gate B distributional WM — DiT diffusion head (the CDWM signature), H=1 (denoise the final resting SE(3)),
conditioned on the release-frame cloud + release params via the shared DiTWMLocal encoder. Oracle arm adds the
hidden CoM latent to the conditioning. Reuses wm/dit.py DDPM(cosine_acp) + DDIM(ddim_sample). Point arm =
deterministic regressor on the same encoder (the baseline the distribution must beat)."""
import torch
import torch.nn as nn
from wm.dit_local import DiTWMLocal
from wm.dit import cosine_acp, ddim_sample                  # reused verbatim
from wm.model import mlp


class GateBDiT(DiTWMLocal):
    """Diffusion head. use_latent=True -> oracle (conditions on the true CoM); False -> no-latent (CoM hidden)."""
    def __init__(self, n_feat=13, use_latent=False, latent_dim=4, D=256, depth=4, **kw):
        super().__init__(n_feat=n_feat, H=1, D=D, depth=depth, **kw)
        self.use_latent = use_latent
        if use_latent:
            self.lat = mlp([latent_dim, 128, D])

    def cond(self, pts, base_rel, closing, table, latent=None):
        c = self.encode(pts, base_rel, closing, table)       # (B,D) shared encoder
        if self.use_latent and latent is not None:
            c = c + self.lat(latent)
        return c
    # model(x, t, cond) -> predicted noise (B,1,9) is inherited from DiTWM.forward


class GateBPoint(DiTWMLocal):
    """Deterministic point regressor on the same encoder (predicts the single resting pose, 9d)."""
    def __init__(self, n_feat=13, D=256, depth=1, **kw):
        super().__init__(n_feat=n_feat, H=1, D=D, depth=depth, **kw)
        self.head = mlp([D, D, 9])

    def predict(self, pts, base_rel, closing, table):
        return self.head(self.encode(pts, base_rel, closing, table))   # (B,9)


def diffusion_loss(model, x0, cond, acp):                    # DDPM epsilon-prediction (B,1,9)
    B = x0.shape[0]
    t = torch.randint(0, len(acp), (B,), device=x0.device)
    a = acp[t][:, None, None]
    eps = torch.randn_like(x0)
    xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
    return ((model(xt, t, cond) - eps) ** 2).mean()
