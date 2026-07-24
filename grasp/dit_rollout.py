"""Full-rollout WM (colleague's 'visualize the future'): DiTWMLocal (local encoder + DiT denoiser — now actually USED)
denoises the full H x 9 object-in-gripper trajectory over ALL v2 outcomes, PLUS a boolean success/fail READ-OFF head off
the shared conditioning embedding. The boolean is NOT fed back into the DiT -> samples stay p(traj | scene, action).
Kendall-weighted DDPM-eps + CE. Sample -> multimodal futures (rigid/slip/drop); classify -> P(fail)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.dit_local import DiTWMLocal


class DiTRollout(nn.Module):
    def __init__(self, n_feat=13, H=32, D=256, depth=4):
        super().__init__()
        self.H = H
        self.dit = DiTWMLocal(n_feat=n_feat, H=H, D=D, depth=depth)     # local encoder + full DiT denoiser
        self.head2 = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, 2))   # read-off success/fail
        self.log_s = nn.Parameter(torch.zeros(2))                      # learned task log-variances (diff, bce)

    def encode(self, pts, base_rel, closing, table):
        return self.dit.encode(pts, base_rel, closing, table)          # (B,D) shared conditioning

    def denoise(self, x, t, cond):
        return self.dit(x, t, cond)                                    # predicted noise (B,H,9)

    def classify(self, cond):
        return self.head2(cond)                                        # (B,2)


def rollout_loss(model, cond, target, yfail, acp, cw):
    """DDPM eps-MSE over the full trajectory + boolean CE, Kendall-weighted. cond is encoded ONCE and reused for both
    (read-off boolean). target (B,H,9) z-scored; acp = alphas_cumprod (T,); cw = binary class weights."""
    B = target.shape[0]; T = len(acp)
    t = torch.randint(0, T, (B,), device=target.device)
    a = acp[t][:, None, None]
    noise = torch.randn_like(target)
    x = a.sqrt() * target + (1 - a).sqrt() * noise
    dmse = ((model.denoise(x, t, cond) - noise) ** 2).mean()
    ce = F.cross_entropy(model.classify(cond), yfail, weight=cw)
    s = model.log_s
    return torch.exp(-s[0]) * dmse + s[0] + torch.exp(-s[1]) * ce + s[1], dmse.detach(), ce.detach()
