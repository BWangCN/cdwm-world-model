"""DiT diffusion head for the WM (Tier-2 C1). Same conditioning as WMNet (PointNet object embedding +
grasp/closing context); the per-step head is replaced by a Diffusion Transformer (AdaLN-Zero) that
denoises the H x 9 delta-chunk. DDPM (epsilon-prediction, cosine schedule) + DDIM sampling. ~4.9M."""
import math
import torch
import torch.nn as nn
from common.model import mlp


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device).float() / half)
    a = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(a), torch.sin(a)], -1)


class DiTBlock(nn.Module):
    def __init__(self, D, heads, mlp_ratio):
        super().__init__()
        self.n1 = nn.LayerNorm(D, elementwise_affine=False)
        self.n2 = nn.LayerNorm(D, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(D, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(D, int(D * mlp_ratio)), nn.GELU(), nn.Linear(int(D * mlp_ratio), D))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(D, 6 * D))
        nn.init.zeros_(self.ada[-1].weight); nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, c):
        s1, b1, g1, s2, b2, g2 = self.ada(c).chunk(6, dim=-1)
        h = self.n1(x) * (1 + s1[:, None]) + b1[:, None]
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + g1[:, None] * a
        h = self.n2(x) * (1 + s2[:, None]) + b2[:, None]
        return x + g2[:, None] * self.mlp(h)


class DiTWM(nn.Module):
    def __init__(self, n_feat=15, H=32, D=256, depth=4, heads=4, mlp_ratio=2.0, obj_dim=512, ctx_dim=256):
        super().__init__()
        self.H, self.D = H, D
        self.point = mlp([n_feat, 128, 256, 512])
        self.obj = mlp([1024, obj_dim])
        self.ctx = mlp([9 + H + 1, 256, ctx_dim])
        self.cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(obj_dim + ctx_dim, D))
        self.t_embed = nn.Sequential(nn.Linear(D, D), nn.SiLU(), nn.Linear(D, D))
        self.x_embed = nn.Linear(9, D)
        self.pos = nn.Parameter(torch.zeros(1, H, D))
        self.blocks = nn.ModuleList([DiTBlock(D, heads, mlp_ratio) for _ in range(depth)])
        self.nf = nn.LayerNorm(D, elementwise_affine=False)
        self.adaf = nn.Sequential(nn.SiLU(), nn.Linear(D, 2 * D))
        nn.init.zeros_(self.adaf[-1].weight); nn.init.zeros_(self.adaf[-1].bias)
        self.out = nn.Linear(D, 9); nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def encode(self, pts, base_rel, closing, table):
        f = self.point(pts)
        o = self.obj(torch.cat([f.max(1).values, f.mean(1)], -1))
        c = self.ctx(torch.cat([base_rel, closing, table], -1))
        return self.cond_proj(torch.cat([o, c], -1))                       # (B,D)

    def forward(self, x, t, cond):                                          # x (B,H,9), t (B,), cond (B,D)
        c = cond + self.t_embed(timestep_embedding(t, self.D))
        h = self.x_embed(x) + self.pos
        for blk in self.blocks:
            h = blk(h, c)
        s, b = self.adaf(c).chunk(2, -1)
        h = self.nf(h) * (1 + s[:, None]) + b[:, None]
        return self.out(h)                                                 # predicted noise (B,H,9)


def cosine_acp(T, s=0.008, device="cpu"):
    x = torch.arange(T + 1, device=device).float() / T
    f = torch.cos((x + s) / (1 + s) * math.pi / 2) ** 2
    acp = f / f[0]
    betas = (1 - acp[1:] / acp[:-1]).clamp(1e-6, 0.999)
    acp = torch.cumprod(1 - betas, 0)                                       # alphas_cumprod, len T
    return acp


@torch.no_grad()
def ddim_sample(model, cond, H, acp, steps=50, device="cuda", x_T=None):
    B = cond.shape[0]; T = len(acp)
    ts = torch.linspace(T - 1, 0, steps, device=device).long()
    x = torch.randn(B, H, 9, device=device) if x_T is None else x_T.to(device)   # x_T: seeded noise for reproducible eval
    for i in range(steps):
        t = ts[i]; a = acp[t]
        eps = model(x, t.expand(B), cond)
        x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-8, 8)        # clamp z-scored x0 -> stable DDIM
        a_prev = acp[ts[i + 1]] if i + 1 < steps else torch.tensor(1.0, device=device)
        x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps                 # DDIM eta=0
    return x0
