"""Baseline drop WM: reuses the slip `local` encoder (DiTWMLocal.encode) to a 256-d embedding, then two heads:
  - basin_transition  (binary: does the object tip into a different stable pose?)
  - rest6d            (6D rotation: the resting orientation relative to release)
Point-prediction baseline (the deterministic reference the CoG/distributional contribution is measured against)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from wm.dit_local import DiTWMLocal


def sixd_to_R(d):                                          # (...,6) -> (...,3,3), columns (matches wm.common.R_to_6d)
    a, b = d[..., :3], d[..., 3:]
    x = F.normalize(a, dim=-1, eps=1e-8)
    b = b - (x * b).sum(-1, keepdim=True) * x
    y = F.normalize(b, dim=-1, eps=1e-8)
    z = torch.cross(x, y, dim=-1)
    return torch.stack([x, y, z], dim=-1)


class DropNet(nn.Module):
    def __init__(self, n_feat=13, D=256, obj_dim=512, ctx_dim=256):
        super().__init__()
        self.enc = DiTWMLocal(n_feat=n_feat, H=32, D=D, depth=1, obj_dim=obj_dim, ctx_dim=ctx_dim)
        self.head_bt = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, 2))
        self.head_rot = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, 6))

    def forward(self, pts, base_rel, closing, table):
        z = self.enc.encode(pts, base_rel, closing, table)     # (B, D)
        return self.head_bt(z), self.head_rot(z)               # (B,2), (B,6)


def rot_loss(rot6d_pred, R_gt):                            # chordal Frobenius on rotation matrices (stable, differentiable)
    Rp = sixd_to_R(rot6d_pred)
    return ((Rp - R_gt) ** 2).sum((-1, -2)).mean()
