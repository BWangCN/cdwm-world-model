"""Deterministic WM (Stage B2): PointNet set-encoder over the variant's gaussians -> object embedding;
MLP over [grasp pose (object frame) + commanded closing profile + table] -> context; per-step head
with a time embedding -> H x (3 trans + 6D rot) delta chunk in the gripper frame. ~1.5M params."""
import math
import torch
import torch.nn as nn


def mlp(dims, last_linear=False, act=nn.SiLU):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if not (last_linear and i == len(dims) - 2):
            layers.append(act())
    return nn.Sequential(*layers)


class WMNet(nn.Module):
    def __init__(self, n_feat=15, H=32, obj_dim=512, ctx_dim=256, t_dim=64):
        super().__init__()
        self.H, self.t_dim = H, t_dim
        self.point = mlp([n_feat, 128, 256, 512])           # per-point shared MLP
        self.obj = mlp([1024, obj_dim])                     # [max|mean]-pool -> object embedding
        self.ctx = mlp([9 + H + 1, 256, ctx_dim])           # base_rel(9)+closing(H)+table(1)
        self.head = mlp([obj_dim + ctx_dim + t_dim, 512, 512, 9], last_linear=True)

    def time_emb(self, B, device):
        t = torch.arange(self.H, device=device).float()[:, None]
        inv = torch.exp(torch.arange(0, self.t_dim, 2, device=device).float() * (-math.log(10000.0) / self.t_dim))
        e = torch.zeros(self.H, self.t_dim, device=device)
        e[:, 0::2] = torch.sin(t * inv); e[:, 1::2] = torch.cos(t * inv)
        return e[None].expand(B, -1, -1)

    def forward(self, pts, base_rel, closing, table):
        # pts (B,N,n_feat); base_rel (B,9); closing (B,H); table (B,1)
        f = self.point(pts)                                  # (B,N,512)
        o = self.obj(torch.cat([f.max(1).values, f.mean(1)], -1))   # (B,obj_dim)
        c = self.ctx(torch.cat([base_rel, closing, table], -1))     # (B,ctx_dim)
        oc = torch.cat([o, c], -1)[:, None, :].expand(-1, self.H, -1)
        x = torch.cat([oc, self.time_emb(pts.shape[0], pts.device)], -1)
        return self.head(x)                                  # (B,H,9)
