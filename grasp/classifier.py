"""Grasp-outcome classifier: reuses the validated `local` encoder (gripper-frame cloud + covariance + contact-weighted
pool) to a 256-d embedding, then TWO linear heads (5-way v2_outcome + 2-tier success/failure), trained directly (Codex).
Post-hoc temperature scaling per head is applied at eval, not here."""
import torch
import torch.nn as nn
from common.dit_local import DiTWMLocal


class OutcomeNet(nn.Module):
    def __init__(self, n_feat=13, D=256, depth=4, obj_dim=512, ctx_dim=256):
        super().__init__()
        # reuse DiTWMLocal purely as the conditioning encoder (its .encode); the DiT denoiser is unused here.
        # H=32 so ctx = mlp([9+H+1,...]) matches the closing(=zeros,32) input; depth=1 keeps the unused blocks tiny.
        self.enc = DiTWMLocal(n_feat=n_feat, H=32, D=D, depth=1, obj_dim=obj_dim, ctx_dim=ctx_dim)
        self.head5 = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, 5))
        self.head2 = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, 2))

    def forward(self, pts, base_rel, closing, table):
        z = self.enc.encode(pts, base_rel, closing, table)     # (B, D) conditioning embedding
        return self.head5(z), self.head2(z)
