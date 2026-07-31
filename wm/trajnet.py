"""Phase-2 model (Codex): shared local encoder -> mode head (5-way, reuse Phase-1 idea) + PER-MODE summary experts.
Train with GT-mode teacher forcing (loss on the GT-mode expert). Test: report GT-mode oracle vs predicted-mode mixture
sum_m P(m|x) summary_m. K=5 modes, summary dim 15."""
import torch
import torch.nn as nn
from wm.dit_local import DiTWMLocal


class DirectTrajNet(nn.Module):
    """(a') Mode-AGNOSTIC direct continuous summary regressor with heteroscedastic uncertainty (Codex #1). Encoder -> per-
    target mean + log-scale (in z-scored space); Gaussian NLL. Derive mode/risk from the predicted continuous summary AFTER.
    Sidesteps the aleatoric discrete-mode bottleneck that capped the mode-conditioned mixture at deployment."""
    def __init__(self, n_feat=13, D=256, S=15):
        super().__init__()
        self.S = S; self.enc = DiTWMLocal(n_feat=n_feat, H=32, D=D, depth=1)
        self.head = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, 2 * S))

    def forward(self, pts, base_rel, closing, table):
        h = self.head(self.enc.encode(pts, base_rel, closing, table))
        return h[:, :self.S], h[:, self.S:].clamp(-6, 3)          # mean, log-scale (z-scored space)


class TrajNet(nn.Module):
    def __init__(self, n_feat=13, D=256, K=5, S=15):
        super().__init__()
        self.K, self.S = K, S
        self.enc = DiTWMLocal(n_feat=n_feat, H=32, D=D, depth=1)
        self.mode = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, K))
        self.experts = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, K * S))   # per-mode summary head

    def forward(self, pts, base_rel, closing, table):
        z = self.enc.encode(pts, base_rel, closing, table)          # (B,D)
        return self.mode(z), self.experts(z).view(-1, self.K, self.S)   # (B,K) logits, (B,K,S) per-mode summaries
