"""Phase-2 model (Codex): shared local encoder -> mode head (5-way, reuse Phase-1 idea) + PER-MODE summary experts.
Train with GT-mode teacher forcing (loss on the GT-mode expert). Test: report GT-mode oracle vs predicted-mode mixture
sum_m P(m|x) summary_m. K=5 modes, summary dim 15."""
import torch
import torch.nn as nn
from wm.dit_local import DiTWMLocal


class MDNTrajNet(nn.Module):
    """(#1 push) Mixture-Density Network over the continuous trajectory summaries: geometry -> a K-component mixture
    (weight, per-dim mean, per-dim log-scale) = a CALIBRATED DISTRIBUTION over contact outcomes, WITHOUT threshold mode
    labels. Modes EMERGE as mixture components. Trained by mixture NLL. Fixes (a')'s single-mean-over-multimodal failure."""
    def __init__(self, n_feat=13, D=256, K=8, S=15):
        super().__init__()
        self.K, self.S = K, S
        self.enc = DiTWMLocal(n_feat=n_feat, H=32, D=D, depth=1)
        self.head = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, K + 2 * K * S))

    def forward(self, pts, base_rel, closing, table):
        h = self.head(self.enc.encode(pts, base_rel, closing, table))
        logit = h[:, :self.K]
        r = h[:, self.K:].view(-1, self.K, 2, self.S)
        return logit, r[:, :, 0], r[:, :, 1].clamp(-6, 3)          # mix logits (B,K), means (B,K,S), log-scales (B,K,S)


def mdn_nll(logit, mean, logscale, y):                            # y (B,S) z-scored; diagonal-Gaussian mixture NLL
    import math
    logN = (-0.5 * ((y[:, None, :] - mean) * torch.exp(-logscale)) ** 2 - logscale - 0.5 * math.log(2 * math.pi)).sum(-1)
    return -torch.logsumexp(torch.log_softmax(logit, -1) + logN, -1).mean()


class JointMDNOutcomeNet(nn.Module):
    """Colleague's suggestion (validation control): ONE shared `local` encoder feeding BOTH the MDN summary head
    (distributional WM, = MDNTrajNet's head) AND a binary success/fail head (= OutcomeNet.head2), JOINTLY trained.
    Tests whether joint success supervision helps the WM vs our SEPARATE MDN + separate classifier. Task losses are
    combined by learned uncertainty weights (Kendall) so there are no hand-tuned coefficients."""
    def __init__(self, n_feat=13, D=256, K=8, S=15):
        super().__init__()
        self.K, self.S = K, S
        self.enc = DiTWMLocal(n_feat=n_feat, H=32, D=D, depth=1)
        self.mdn = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, K + 2 * K * S))   # = MDNTrajNet head
        self.head2 = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, 2))              # = OutcomeNet.head2
        self.log_s = nn.Parameter(torch.zeros(2))                  # learned task log-variances (mdn, bce)

    def forward(self, pts, base_rel, closing, table):
        z = self.enc.encode(pts, base_rel, closing, table)
        h = self.mdn(z); r = h[:, self.K:].view(-1, self.K, 2, self.S)
        return h[:, :self.K], r[:, :, 0], r[:, :, 1].clamp(-6, 3), self.head2(z)   # logit(B,K), mean, logscale, cls2(B,2)


def joint_loss(logit, mean, logscale, cls2, y_summ, y_fail, log_s, cw):
    nll = mdn_nll(logit, mean, logscale, y_summ)
    ce = nn.functional.cross_entropy(cls2, y_fail, weight=cw)
    return torch.exp(-log_s[0]) * nll + log_s[0] + torch.exp(-log_s[1]) * ce + log_s[1], nll.detach(), ce.detach()


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
