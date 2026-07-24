"""Grasp-LOCAL encoder variant of DiTWM (Lever 1). Same DiT diffusion head; only the object encoder changes.

Motivation (grounded): the baseline pools 4096 gaussians with a single global max+mean -> grasp-blind, so the network
must implicitly cross-reference "where are the jaws vs the shape". Held-out tilt-DIRECTION error stays ~15-20deg.
Here the cloud is in the GRIPPER frame (contact geometry is analytically fixed: pinch point (0,0,0.1159), jaws close
along +x -- verified across all objects), gaussians carry a sign/axis-unambiguous COVARIANCE (Codex lever c), and we
add a CONTACT-WEIGHTED pool (soft, parameter-free, emphasizes points near the pinch line) alongside the global pool so
the embedding sees BOTH the contact patch and the far mass (lever arm) that sets the tilt torque.

Contract: pts is (B,N,14) = [13 standardized point features | raw d_perp for pooling weights]. n_feat=13 (point MLP).
Everything else (DiT blocks, DDPM schedule, DDIM sampler) is inherited unchanged from wm.dit."""
import torch
import torch.nn as nn
from common.model import mlp
from common.dit import DiTWM


class DiTWMCov(DiTWM):
    """Ablation: covariance features (gripper frame) but PLAIN global max+mean pool (no contact weighting). Isolates the
    covariance contribution between frame_only and the full local encoder. pts is (B,N,14); drops the weight channel."""
    def encode(self, pts, base_rel, closing, table):
        f = self.point(pts[..., :-1])                        # (B,N,13); ignore the raw-d_perp weight channel
        o = self.obj(torch.cat([f.max(1).values, f.mean(1)], -1))
        c = self.ctx(torch.cat([base_rel, closing, table], -1))
        return self.cond_proj(torch.cat([o, c], -1))


class DiTWMLR(DiTWM):
    """L/R finger-specific contact pooling (Codex #1). Same 13 gripper-frame point features as DiTWMLocal, but FIVE pools:
    global max+mean, the pinch-line pool (d_perp), and TWO finger-specific pools weighted toward the +x / -x object extremes
    (d_pos = x_max-x, d_neg = x-x_min). Targets tilt-DIRECTION asymmetry in the high-tilt tail. pts=(B,N,16) =
    [13 std feats | raw d_perp | raw d_pos | raw d_neg]. Soft weights (runtime-median tau) -> no empty-side/origin issues."""
    def __init__(self, n_feat=13, H=32, D=256, depth=4, heads=4, mlp_ratio=2.0, obj_dim=512, ctx_dim=256):
        super().__init__(n_feat=n_feat, H=H, D=D, depth=depth, heads=heads, mlp_ratio=mlp_ratio, obj_dim=obj_dim, ctx_dim=ctx_dim)
        self.obj = mlp([512 * 5, obj_dim])                    # global max+mean + pinch + pos-finger + neg-finger

    def encode(self, pts, base_rel, closing, table):
        pf = pts[..., :-3]                                     # (B,N,13)
        d = pts[..., -3:]                                      # (B,N,3) raw d_perp,d_pos,d_neg
        f = self.point(pf)                                     # (B,N,512)
        tau = d.median(1, keepdim=True).values.clamp_min(1e-6)
        w = torch.softmax(-d / tau, dim=1)                    # (B,N,3) three soft weightings
        pools = torch.bmm(w.transpose(1, 2), f)               # (B,3,512): pinch, pos-finger, neg-finger pools
        o = self.obj(torch.cat([f.max(1).values, f.mean(1), pools.reshape(pools.shape[0], -1)], -1))
        c = self.ctx(torch.cat([base_rel, closing, table], -1))
        return self.cond_proj(torch.cat([o, c], -1))


class DiTWMLocal(DiTWM):
    def __init__(self, n_feat=13, H=32, D=256, depth=4, heads=4, mlp_ratio=2.0, obj_dim=512, ctx_dim=256):
        super().__init__(n_feat=n_feat, H=H, D=D, depth=depth, heads=heads, mlp_ratio=mlp_ratio,
                         obj_dim=obj_dim, ctx_dim=ctx_dim)
        # three pooled statistics (global max, global mean, contact-weighted mean) -> obj embedding
        self.obj = mlp([512 * 3, obj_dim])

    def encode(self, pts, base_rel, closing, table):
        pf = pts[..., :-1]                                   # (B,N,13) standardized point features
        wraw = pts[..., -1]                                  # (B,N) raw d_perp (dist to pinch line, meters)
        f = self.point(pf)                                   # (B,N,512)
        gmax = f.max(1).values
        gmean = f.mean(1)
        tau = wraw.median(1, keepdim=True).values + 1e-6     # per-cloud scale -> dimensionless, parameter-free
        w = torch.softmax(-wraw / tau, dim=1)                # (B,N) emphasize points near the pinch line
        cmean = (w[..., None] * f).sum(1)                    # contact-weighted mean
        o = self.obj(torch.cat([gmax, gmean, cmean], -1))
        c = self.ctx(torch.cat([base_rel, closing, table], -1))
        return self.cond_proj(torch.cat([o, c], -1))
