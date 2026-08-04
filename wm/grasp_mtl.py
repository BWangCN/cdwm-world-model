"""Clean multi-task grasp WM (the locked ideal workflow). ONE model:
  shared local encoder TRUNK (DiTWMLocal.encode -> 256-d cond)
    -> PRIVATE trajectory branch -> DiT diffusion head (rigid H=32 settle rollout)
    -> PRIVATE boolean branch    -> slip head (rigid vs not-rigid)
    -> PRIVATE gripper branch    -> grip head (achieved closure driver_rad over the H=32 window, normalized [0,1])
Trained jointly FROM SCRATCH; per-batch multi-task loss with the trajectory loss MASKED to rigid episodes and the
boolean CE on all outcomes (Kendall-weighted). Use = classify -> gate (discard slip) -> roll out the trajectory.
The private branches keep the boolean's gradients from corrupting the DiT conditioning (avoids the negative transfer
seen with shared-encoder-only co-training)."""
import torch
import torch.nn as nn
from wm.model import mlp
from wm.dit_local import DiTWMLocal


class GraspMTL(DiTWMLocal):
    def __init__(self, n_feat=13, H=32, D=256, depth=4, **kw):
        super().__init__(n_feat=n_feat, H=H, D=D, depth=depth, **kw)   # shared trunk: point MLP + pools + obj/ctx -> cond(D)
        self.H = H
        self.traj_branch = mlp([D, D, D])                              # private trajectory branch (conditions the DiT)
        self.bool_branch = mlp([D, D, D // 2])                         # private boolean branch
        self.head2 = nn.Linear(D // 2, 2)                              # slip logits (0=rigid, 1=slip/not-rigid)
        self.grip_branch = mlp([D, D, D // 2])                         # private gripper branch
        self.grip_head = nn.Linear(D // 2, H)                          # achieved closure over the H window (normalized [0,1])
        self.log_s = nn.Parameter(torch.zeros(3))                      # Kendall task log-variances (traj, bool, grip)

    def forward(self, x, t, cond):                                     # DiT eps-prediction with PRIVATE trajectory cond
        return super().forward(x, t, self.traj_branch(cond))

    def classify(self, cond):
        return self.head2(self.bool_branch(cond))                      # (B,2)

    def predict_grip(self, cond):
        return self.grip_head(self.grip_branch(cond))                  # (B,H) normalized achieved closure
