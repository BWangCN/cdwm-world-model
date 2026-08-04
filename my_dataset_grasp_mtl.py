"""Unified grasp MTL dataset (the locked clean workflow): ONE dataset = outcomes_v2 (all outcomes). Per episode returns
the encoder input (gripper-frame cloud built from the settle-target's OWN stored t0 pose -> guaranteed frame-consistent
with the target), base_rel, the re-derived H=32 rigid closing-settle target (z-scored by rigid stats; supervised only
for rigid via `is_rigid`), and the slip label `y_slip` (1 = not-rigid). Trajectory loss is masked to rigid; boolean CE
on all outcomes."""
import os
import numpy as np, torch
from torch.utils.data import Dataset
import my_config as C
from wm.common import quat_to_R, R_to_6d
from my_dataset_outcomes import OV, _splits, _raw_cloud, build_feats, _STATS

SETTLE = os.environ.get("CDWM_SETTLE",                              # v3-derived target (poses + grip); OV still serves clouds/stats/splits
    "/misc/kcgscratch1/MengyeGroup/yy5259/datasets/cdwm-grasp-dataset/gripper_v3/tier_a_outcomes_v2_aligned/settle_target.npz")
GRIP_MAX = 0.8                                                      # right_driver_joint range (rad) -> normalize driver_rad to [0,1]


class GraspMTLDS(Dataset):
    def __init__(self, split, stats=None, n_points=4096, seed=0):
        z = np.load(SETTLE, allow_pickle=True)
        sp = _splits()
        keep = np.array([i for i, o in enumerate(z["object_id"]) if sp.get(str(o)) == split])
        assert len(keep), f"no episodes for split {split}"
        self.target = z["target"][keep].astype(np.float32)              # (n,H,9) re-derived settle target
        self.base_rel = z["base_rel"][keep].astype(np.float32)          # (n,9)
        self.t0 = z["t0"][keep].astype(np.float64)                      # (n,14) obj_quat,obj_pos,base_quat,base_pos
        self.obj = z["object_id"][keep]
        self.is_rigid = (z["v2_outcome"][keep] == "RIGID").astype(np.float32)
        self.grip = (z["grip_target"][keep].astype(np.float32) / GRIP_MAX).clip(0, 1)   # (n,H) achieved closure, normalized [0,1]
        self.grip_phase = z["grip_phase"][keep].astype(np.int8)         # (n,H) phase per frame (1=close 2=hold) for phase-sliced eval
        self.n, self.clc = n_points, {}
        self.rng, self.fixed = np.random.default_rng(seed), (split != "train")
        d = np.load(_STATS); self.fmean, self.fstd = d["mean"], d["std"]  # feature z-score (outcomes_v2 local stats)
        if stats is None:                                              # target z-score from RIGID targets only
            T = self.target[self.is_rigid.astype(bool)].reshape(-1, 9)
            sel = np.random.default_rng(0).choice(len(T), min(20000, len(T)), replace=False)
            stats = {"mean": T[sel].mean(0).astype(np.float32), "std": (T[sel].std(0) + 1e-4).astype(np.float32)}
        self.stats = stats

    def _cl(self, o):
        if o not in self.clc: self.clc[o] = _raw_cloud(o)
        return self.clc[o]

    def __len__(self): return len(self.obj)

    def rigid_frac(self): return float(self.is_rigid.mean())

    def __getitem__(self, j):
        o = str(self.obj[j]); rc = self._cl(o); t0 = self.t0[j]
        R0, p0, Rg, bp = quat_to_R(t0[:4]), t0[4:7], quat_to_R(t0[7:11]), t0[11:14]
        rng = np.random.default_rng(7000 + j) if self.fixed else self.rng
        idx = rng.choice(len(rc["mu"]), self.n, replace=len(rc["mu"]) < self.n)
        feat, dperp = build_feats(rc["mu"][idx], rc["Sig_obj"][idx], rc["opacity"][idx], rc["ic"][idx], R0, p0, Rg, bp)
        feat = (feat - self.fmean) / self.fstd
        pts = np.concatenate([feat, dperp[:, None]], 1).astype(np.float32)
        tgt = np.clip((self.target[j] - self.stats["mean"]) / self.stats["std"], -8, 8).astype(np.float32)
        return dict(pts=torch.from_numpy(pts), base_rel=torch.from_numpy(self.base_rel[j]),
                    closing=torch.zeros(C.H, dtype=torch.float32), table=torch.zeros(1),   # driver_rad is a TARGET, never an input (no leakage)
                    target=torch.from_numpy(tgt), is_rigid=torch.tensor(self.is_rigid[j], dtype=torch.float32),
                    y_slip=torch.tensor(int(self.is_rigid[j] == 0)), object=o,
                    grip_tgt=torch.from_numpy(self.grip[j]), grip_phase=torch.from_numpy(self.grip_phase[j]),   # (H,) achieved closure [0,1] + phase
                    t0=torch.from_numpy(self.t0[j].astype(np.float32)))   # (14) obj_quat,obj_pos,base_quat,base_pos — for ADD frame
