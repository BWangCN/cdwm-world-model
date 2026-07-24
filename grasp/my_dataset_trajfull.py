"""Rollout dataset: reuses OutcomeDS (t=0 cloud+grasp local features + outcome label) and adds the SHORT native-time
rollout target (K x 9 object-in-gripper motion relative to lift-onset, from traj_full.npz key 'short'; per-dim z-scored,
frame-independent). kind='full' loads the absolute 32-frame resample for viz instead. Overrides `closing` to length K so
the DiT ctx (9+K+1) matches the shortened horizon. Trains on ALL outcomes.
"""
import os, sys, numpy as np, torch
from grasp import my_dataset_outcomes as O

OV = O.OV
_FS = os.path.join(OV, "trajfull_stats_%s.npz")


class TrajFullDS(O.OutcomeDS):
    def __init__(self, split, modality="full", kind="short", **kw):
        super().__init__(split, modality=modality, **kw)
        self.kind = kind
        z = np.load(os.path.join(OV, "traj_full.npz"), allow_pickle=True)
        self.T = z["short"] if kind == "short" else z["traj"]        # (N,K,9) or (N,32,9)
        self.uid2t = {u: i for i, u in enumerate(z["uid"])}
        self.K = self.T.shape[1]
        fs = _FS % kind
        if os.path.exists(fs): d = np.load(fs); self.tmean, self.tstd = d["mean"], d["std"]
        else:
            tr = np.stack([self.T[self.uid2t[r["uid"]]] for r in O._all_rows()
                           if O._splits().get(r["object_id"]) == "train" and r["uid"] in self.uid2t])  # (Ntr,K,9)
            flat = tr.reshape(-1, 9)
            self.tmean = flat.mean(0).astype(np.float32); self.tstd = (flat.std(0) + 1e-4).astype(np.float32)
            np.savez(fs, mean=self.tmean, std=self.tstd); print(f"[trajfull stats {kind}] cached {fs}")

    def __getitem__(self, j):
        d = super().__getitem__(j)
        t = self.T[self.uid2t[self.rows[j]["uid"]]]                  # (K,9)
        d["target"] = torch.from_numpy(((t - self.tmean) / self.tstd).astype(np.float32))
        d["target_raw"] = torch.from_numpy(t.astype(np.float32))
        d["closing"] = torch.zeros(self.K, dtype=torch.float32)      # match DiT ctx to the K-frame horizon
        return d
