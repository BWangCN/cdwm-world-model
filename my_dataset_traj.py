"""Phase-2 dataset: reuses OutcomeDS (t=0 cloud+grasp local features + mode label) and adds the per-episode trajectory
SUMMARY target (from traj_summaries.npz, uid-keyed; z-scored by train stats). Pairs with wm.trajnet.TrajNet.
modality full|pose_only carried through OutcomeDS. CDWM_TASK=lifted supported (drops never-lifted)."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import my_dataset_outcomes as O

OV = O.OV
_SS = os.path.join(OV, "summ_stats.npz")


def _summ():
    z = np.load(os.path.join(OV, "traj_summaries.npz"), allow_pickle=True)
    return z["summ"], {u: i for i, u in enumerate(z["uid"])}


class TrajDS(O.OutcomeDS):
    def __init__(self, split, modality="full", **kw):
        super().__init__(split, modality=modality, **kw)
        self.S, self.uid2s = _summ()
        if os.path.exists(_SS): d = np.load(_SS); self.smean, self.sstd = d["mean"], d["std"]
        else:
            tr = np.stack([self.S[self.uid2s[r["uid"]]] for r in O._all_rows()
                           if O._splits().get(r["object_id"]) == "train" and r["uid"] in self.uid2s])
            self.smean = tr.mean(0).astype(np.float32); self.sstd = (tr.std(0) + 1e-4).astype(np.float32)
            np.savez(_SS, mean=self.smean, std=self.sstd); print(f"[summ stats] cached {_SS}")

    def __getitem__(self, j):
        d = super().__getitem__(j)
        s = self.S[self.uid2s[self.rows[j]["uid"]]]
        d["summ"] = torch.from_numpy(((s - self.smean) / self.sstd).astype(np.float32))
        d["summ_raw"] = torch.from_numpy(s.astype(np.float32))
        return d
