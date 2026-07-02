"""COLORLESS cross-object WM dataset for the corrected stable-187. Same target/conditioning as wm_dataset_scaling, EXCEPT
cluster_feats drops sh_dc -> 12-feat COLORLESS (mu+rot_quat+scale+opacity+is_completed). Chunk = full close+settle [0,H];
target = per-step gripper-frame SE(3) delta (z-scored); conditioning = colorless cluster (N=4096) + grasp-pose + COMMANDED
closing + table. CSV rows carry episode ABSPATH, object, shape, net_tilt, split, weight."""
import csv, os, sys
import numpy as np, torch
from torch.utils.data import Dataset
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import my_config as C
from wm.common import quat_to_R, R_to_6d


def load_rows(split=None):
    rows = list(csv.DictReader(open(C.CHUNKS_CSV)))
    rows = [r for r in rows if (split is None or r["split"] == split)]
    return [r for r in rows if os.path.exists(f"{C.CLUSTERS}/gaussians_{r['object']}_clean.npz")]


_FS = np.load(f"{C.RUN}/feat_stats.npz"); FMEAN, FSTD = _FS["mean"], _FS["std"]   # per-channel standardization (opacity 0-20 dwarfs mu/scale -> unstable PointNet)


def cluster_feats(obj):
    g = np.load(f"{C.CLUSTERS}/gaussians_{obj}_clean.npz"); ic = g["is_completed"].astype(np.float32)[:, None]
    feat = np.concatenate([g["mu"], g["rot_quat"], g["scale"], g["opacity"], ic], 1).astype(np.float32)   # 12-feat COLORLESS, NO sh_dc
    return ((feat - FMEAN) / FSTD).astype(np.float32)


class ChunkDS(Dataset):
    def __init__(self, rows, H=C.H, n_points=4096, stats=None, fixed_subsample=False, seed=0):
        self.rows, self.H, self.n, self.stats, self.fixed = rows, H, n_points, stats, fixed_subsample
        self.epc, self.clc, self.rng = {}, {}, np.random.default_rng(seed)

    def _ep(self, path):
        if path not in self.epc:
            self.epc[path] = dict(np.load(path, allow_pickle=True))
        return self.epc[path]

    def _cl(self, obj):
        if obj not in self.clc:
            self.clc[obj] = cluster_feats(obj)
        return self.clc[obj]

    def __len__(self): return len(self.rows)
    def weights(self): return np.array([float(r["weight"]) for r in self.rows], np.float64)

    def __getitem__(self, i):
        r = self.rows[i]; H = self.H; s = 0; z = self._ep(r["episode"]); obj = r["object"]
        pos = z["obj_pos"][s:s+H].astype(np.float64); quat = z["obj_quat"][s:s+H].astype(np.float64)
        base_pos = z["base_pos"][s].astype(np.float64); base_quat = z["base_quat"][s].astype(np.float64)
        ctrl = z["ctrl"][s:s+H].astype(np.float32)
        Rg = quat_to_R(base_quat); R0 = quat_to_R(quat[0]); p0 = pos[0]
        Rt = quat_to_R(quat); dR = Rt @ R0.T; dp = pos - p0
        tgt = np.concatenate([(Rg.T @ dp[..., None])[..., 0], R_to_6d(Rg.T @ dR @ Rg)], -1).astype(np.float32)
        base_rel = np.concatenate([R0.T @ (base_pos - p0), R_to_6d(R0.T @ Rg)]).astype(np.float32)
        feats = self._cl(obj); rng = np.random.default_rng(7000+i) if self.fixed else self.rng
        idx = rng.choice(len(feats), self.n, replace=len(feats) < self.n)
        if self.stats is not None: tgt = np.clip((tgt - self.stats["mean"]) / self.stats["std"], -8.0, 8.0)   # clip heavy tail (187 has 20σ close+settle outliers)
        return dict(pts=torch.from_numpy(feats[idx]), base_rel=torch.from_numpy(base_rel), closing=torch.from_numpy(ctrl),
                    table=torch.zeros(1), target=torch.from_numpy(tgt), weight=torch.tensor(float(r["weight"]), dtype=torch.float32),
                    tilt=torch.tensor(float(r["net_tilt"]), dtype=torch.float32), object=obj, shape=r["shape"], episode=r["episode"])


def compute_stats(rows, H=C.H, n=2500, seed=0):
    ds = ChunkDS(rows, H); rng = np.random.default_rng(seed); sel = rng.choice(len(ds), min(n, len(ds)), replace=False)
    T = np.stack([ds[i]["target"].numpy() for i in sel]).reshape(-1, 9)
    return {"mean": T.mean(0).astype(np.float32), "std": (T.std(0)+1e-4).astype(np.float32)}
