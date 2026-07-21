"""HF-sourced drop-in for my_dataset. Reconstructs the FULL 15,013-grasp trainable set directly from the published
BWangCN/cdwm-grasp-dataset (objects/<id>/grasps.npz + point_cloud.npz), so no raw settle episodes are needed.

Verified faithful: HF's precomputed target(K,32,9)/base_rel(K,9) equal my_dataset's recompute to ~1e-16, and the
model's `closing` input (gripper actuator ctrl) is a FIXED ramp (0->255 over 20 steps, then held) that is byte-identical
across all grasps -> a constant, reproduced here exactly. Same API as my_dataset: load_rows/ChunkDS/compute_stats, same
12-feat colorless representation, same per-channel feat_stats standardization, same target z-score + -+8 clamp.

    python train.py --dataset_mod my_dataset_hf --seed 0 --tag base_hf ...
"""
import os, sys, json
import numpy as np, torch
from torch.utils.data import Dataset
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import my_config as C

HF = os.environ.get("CDWM_HF", "/misc/kcgscratch1/MengyeGroup/yy5259/datasets/cdwm-grasp-dataset")
_MANI = json.load(open(f"{HF}/manifest.json"))
_SHAPE = {m["object_id"]: m["shape_class"] for m in _MANI}
_FS = np.load(f"{C.RUN}/feat_stats.npz"); FMEAN, FSTD = _FS["mean"], _FS["std"]   # per-channel 12-feat standardization
# commanded closing profile: verified identical for every grasp (0->255 ramp over 20 steps, held at 255). H=32.
_CTRL = np.array([0., 13.26, 26.52, 39.78, 53.04, 66.3, 79.56, 92.82, 106.08, 119.34, 132.6, 145.86, 159.12,
                  172.38, 185.64, 198.9, 212.16, 225.76, 239.02, 252.28, 255., 255., 255., 255., 255., 255.,
                  255., 255., 255., 255., 255., 255.], np.float32)  # exact ctrl[0:32]


def _band(t):
    for lo, hi, lab in [(0, 2, "0-2"), (2, 5, "2-5"), (5, 15, "5-15"), (15, 30, "15-30"), (30, 1e9, "30+")]:
        if lo <= t < hi: return lab
    return "30+"


def load_rows(split=None):
    """rows: one per grasp; split in {train, valgrasp, heldout_object}. train rows get inverse-band-freq weights
    (upsample rare high-tilt grasps, matching the colleague's WeightedRandomSampler intent)."""
    rows = []
    for m in _MANI:
        obj = m["object_id"]; gp = f"{HF}/objects/{obj}/grasps.npz"
        if not os.path.exists(gp) or not os.path.exists(f"{HF}/objects/{obj}/point_cloud.npz"): continue
        z = np.load(gp, allow_pickle=True); sp = z["split"].astype(str); nt = z["net_tilt_deg"].astype(np.float32)
        for k in range(len(sp)):
            if split is not None and sp[k] != split: continue
            rows.append(dict(object=obj, k=int(k), shape=_SHAPE.get(obj, "other"),
                             net_tilt=float(nt[k]), band=_band(float(nt[k])), split=sp[k],
                             episode=f"{obj}#{k}", weight=1.0))
    if split == "train" and rows:
        from collections import Counter
        c = Counter(r["band"] for r in rows); N = len(rows); K = len(c)
        for r in rows: r["weight"] = N / (K * c[r["band"]])   # inverse band frequency, mean ~1
    return rows


def cluster_feats(obj):
    g = np.load(f"{HF}/objects/{obj}/point_cloud.npz")
    ic = g["is_completed"].astype(np.float32).reshape(-1, 1)
    feat = np.concatenate([g["mu"], g["rot_quat"], g["scale"], g["opacity"].reshape(-1, 1), ic], 1).astype(np.float32)
    return ((feat - FMEAN) / FSTD).astype(np.float32)


class ChunkDS(Dataset):
    def __init__(self, rows, H=C.H, n_points=4096, stats=None, fixed_subsample=False, seed=0):
        self.rows, self.H, self.n, self.stats, self.fixed = rows, H, n_points, stats, fixed_subsample
        self.gc, self.clc, self.rng = {}, {}, np.random.default_rng(seed)

    def _g(self, obj):
        if obj not in self.gc: self.gc[obj] = dict(np.load(f"{HF}/objects/{obj}/grasps.npz", allow_pickle=True))
        return self.gc[obj]

    def _cl(self, obj):
        if obj not in self.clc: self.clc[obj] = cluster_feats(obj)
        return self.clc[obj]

    def __len__(self): return len(self.rows)
    def weights(self): return np.array([float(r["weight"]) for r in self.rows], np.float64)

    def __getitem__(self, i):
        r = self.rows[i]; obj = r["object"]; k = r["k"]; g = self._g(obj)
        tgt = g["target"][k][:self.H].astype(np.float32)          # (H,9) gripper-frame SE(3) delta (== my_dataset recompute)
        base_rel = g["base_rel"][k].astype(np.float32)            # (9,)
        feats = self._cl(obj); rng = np.random.default_rng(7000 + i) if self.fixed else self.rng
        idx = rng.choice(len(feats), self.n, replace=len(feats) < self.n)
        if self.stats is not None: tgt = np.clip((tgt - self.stats["mean"]) / self.stats["std"], -8.0, 8.0)
        return dict(pts=torch.from_numpy(feats[idx]), base_rel=torch.from_numpy(base_rel),
                    closing=torch.from_numpy(_CTRL[:self.H].copy()), table=torch.zeros(1),
                    target=torch.from_numpy(tgt), weight=torch.tensor(float(r["weight"]), dtype=torch.float32),
                    tilt=torch.tensor(float(r["net_tilt"]), dtype=torch.float32),
                    object=obj, shape=r["shape"], episode=r["episode"])


def compute_stats(rows, H=C.H, n=2500, seed=0):
    ds = ChunkDS(rows, H); rng = np.random.default_rng(seed); sel = rng.choice(len(ds), min(n, len(ds)), replace=False)
    T = np.stack([ds[i]["target"].numpy() for i in sel]).reshape(-1, 9)
    return {"mean": T.mean(0).astype(np.float32), "std": (T.std(0) + 1e-4).astype(np.float32)}
