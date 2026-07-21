"""Grasp-LOCAL HF dataset (Lever 1). Same grasps/splits/weights/target as my_dataset_hf, but each gaussian is expressed
in the GRIPPER frame with a sign/axis-unambiguous COVARIANCE and contact-geometry features. Pairs with wm.dit_local.DiTWMLocal.

Per-gaussian 13 point features (then z-scored by feat_stats_hf_local.npz) + 1 raw channel for pooling weights:
  [ mu_g(3)  |  Sigma_g upper-tri(6): xx,yy,zz,xy,xz,yz  |  opacity(1)  |  is_completed(1)  |  d_par(1)  |  d_perp(1) ]  (13)
  [ raw d_perp(1) ]  -> channel 14, used by the encoder's contact-weighted pool (unstandardized).
where, in the gripper frame (verified constant across all objects): pinch point PP=(0,0,0.1159), jaws close along CAXIS=+x.
  d_par  = mu_g.x                       signed position along the closing axis (fingers contact at +/-x extremes)
  d_perp = ||(mu_g.y, mu_g.z-PPz)||     distance from the pinch line (small = inside the grasp corridor)
Covariance is rotated into the gripper frame: Sigma_g = M Sigma_obj M^T, M = Rg^T R0, Sigma_obj = Rq diag(scale^2) Rq^T.

    python train.py --dataset_mod my_dataset_hf_local --arch dit_local --tag local ...
"""
import os, sys, json
import numpy as np, torch
from torch.utils.data import Dataset
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import my_config as C
from wm.common import quat_to_R
from my_dataset_hf import HF, _MANI, _SHAPE, _CTRL, _band, load_rows   # reuse split/rows/weights logic

PPZ = 0.1159                                              # pinch-point z in gripper frame (probed: std ~1e-4 across objects)
_LOCAL_STATS = os.path.join(C.RUN, "feat_stats_hf_local.npz")


def _quat_to_R_batch(q):                                 # (N,4 wxyz)->(N,3,3)
    return quat_to_R(q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12))


def build_feats(mu, Sig_obj, opacity, ic, R0, p0, Rg, bp):   # Sig_obj = object-frame covariance, PRECOMPUTED per object
    mu_g = (p0 + mu @ R0.T - bp) @ Rg                     # (N,3) gripper-frame centers (mirrors my_dataset_frame)
    M = Rg.T @ R0                                         # object->gripper rotation (col-vec)
    Sig_obj = Sig_obj.astype(np.float64)
    Sig_g = np.einsum("ij,njk,lk->nil", M, Sig_obj, M)   # M Sig_obj M^T
    tri = np.stack([Sig_g[:, 0, 0], Sig_g[:, 1, 1], Sig_g[:, 2, 2],
                    Sig_g[:, 0, 1], Sig_g[:, 0, 2], Sig_g[:, 1, 2]], 1)             # (N,6) upper-tri
    d_par = mu_g[:, 0:1]                                  # along +x closing axis
    d_perp = np.sqrt(mu_g[:, 1] ** 2 + (mu_g[:, 2] - PPZ) ** 2)[:, None]            # dist to pinch line
    feat = np.concatenate([mu_g, tri, opacity, ic, d_par, d_perp], 1).astype(np.float32)  # (N,13)
    return feat, d_perp[:, 0].astype(np.float32)


def _raw_cloud(obj):                                     # covariance precomputed ONCE per object (was recomputed per sample)
    g = np.load(f"{HF}/objects/{obj}/point_cloud.npz")
    ic = g["is_completed"].astype(np.float32).reshape(-1, 1)
    Rq = _quat_to_R_batch(g["rot_quat"].astype(np.float64))
    Sig_obj = np.einsum("nij,nj,nkj->nik", Rq, g["scale"].astype(np.float64) ** 2, Rq).astype(np.float32)  # Rq diag(s^2) Rq^T
    return dict(mu=g["mu"].astype(np.float32), Sig_obj=Sig_obj,
                opacity=g["opacity"].astype(np.float32).reshape(-1, 1), ic=ic)


def _ensure_stats(seed=0, n_obj=60, n_pts=3000):
    if os.path.exists(_LOCAL_STATS): return np.load(_LOCAL_STATS)
    rows = load_rows("train"); rng = np.random.default_rng(seed)
    objs = sorted({r["object"] for r in rows}); objs = [objs[i] for i in rng.choice(len(objs), min(n_obj, len(objs)), replace=False)]
    acc = []
    for obj in objs:
        g = np.load(f"{HF}/objects/{obj}/grasps.npz", allow_pickle=True); rc = _raw_cloud(obj)
        for k in rng.choice(len(g["target"]), min(4, len(g["target"])), replace=False):
            p0 = g["obj_pos"][k, 0].astype(np.float64); R0 = quat_to_R(g["obj_quat"][k, 0].astype(np.float64))
            Rg = quat_to_R(g["base_quat"][k].astype(np.float64)); bp = g["base_pos"][k].astype(np.float64)
            idx = rng.choice(len(rc["mu"]), min(n_pts, len(rc["mu"])), replace=False)
            f, _ = build_feats(rc["mu"][idx], rc["Sig_obj"][idx], rc["opacity"][idx], rc["ic"][idx], R0, p0, Rg, bp)
            acc.append(f)
    A = np.concatenate(acc); mean = A.mean(0).astype(np.float32); std = (A.std(0) + 1e-6).astype(np.float32)
    np.savez(_LOCAL_STATS, mean=mean, std=std); print(f"[local stats] wrote {_LOCAL_STATS} from {len(A)} pts")
    return np.load(_LOCAL_STATS)


class ChunkDS(Dataset):
    def __init__(self, rows, H=C.H, n_points=4096, stats=None, fixed_subsample=False, seed=0):
        self.rows, self.H, self.n, self.stats, self.fixed = rows, H, n_points, stats, fixed_subsample
        self.gc, self.clc, self.rng = {}, {}, np.random.default_rng(seed)
        fs = _ensure_stats(); self.fmean, self.fstd = fs["mean"], fs["std"]

    def _g(self, obj):
        if obj not in self.gc: self.gc[obj] = dict(np.load(f"{HF}/objects/{obj}/grasps.npz", allow_pickle=True))
        return self.gc[obj]

    def _cl(self, obj):
        if obj not in self.clc: self.clc[obj] = _raw_cloud(obj)
        return self.clc[obj]

    def __len__(self): return len(self.rows)
    def weights(self): return np.array([float(r["weight"]) for r in self.rows], np.float64)

    def __getitem__(self, i):
        r = self.rows[i]; obj = r["object"]; k = r["k"]; g = self._g(obj); rc = self._cl(obj)
        tgt = g["target"][k][:self.H].astype(np.float32); base_rel = g["base_rel"][k].astype(np.float32)
        p0 = g["obj_pos"][k, 0].astype(np.float64); R0 = quat_to_R(g["obj_quat"][k, 0].astype(np.float64))
        Rg = quat_to_R(g["base_quat"][k].astype(np.float64)); bp = g["base_pos"][k].astype(np.float64)
        rng = np.random.default_rng(7000 + i) if self.fixed else self.rng
        idx = rng.choice(len(rc["mu"]), self.n, replace=len(rc["mu"]) < self.n)
        feat, dperp = build_feats(rc["mu"][idx], rc["Sig_obj"][idx], rc["opacity"][idx], rc["ic"][idx], R0, p0, Rg, bp)
        feat = (feat - self.fmean) / self.fstd
        pts = np.concatenate([feat, dperp[:, None]], 1).astype(np.float32)          # (N,14): [13 std feats | raw d_perp]
        if self.stats is not None: tgt = np.clip((tgt - self.stats["mean"]) / self.stats["std"], -8.0, 8.0)
        return dict(pts=torch.from_numpy(pts), base_rel=torch.from_numpy(base_rel),
                    closing=torch.from_numpy(_CTRL[:self.H].copy()), table=torch.zeros(1),
                    target=torch.from_numpy(tgt), weight=torch.tensor(float(r["weight"]), dtype=torch.float32),
                    tilt=torch.tensor(float(r["net_tilt"]), dtype=torch.float32),
                    object=obj, shape=r["shape"], episode=r["episode"])


def compute_stats(rows, H=C.H, n=2500, seed=0):          # target z-score (frame-independent -> identical to my_dataset_hf)
    import my_dataset_hf as base
    return base.compute_stats(rows, H, n, seed)
