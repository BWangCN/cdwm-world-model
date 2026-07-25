"""Baseline drop-WM dataset. Per episode: object cloud (colorless-12) expressed in the RELEASE frame
(gravity-aligned, the drop analogue of the grasp gripper frame) + release params -> outcome labels
(basin_transition, validity) + rotation targets (net_rot, rest orientation rel. release).

Reuses the slip encoder verbatim: returns pts(N,14)=[13 feats | pool weight], base_rel(9), closing(zeros H),
table(zeros 1) so `OutcomeNet(n_feat=13)` / `DiTWMLocal.encode` consume it unchanged. Poses + labels come from
precompute_drop.py ($CDWM_DROP/derived). modality: 'full' (cloud+release) | 'pose_only' (cloud->0, release kept).
frame: 'release' (cloud rotated into the release orientation) | 'obj' (object-canonical; the weak base).

    CDWM_MODALITY=full python train_drop.py --tag full
"""
import os, sys, csv, collections
import numpy as np, torch
from torch.utils.data import Dataset
from common import my_config as C
from common.utils import quat_to_R, R_to_6d

from common.paths import DROP_CORPUS as DROP, PCDIR, DERIVED
_IDX, _POSES = f"{DERIVED}/drop_index.csv", f"{DERIVED}/drop_poses.npz"
_STATS = f"{DERIVED}/drop_feat_stats.npz"
_CAP = 8192


def _quat_to_R_batch(q):                                   # (N,4) wxyz -> (N,3,3)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3), np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - z * w); R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w); R[:, 2, 1] = 2 * (y * z + x * w); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _raw_cloud(obj):                                       # pre-subsampled + covariance precomputed once
    g = np.load(f"{PCDIR}/{obj}.npz"); N = len(g["mu"])
    sel = np.random.default_rng(0).choice(N, min(_CAP, N), replace=False)
    ic = g["is_completed"][sel].astype(np.float32).reshape(-1, 1)
    Rq = _quat_to_R_batch(g["rot_quat"][sel].astype(np.float64))
    Sig = np.einsum("nij,nj,nkj->nik", Rq, g["scale"][sel].astype(np.float64) ** 2, Rq).astype(np.float32)
    return dict(mu=g["mu"][sel].astype(np.float32), Sig_obj=Sig,
                opacity=g["opacity"][sel].astype(np.float32).reshape(-1, 1), ic=ic)


def build_feats_drop(mu, Sig_obj, opacity, ic, R_rel, frame="release"):
    """Cloud in the gravity-aligned release frame. Returns (feat[N,13], pool_weight[N]).
    feat = [mu_r(3) | Sig_tri(6) | opacity(1) | ic(1) | height_above_lowest(1) | horiz_radius(1)].
    Contact analogue of the gripper d_perp = height above the object's lowest point (the support patch)."""
    Rf = (R_rel if frame == "release" else np.eye(3)).astype(np.float32)   # float32: covariance rotation is the hot path
    c = mu.mean(0)
    mu_r = (mu - c) @ Rf.T                                  # centered, rotated into release orientation (f32)
    Sig_r = np.einsum("ij,njk,lk->nil", Rf, Sig_obj, Rf)   # Sig_obj already f32
    tri = np.stack([Sig_r[:, 0, 0], Sig_r[:, 1, 1], Sig_r[:, 2, 2],
                    Sig_r[:, 0, 1], Sig_r[:, 0, 2], Sig_r[:, 1, 2]], 1)
    height = (mu_r[:, 2] - mu_r[:, 2].min())[:, None]       # above lowest point (table contact) -> pool weight
    hr = np.sqrt(mu_r[:, 0] ** 2 + mu_r[:, 1] ** 2)[:, None]  # horizontal radius from vertical axis (lever arm)
    feat = np.concatenate([mu_r, tri, opacity, ic, height, hr], 1).astype(np.float32)
    return feat, height[:, 0].astype(np.float32)


def _all_rows():
    return list(csv.DictReader(open(_IDX)))


def _ensure_stats(rows, poses, frame):
    key = _STATS.replace(".npz", f"_{frame}.npz")
    if os.path.exists(key): d = np.load(key); return d["mean"], d["std"]
    rng = np.random.default_rng(0)
    tr = [(i, r) for i, r in enumerate(rows) if r["split"] == "train" and r["has_cloud"] == "True"]
    sel = [tr[i] for i in rng.choice(len(tr), min(2000, len(tr)), replace=False)]
    cache, acc = {}, []
    for gi, r in sel:
        o = r["object"]
        if o not in cache: cache[o] = _raw_cloud(o)
        rc = cache[o]; R_rel = quat_to_R(poses["release_quat"][gi].astype(np.float64))
        idx = rng.choice(len(rc["mu"]), min(4096, len(rc["mu"])), replace=False)
        f, _ = build_feats_drop(rc["mu"][idx], rc["Sig_obj"][idx], rc["opacity"][idx], rc["ic"][idx], R_rel, frame)
        acc.append(f)
    A = np.concatenate(acc); mean = A.mean(0).astype(np.float32); std = (A.std(0) + 1e-6).astype(np.float32)
    np.savez(key, mean=mean, std=std); print(f"[stats] cached {key}")
    return mean, std


class DropDS(Dataset):
    def __init__(self, split, modality="full", frame="release", n_points=4096, settled_only=True, seed=0):
        allrows = _all_rows()
        _p = np.load(_POSES); self.poses = {k: _p[k] for k in _p.files}   # eager: NpzFile is not fork-safe across workers
        rows = [(i, r) for i, r in enumerate(allrows)
                if r["split"] == split and r["has_cloud"] == "True"
                and (not settled_only or r["valid_training"] == "True")]
        self.gi = np.array([i for i, _ in rows]); self.rows = [r for _, r in rows]
        self.mod, self.frame, self.n = modality, frame, n_points
        self.fmean, self.fstd = _ensure_stats(allrows, self.poses, frame)
        self.clc, self.rng, self.fixed = {}, np.random.default_rng(seed), (split != "train")
        # precompute per-episode rotations/targets ONCE (Codex: keep quat_to_R/R_to_6d out of the hot path)
        rq = self.poses["release_quat"][self.gi].astype(np.float64)
        R_rel = quat_to_R(rq); R_rest = quat_to_R(self.poses["rest_quat"][self.gi].astype(np.float64))
        self.R_rel = R_rel.astype(np.float32)
        dR = np.einsum("nij,nkj->nik", R_rest, R_rel)          # R_rest @ R_rel^T (world reorientation)
        self.rest6d = R_to_6d(dR).astype(np.float32)
        dh = np.array([float(r["drop_h"]) for r in self.rows], np.float32)[:, None]
        tl = (np.array([float(r["tilt_deg"]) for r in self.rows], np.float32) / 15.0)[:, None]
        self.base_rel = np.concatenate([dh, tl, R_to_6d(R_rel).astype(np.float32),
                                        np.zeros((len(self.rows), 1), np.float32)], 1)
        self.y_bt = np.array([int(r["basin_transition"] == "True") for r in self.rows])
        self.netrot = np.array([float(r["net_rot_from_release_deg"]) for r in self.rows], np.float32)

    def _cl(self, o):
        if o not in self.clc: self.clc[o] = _raw_cloud(o)
        return self.clc[o]

    def __len__(self): return len(self.rows)

    def labels(self):                                      # for class-balanced sampler
        return self.y_bt

    def __getitem__(self, j):
        o = self.rows[j]["object"]; rc = self._cl(o)
        rng = np.random.default_rng(7000 + int(self.gi[j])) if self.fixed else self.rng
        idx = rng.choice(len(rc["mu"]), self.n, replace=len(rc["mu"]) < self.n)
        feat, w = build_feats_drop(rc["mu"][idx], rc["Sig_obj"][idx], rc["opacity"][idx], rc["ic"][idx],
                                   self.R_rel[j], self.frame)
        feat = (feat - self.fmean) / self.fstd
        if self.mod == "pose_only": feat = np.zeros_like(feat); w = np.zeros_like(w)
        pts = np.concatenate([feat, w[:, None]], 1).astype(np.float32)
        return dict(pts=torch.from_numpy(pts), base_rel=torch.from_numpy(self.base_rel[j]),
                    closing=torch.zeros(C.H, dtype=torch.float32), table=torch.zeros(1),
                    y_bt=int(self.y_bt[j]),
                    netrot=torch.tensor(float(self.netrot[j]), dtype=torch.float32),
                    rest6d=torch.from_numpy(self.rest6d[j]), object=o)


class DropTrajDS(DropDS):
    """Trajectory-supervision arm: same features, target = K=8 anchor orientations (rel. release, 6d) spanning
    contact-onset -> settle (precompute_droptraj.py; during-fall frames carry zero signal under zero-velocity
    kinematic release). Anchor K-1 = the final frame = the resting pose, so the endpoint readout is comparable
    to DropDS.rest6d. All targets precomputed here; the __getitem__ hot path is unchanged."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        t = np.load(f"{DERIVED}/drop_traj8.npz")
        rq_chk = t["release_quat"][self.gi].astype(np.float64)
        rq_ref = self.poses["release_quat"][self.gi].astype(np.float64)
        assert np.abs((rq_chk * rq_ref).sum(1)).min() > 0.999, "drop_traj8 rows misaligned with drop_poses"
        tq = t["traj_quat"][self.gi].astype(np.float64)             # (n,K,4)
        n, K = tq.shape[:2]
        Rk = _quat_to_R_batch(tq.reshape(-1, 4)).reshape(n, K, 3, 3)
        dR = np.einsum("nkij,nlj->nkil", Rk, self.R_rel.astype(np.float64))   # R_k @ R_rel^T per anchor
        self.traj6d = R_to_6d(dR.reshape(-1, 3, 3)).reshape(n, K, 6).astype(np.float32)
        self.K = K

    def __getitem__(self, j):
        d = super().__getitem__(j)
        d["traj6d"] = torch.from_numpy(self.traj6d[j])
        return d
