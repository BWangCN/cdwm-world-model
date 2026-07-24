"""Gate B dataset: join each near-boundary/hidden-CoM episode (gateb/<obj>_gateb_s0.npz) to its object point
cloud. Input = release-frame cloud (reuse build_feats_drop) + release params; the CoM is the HIDDEN latent
(returned separately, used only by the oracle arm). Target = resting orientation rel. release (6D, padded to 9
to reuse DiTWM). Object-disjoint split. Excludes the hammer set (elpis-hull geometry != the WM cloud).

    from drop.my_dataset_gateb import GateBDS
"""
import os, sys, glob
import numpy as np, torch
from torch.utils.data import Dataset
from common import my_config as C
from common.utils import quat_to_R, R_to_6d
from drop.my_dataset_drop import _raw_cloud, build_feats_drop, DERIVED, PCDIR

HERE = os.path.dirname(os.path.abspath(__file__))
GATEB = os.path.join(HERE, "gateb")
_STATS = os.path.join(GATEB, "gateb_target_stats.npz")
_FEAT_STATS = os.path.join(DERIVED, "drop_feat_stats_release.npz")   # reuse the drop release-frame cloud stats


def _load_all():
    OBJ, RELQ, DH, COM, RESTQ, BAS = [], [], [], [], [], []
    for f in sorted(glob.glob(f"{GATEB}/*_gateb_s0.npz")):
        if "hammer" in os.path.basename(f): continue          # elpis-hull geometry, excluded
        z = np.load(f, allow_pickle=True); o = str(z["object"]); n = len(z["release_quat"])
        OBJ += [o] * n; RELQ.append(z["release_quat"]); DH.append(z["drop_h"])
        COM.append(z["com"].astype(np.float32)); RESTQ.append(z["rest_quat"]); BAS.append(z["basin"])
    return (np.array(OBJ), np.concatenate(RELQ), np.concatenate(DH),
            np.concatenate(COM), np.concatenate(RESTQ), np.concatenate(BAS))


def object_split(objs, seed=0):                               # object-disjoint 6/2/2 of the 10
    uo = sorted(set(objs)); rng = np.random.default_rng(seed); rng.shuffle(uo)
    ntr, nva = int(0.6 * len(uo)), int(0.2 * len(uo))
    return {o: ("train" if i < ntr else "val" if i < ntr + nva else "test") for i, o in enumerate(uo)}


SPLIT_MODE = os.environ.get("CDWM_GATEB_SPLIT", "object")      # object | within | reldisjoint


def _mask_for(OBJ, RELQ, split):
    if SPLIT_MODE == "object":                                # object-disjoint (hardest: unseen geometry)
        sp = object_split(OBJ); return np.array([sp[o] == split for o in OBJ])
    if SPLIT_MODE == "within":                                # per-episode random (has release-neighborhood leakage)
        r = np.random.default_rng(0).random(len(OBJ))
        return np.where(r < 0.7, "train", np.where(r < 0.85, "val", "test")) == split
    # reldisjoint: hold out coarse (object, release-orientation cell) NEIGHBORHOODS -> known objects, UNSEEN release regions
    R6 = R_to_6d(quat_to_R(RELQ.astype(np.float64)))          # (N,6) release orientation
    cells = [f"{o}|{tuple(c)}" for o, c in zip(OBJ, np.round(R6 * 2).astype(int))]   # ~30deg cells
    uc = sorted(set(cells)); rng = np.random.default_rng(0); rng.shuffle(uc)
    ntr, nva = int(0.7 * len(uc)), int(0.15 * len(uc))
    cs = {c: ("train" if i < ntr else "val" if i < ntr + nva else "test") for i, c in enumerate(uc)}
    return np.array([cs[c] == split for c in cells])


class GateBDS(Dataset):
    def __init__(self, split, n_points=4096, stats=None, seed=0):
        OBJ, RELQ, DH, COM, RESTQ, BAS = _load_all()
        m = _mask_for(OBJ, RELQ, split)
        self.obj = OBJ[m]; self.dh = DH[m].astype(np.float32); self.com = COM[m]; self.basin = BAS[m]
        Rrel = quat_to_R(RELQ[m].astype(np.float64)); Rrest = quat_to_R(RESTQ[m].astype(np.float64))
        self.R_rel = Rrel.astype(np.float32)
        dR = np.einsum("nij,nkj->nik", Rrest, Rrel)           # resting orientation rel. release
        rest6d = R_to_6d(dR).astype(np.float32)
        self.target = np.concatenate([rest6d, np.zeros((len(rest6d), 3), np.float32)], 1)   # 9d (pad rot->reuse DiTWM)
        self.base_rel = np.concatenate([self.dh[:, None], R_to_6d(self.R_rel).astype(np.float32),
                                        np.zeros((len(self.obj), 2), np.float32)], 1)        # (N,9)
        # latent (oracle only): [delta_scaled, onehot(axis,3)]
        ax = self.com[:, 0].astype(int); dl = self.com[:, 1].astype(np.float32)
        oh = np.eye(3, dtype=np.float32)[np.clip(ax, 0, 2)]
        self.latent = np.concatenate([(dl * 20.0)[:, None], oh], 1).astype(np.float32)       # (N,4)
        self.n = n_points; self.clc = {}; self.rng = np.random.default_rng(seed); self.fixed = (split != "train")
        self.stats = stats if stats is not None else self._compute_stats()
        _fs = np.load(_FEAT_STATS); self.fmean, self.fstd = _fs["mean"], _fs["std"]
        # cross-object transfer study: how the hidden CoM is fed. none|abstract (base_rel) | grounded|shuffled (per-point)
        self.latent_mode = os.environ.get("CDWM_GATEB_LATENT", "abstract")
        self._hc = {}                                          # per-object hull (center, principal axes) to locate the CoM
        self.com_shuf = self.com[np.random.default_rng(1).permutation(len(self.com))]   # wrong-CoM control (shuffled)

    def _cl(self, o):
        if o not in self.clc: self.clc[o] = _raw_cloud(o)
        return self.clc[o]

    def _hull(self, o):                                        # match generate_obj/drop_sweep.hull (subsample seed 0)
        if o in self._hc: return self._hc[o]
        from scipy.spatial import ConvexHull
        mu = np.load(f"{PCDIR}/{o}.npz")["mu"].astype(np.float64)
        if len(mu) > 4000: mu = mu[np.random.default_rng(0).choice(len(mu), 4000, replace=False)]
        V = mu[ConvexHull(mu).vertices]; hc = V.mean(0); _, _, vt = np.linalg.svd(V - hc)
        self._hc[o] = (hc.astype(np.float32), vt.astype(np.float32)); return self._hc[o]

    def _compute_stats(self):
        _s = os.path.join(GATEB, f"gateb_target_stats_{SPLIT_MODE}.npz")
        if os.path.exists(_s): d = np.load(_s); return {"mean": d["mean"], "std": d["std"]}
        sel = self.rng.choice(len(self.target), min(4000, len(self.target)), replace=False)
        T = self.target[sel]; st = {"mean": T.mean(0).astype(np.float32), "std": (T.std(0) + 1e-4).astype(np.float32)}
        np.savez(_s, **st); return st

    def __len__(self): return len(self.obj)

    def __getitem__(self, j):
        o = self.obj[j]; rc = self._cl(o)
        rng = np.random.default_rng(7000 + j) if self.fixed else self.rng
        idx = rng.choice(len(rc["mu"]), self.n, replace=len(rc["mu"]) < self.n); mu = rc["mu"][idx]
        feat, w = build_feats_drop(mu, rc["Sig_obj"][idx], rc["opacity"][idx], rc["ic"][idx], self.R_rel[j], "release")
        feat = (feat - self.fmean) / self.fstd               # standardize cloud feats (drop release-frame stats)
        parts = [feat]
        if self.latent_mode in ("grounded", "shuffled"):     # geometry-grounded CoM: per-point vector-to-CoM (3) + dist (1)
            com = self.com_shuf[j] if self.latent_mode == "shuffled" else self.com[j]
            hc, axes = self._hull(o); com_obj = hc + float(com[1]) * axes[int(com[0])]
            c = mu.mean(0); Rf = self.R_rel[j]
            mu_r = (mu - c) @ Rf.T; com_r = (com_obj - c) @ Rf.T
            vec = com_r[None, :] - mu_r; dist = np.linalg.norm(vec, axis=1, keepdims=True)
            parts.append(np.concatenate([vec, dist], 1).astype(np.float32))
        pts = np.concatenate(parts + [w[:, None]], 1).astype(np.float32)
        tgt = np.clip((self.target[j] - self.stats["mean"]) / self.stats["std"], -8, 8).astype(np.float32)
        return dict(pts=torch.from_numpy(pts), base_rel=torch.from_numpy(self.base_rel[j]),
                    closing=torch.zeros(1, dtype=torch.float32), table=torch.zeros(1),
                    target=torch.from_numpy(tgt).unsqueeze(0),   # (1,9)
                    latent=torch.from_numpy(self.latent[j]),
                    basin=int(self.basin[j]), object=o)
