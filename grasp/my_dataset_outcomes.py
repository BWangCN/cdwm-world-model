"""Grasp-OUTCOME dataset (outcomes_v2). Per episode: t=0 cloud through our `local` gripper-frame+covariance features
(reuse build_feats) + grasp pose base_rel -> labels (5-way v2_outcome + 2-tier). Object-disjoint via outcome_split.csv.

Efficient: t=0 poses precomputed ONCE from the trajectory shards (cache t0_poses.npz); per-object cloud covariance
precomputed once (reuse my_dataset_hf_local). modality: 'full' (real cloud+grasp) | 'pose_only' (cloud->train-mean, keeps
grasp; Codex: mean not zeros to avoid OOD). Pairs with wm.classifier.OutcomeNet.

    CDWM_MODALITY=full python train_cls.py --split ... ; controls via CDWM_MODALITY=pose_only
"""
import os, sys, csv, collections
import numpy as np, torch
from torch.utils.data import Dataset
from common import my_config as C
from common.utils import quat_to_R, R_to_6d
from grasp.my_dataset_hf_local import build_feats, _quat_to_R_batch, PPZ

from common.paths import OUTCOMES as OV
FIVE = {"RIGID": 0, "TRANSIENT_SLIP": 1, "PERSISTENT_SLIP": 2, "LIFTED_DROPPED": 3, "CLOSED_NEVER_LIFTED": 4}
TIER = {"SUCCESS": 0, "FAILURE": 1}
_T0 = os.path.join(OV, "t0_poses.npz")
_STATS = os.path.join(OV, "feat_stats_ov_local.npz")


def _index():
    rows = [r for r in csv.DictReader(open(f"{OV}/outcomes_index.csv")) if r["v2_outcome"] in FIVE]   # drop 23 CLEARANCE_VIOLATION
    if os.environ.get("CDWM_TASK") == "lifted":               # Codex: given-it-lifted mode test -> drop pose-predictable never-lifted
        rows = [r for r in rows if r["v2_outcome"] != "CLOSED_NEVER_LIFTED"]
    return rows


def _splits():
    return {r["object_id"]: r["split"] for r in csv.DictReader(open(f"{OV}/outcome_split.csv"))}


def _all_rows():                                              # canonical FULL index (order-stable; NOT filtered by task)
    return list(csv.DictReader(open(f"{OV}/outcomes_index.csv")))


def _ensure_t0():                                             # t=0 poses for ALL episodes, uid-keyed (robust to task filtering)
    allr = _all_rows(); uids = np.array([r["uid"] for r in allr])
    if os.path.exists(_T0):
        z = np.load(_T0, allow_pickle=True)
        return {k: z[k] for k in ("obj_pos", "obj_quat", "base_pos", "base_quat")}, {u: i for i, u in enumerate(z["uid"])}
    byshard = collections.defaultdict(list)
    for i, r in enumerate(allr): byshard[r["shard"]].append(i)
    op = np.zeros((len(allr), 3), np.float32); oq = np.zeros((len(allr), 4), np.float32)
    bp = np.zeros((len(allr), 3), np.float32); bq = np.zeros((len(allr), 4), np.float32)
    for shard, ii in byshard.items():
        z = np.load(f"{OV}/{shard}", mmap_mode="r"); fs = np.array([int(allr[i]["frame_start"]) for i in ii])
        op[ii] = z["obj_pos"][fs]; oq[ii] = z["obj_quat"][fs]; bp[ii] = z["base_pos"][fs]; bq[ii] = z["base_quat"][fs]
    np.savez(_T0, obj_pos=op, obj_quat=oq, base_pos=bp, base_quat=bq, uid=uids); print(f"[t0] cached {_T0}")
    return {"obj_pos": op, "obj_quat": oq, "base_pos": bp, "base_quat": bq}, {u: i for i, u in enumerate(uids)}


_CAP = 8192                                                     # pre-subsample per object -> bounds per-worker memory (Codex)


def _raw_cloud(obj):                                            # PRE-subsampled to <=8192 pts + covariance precomputed once
    g = np.load(f"{OV}/point_clouds/{obj}.npz"); N = len(g["mu"])
    sel = np.random.default_rng(0).choice(N, min(_CAP, N), replace=False)   # deterministic across workers
    ic = g["is_completed"][sel].astype(np.float32).reshape(-1, 1)
    Rq = _quat_to_R_batch(g["rot_quat"][sel].astype(np.float64))
    Sig = np.einsum("nij,nj,nkj->nik", Rq, g["scale"][sel].astype(np.float64) ** 2, Rq).astype(np.float32)
    return dict(mu=g["mu"][sel].astype(np.float32), Sig_obj=Sig, opacity=g["opacity"][sel].astype(np.float32).reshape(-1, 1), ic=ic)


def _ensure_stats(sp, t0, uid2i):
    if os.path.exists(_STATS): d = np.load(_STATS); return d["mean"], d["std"]
    rng = np.random.default_rng(0)
    tr = [r for r in _all_rows() if sp.get(r["object_id"]) == "train"]
    sel = [tr[i] for i in rng.choice(len(tr), min(2000, len(tr)), replace=False)]; acc = []
    cache = {}
    for r in sel:
        o = r["object_id"]; gi = uid2i[r["uid"]]
        if o not in cache: cache[o] = _raw_cloud(o)
        rc = cache[o]; R0 = quat_to_R(t0["obj_quat"][gi].astype(np.float64)); p0 = t0["obj_pos"][gi].astype(np.float64)
        Rg = quat_to_R(t0["base_quat"][gi].astype(np.float64)); bp = t0["base_pos"][gi].astype(np.float64)
        idx = rng.choice(len(rc["mu"]), min(4096, len(rc["mu"])), replace=False)
        f, _ = build_feats(rc["mu"][idx], rc["Sig_obj"][idx], rc["opacity"][idx], rc["ic"][idx], R0, p0, Rg, bp)
        acc.append(f)
    A = np.concatenate(acc); mean = A.mean(0).astype(np.float32); std = (A.std(0) + 1e-6).astype(np.float32)
    np.savez(_STATS, mean=mean, std=std); print(f"[stats] cached {_STATS}")
    return mean, std


class OutcomeDS(Dataset):
    def __init__(self, split, modality="full", n_points=4096, seed=0):
        allrows = _index(); sp = _splits()
        so = {s: {r["object_id"] for r in allrows if sp.get(r["object_id"]) == s} for s in ("train", "val", "test")}
        assert so["train"].isdisjoint(so["val"]) and so["train"].isdisjoint(so["test"]) and so["val"].isdisjoint(so["test"]), "split overlap!"
        assert all(r["object_id"] in sp for r in allrows), "episode object missing from split manifest"
        self.rows = [r for r in allrows if sp.get(r["object_id"]) == split]
        self.mod, self.n = modality, n_points
        self.t0, uid2i = _ensure_t0()
        self.gi = [uid2i[r["uid"]] for r in self.rows]                # uid -> canonical t0 index (robust to task filtering)
        self.fmean, self.fstd = _ensure_stats(sp, self.t0, uid2i)
        self.clc, self.rng, self.fixed = {}, np.random.default_rng(seed), (split != "train")

    def _cl(self, o):
        if o not in self.clc: self.clc[o] = _raw_cloud(o)
        return self.clc[o]

    def __len__(self): return len(self.rows)

    def labels(self):                                          # for class-balanced sampler/weights
        return np.array([FIVE[r["v2_outcome"]] for r in self.rows]), np.array([TIER[r["tier"]] for r in self.rows])

    def __getitem__(self, j):
        r = self.rows[j]; gi = self.gi[j]; o = r["object_id"]; rc = self._cl(o)
        R0 = quat_to_R(self.t0["obj_quat"][gi].astype(np.float64)); p0 = self.t0["obj_pos"][gi].astype(np.float64)
        Rg = quat_to_R(self.t0["base_quat"][gi].astype(np.float64)); bp = self.t0["base_pos"][gi].astype(np.float64)
        base_rel = np.concatenate([R0.T @ (bp - p0), R_to_6d(R0.T @ Rg)]).astype(np.float32)
        rng = np.random.default_rng(7000 + gi) if self.fixed else self.rng
        idx = rng.choice(len(rc["mu"]), self.n, replace=len(rc["mu"]) < self.n)
        feat, dperp = build_feats(rc["mu"][idx], rc["Sig_obj"][idx], rc["opacity"][idx], rc["ic"][idx], R0, p0, Rg, bp)
        feat = (feat - self.fmean) / self.fstd
        if self.mod == "pose_only":                                      # neutralize the ENTIRE cloud path (feat AND the d_perp pool weights); keeps base_rel
            feat = np.zeros_like(feat); dperp = np.zeros_like(dperp)
        pts = np.concatenate([feat, dperp[:, None]], 1).astype(np.float32)
        return dict(pts=torch.from_numpy(pts), base_rel=torch.from_numpy(base_rel),
                    closing=torch.zeros(C.H, dtype=torch.float32), table=torch.zeros(1),
                    y5=FIVE[r["v2_outcome"]], y2=TIER[r["tier"]], object=o)
