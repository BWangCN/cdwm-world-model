"""CASE-A intervention dataset: the t=0 point cloud expressed in the GRIPPER frame (pinch axis = fixed coord axis).
IDENTICAL to my_dataset EXCEPT pts: each chunk transforms the object-frame cloud (mu + per-gaussian rot_quat) into the
gripper frame using the t=0 poses (R0,p0 object; Rg,base_pos gripper) — all t=0, time-invariant. Target / base_rel /
closing / compute_stats are UNCHANGED (the intervention is ONLY the cloud frame). Gripper-frame feat_stats (feat_stats_frame.npz)."""
import csv, os, sys
import numpy as np, torch
from torch.utils.data import Dataset
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import my_config as C
from wm.common import quat_to_R, R_to_6d
FRAME_STATS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feat_stats_frame.npz")


def load_rows(split=None):
    rows = list(csv.DictReader(open(C.CHUNKS_CSV)))
    rows = [r for r in rows if (split is None or r["split"] == split)]
    return [r for r in rows if os.path.exists(f"{C.CLUSTERS}/gaussians_{r['object']}_clean.npz")]


def cluster_raw(obj):                                    # RAW (unstandardized) 12-feat, object/reconstruction frame
    g = np.load(f"{C.CLUSTERS}/gaussians_{obj}_clean.npz"); ic = g["is_completed"].astype(np.float32)[:, None]
    return dict(mu=g["mu"].astype(np.float32), q=g["rot_quat"].astype(np.float32),
                so=np.concatenate([g["scale"], g["opacity"], ic], 1).astype(np.float32))   # scale/opacity/is_completed (frame-invariant)


def _mat2quat(R):                                        # (3,3)->wxyz
    t = np.trace(R)
    if t > 0: s = np.sqrt(t+1)*2; return np.array([0.25*s, (R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s])
    i = np.argmax([R[0,0], R[1,1], R[2,2]]);
    if i == 0: s = np.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2; return np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
    if i == 1: s = np.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2; return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s])
    s = np.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2; return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s])


def _qmul(a, b):                                         # a:(4,) wxyz applied to b:(N,4)
    aw, ax, ay, az = a
    return np.stack([aw*b[:,0]-ax*b[:,1]-ay*b[:,2]-az*b[:,3], aw*b[:,1]+ax*b[:,0]+ay*b[:,3]-az*b[:,2],
                     aw*b[:,2]-ax*b[:,3]+ay*b[:,0]+az*b[:,1], aw*b[:,3]+ax*b[:,2]-ay*b[:,1]+az*b[:,0]], 1)


def to_gripper(mu, q, so, R0, p0, Rg, bp):               # (already-subsampled) object-frame cloud -> gripper frame (t=0)
    mu_g = (p0 + mu @ R0.T - bp) @ Rg                     # positions:  Rg^T (p0 + R0 mu - bp)
    q_g = _qmul(_mat2quat(Rg.T @ R0).astype(np.float32), q)          # per-gaussian orientation rotated by object->gripper
    return np.concatenate([mu_g.astype(np.float32), q_g, so], 1)


_FS = np.load(FRAME_STATS) if os.path.exists(FRAME_STATS) else None
FMEAN = _FS["mean"] if _FS is not None else None; FSTD = _FS["std"] if _FS is not None else None


class ChunkDS(Dataset):
    def __init__(self, rows, H=C.H, n_points=4096, stats=None, fixed_subsample=False, seed=0):
        self.rows, self.H, self.n, self.stats, self.fixed = rows, H, n_points, stats, fixed_subsample
        self.epc, self.clc, self.rng = {}, {}, np.random.default_rng(seed)

    def _ep(self, p):
        if p not in self.epc: self.epc[p] = dict(np.load(p, allow_pickle=True))
        return self.epc[p]

    def _cl(self, obj):
        if obj not in self.clc: self.clc[obj] = cluster_raw(obj)
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
        raw = self._cl(obj); rng = np.random.default_rng(7000+i) if self.fixed else self.rng
        idx = rng.choice(len(raw["mu"]), self.n, replace=len(raw["mu"]) < self.n)   # subsample FIRST (fast), then transform
        feat = ((to_gripper(raw["mu"][idx], raw["q"][idx], raw["so"][idx], R0, p0, Rg, base_pos) - FMEAN) / FSTD).astype(np.float32)
        if self.stats is not None: tgt = np.clip((tgt - self.stats["mean"]) / self.stats["std"], -8.0, 8.0)
        return dict(pts=torch.from_numpy(feat), base_rel=torch.from_numpy(base_rel), closing=torch.from_numpy(ctrl),
                    table=torch.zeros(1), target=torch.from_numpy(tgt), weight=torch.tensor(float(r["weight"]), dtype=torch.float32),
                    tilt=torch.tensor(float(r["net_tilt"]), dtype=torch.float32), object=obj, shape=r["shape"], episode=r["episode"])


def compute_stats(rows, H=C.H, n=2500, seed=0):          # target z-score (UNCHANGED from my_dataset — target is frame-independent)
    ds = ChunkDS(rows, H); rng = np.random.default_rng(seed); sel = rng.choice(len(ds), min(n, len(ds)), replace=False)
    T = np.stack([ds[i]["target"].numpy() for i in sel]).reshape(-1, 9)
    return {"mean": T.mean(0).astype(np.float32), "std": (T.std(0)+1e-4).astype(np.float32)}
