"""Rollout dataset: K-step settling-trajectory target (from gen_rollout.py's <obj>_roll_s0.npz) on the SAME
release-frame cloud + release params + hidden-CoM latent as GateBDS. Target = orientation at each of the K steps
relative to release (6D, padded to 9), so GateBDiT(H=K) denoises the whole settling trajectory instead of the
endpoint. Object-disjoint split shared with Gate B. Subclasses GateBDS to reuse the encoder-input machinery.

    from drop.my_dataset_roll import RollDS
"""
import os, sys, glob
import numpy as np, torch
from common.utils import quat_to_R, R_to_6d
from drop.my_dataset_gateb import GateBDS, _mask_for, GATEB, _FEAT_STATS

K = 16


def _load_roll():
    OBJ, RELQ, DH, COM, TRAJ, BAS = [], [], [], [], [], []
    for f in sorted(glob.glob(f"{GATEB}/*_roll_s0.npz")):
        z = np.load(f, allow_pickle=True); o = str(z["object"]); n = len(z["release_quat"])
        OBJ += [o] * n; RELQ.append(z["release_quat"]); DH.append(z["drop_h"]); COM.append(z["com"].astype(np.float32))
        TRAJ.append(z["traj_quat"]); BAS.append(z["basin"])
    return (np.array(OBJ), np.concatenate(RELQ), np.concatenate(DH), np.concatenate(COM),
            np.concatenate(TRAJ), np.concatenate(BAS))


class RollDS(GateBDS):
    def __init__(self, split, n_points=4096, stats=None, seed=0):
        OBJ, RELQ, DH, COM, TRAJ, BAS = _load_roll()
        m = _mask_for(OBJ, RELQ, split)
        self.obj = OBJ[m]; self.dh = DH[m].astype(np.float32); self.com = COM[m]; self.basin = BAS[m]
        Rrel = quat_to_R(RELQ[m].astype(np.float64)); self.R_rel = Rrel.astype(np.float32)
        traj = TRAJ[m].astype(np.float64); Nn, Kk = traj.shape[0], traj.shape[1]
        Rtr = quat_to_R(traj.reshape(-1, 4)).reshape(Nn, Kk, 3, 3)
        dR = np.einsum("nkij,nlj->nkil", Rtr, Rrel)          # R_traj_k @ R_rel^T : orientation at step k rel. release
        rest6d = R_to_6d(dR.reshape(-1, 3, 3)).reshape(Nn, Kk, 6).astype(np.float32)
        self.target = np.concatenate([rest6d, np.zeros((Nn, Kk, 3), np.float32)], -1)   # (N,K,9)
        self.base_rel = np.concatenate([self.dh[:, None], R_to_6d(self.R_rel).astype(np.float32),
                                        np.zeros((len(self.obj), 2), np.float32)], 1)    # (N,9)
        ax = self.com[:, 0].astype(int); dl = self.com[:, 1].astype(np.float32)
        oh = np.eye(3, dtype=np.float32)[np.clip(ax, 0, 2)]
        self.latent = np.concatenate([(dl * 20.0)[:, None], oh], 1).astype(np.float32)   # (N,4)
        self.n = n_points; self.clc = {}; self.rng = np.random.default_rng(seed); self.fixed = (split != "train")
        self.stats = stats if stats is not None else self._roll_stats()
        _fs = np.load(_FEAT_STATS); self.fmean, self.fstd = _fs["mean"], _fs["std"]
        self.latent_mode = os.environ.get("CDWM_GATEB_LATENT", "abstract")
        self._hc = {}; self.com_shuf = self.com[np.random.default_rng(1).permutation(len(self.com))]

    def _roll_stats(self):                                    # per-channel target stats over all K steps
        T = self.target.reshape(-1, 9); sel = self.rng.choice(len(T), min(20000, len(T)), replace=False)
        return {"mean": T[sel].mean(0).astype(np.float32), "std": (T[sel].std(0) + 1e-4).astype(np.float32)}

    def __getitem__(self, j):
        d = super().__getitem__(j)                            # reuse pts + latent + base_rel building
        d["target"] = d["target"].squeeze(0)                 # (1,K,9) -> (K,9); batched -> (B,K,9) for GateBDiT(H=K)
        d["closing"] = torch.zeros(K)                        # H=K context (drop has no gripper signal); encoder wants len-K
        return d
