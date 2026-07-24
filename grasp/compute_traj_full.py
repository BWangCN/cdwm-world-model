"""Rollout targets (colleague's 'visualize the future', short-window revision). Per v2 episode, TWO targets:
 - PRIMARY `short` (K=10 native steps from LIFT-ONSET, ~0.32s): object-in-gripper motion RELATIVE to the lift-onset frame
   T_rel(k) = (6D(R0^T R(t)), p(t)-p0) -> starts at [identity6d, 0]; the model predicts DIVERGENCE (rigid~0, slip/drop grow).
   Native time (no resample) = honest 'predict near-term contact dynamics from grasp geometry'. Operating horizon (~5-8)
   read off the data (see diagnostic_window.py); we predict 10 and report the horizon curve.
 - SECONDARY `traj` (full close->lift->outcome window resampled to 32, ABSOLUTE T_go): viz/ablation only.
Trains the DiT on ALL outcomes. Cache uid-keyed to traj_full.npz.
    python compute_traj_full.py     (sbatch; ~1min)
"""
import os, sys, csv, collections
import numpy as np
from common.utils import quat_to_R, R_to_6d, sixd_to_R, geodesic_deg
from common.paths import OUTCOMES as OV
_OUT = os.path.join(OV, "traj_full.npz")
K = 10                                                        # native short-window steps from lift-onset
HFULL = 32                                                    # secondary full-resample length


def rel_traj(op, oq, bp, bq):                                 # world -> object-in-gripper (R_rel (N,3,3), p_rel (N,3) m)
    Ro = quat_to_R(oq); Rg = quat_to_R(bq)
    R_rel = np.einsum("nij,nik->njk", Rg, Ro)
    p_rel = np.einsum("nij,ni->nj", Rg, op - bp)
    return R_rel, p_rel


def _onset(phase):
    lift = np.where(phase >= 3)[0]
    if len(lift): return int(lift[0])                        # lift-onset = first lift frame
    hold = np.where(phase >= 2)[0]
    return int(hold[0]) if len(hold) else 0


def short_rel(R_rel, p_rel, phase):                          # (K,9) motion relative to lift-onset frame
    o0 = _onset(phase)
    idx = np.clip(np.arange(o0, o0 + K), 0, len(phase) - 1)  # K native steps, pad-last at episode end
    Rd = np.einsum("ij,njk->nik", R_rel[o0].T, R_rel[idx])   # R0^T R(t): starts at I
    pd = p_rel[idx] - p_rel[o0]                              # starts at 0
    return np.concatenate([R_to_6d(Rd), pd], -1).astype(np.float32)


def full_abs(R_rel, p_rel, phase):                          # (HFULL,9) absolute T_go over close->lift->outcome (viz)
    win = np.where(phase >= 2)[0]
    if len(win) < 2: win = np.arange(len(phase))
    idx = np.round(np.linspace(win[0], win[-1], HFULL)).astype(int)
    return np.concatenate([R_to_6d(R_rel[idx]), p_rel[idx]], -1).astype(np.float32)


def main():
    rows = list(csv.DictReader(open(f"{OV}/outcomes_index.csv")))
    byshard = collections.defaultdict(list)
    for i, r in enumerate(rows): byshard[r["shard"]].append(i)
    SH = np.zeros((len(rows), K, 9), np.float32); T = np.zeros((len(rows), HFULL, 9), np.float32)
    uids = np.array([r["uid"] for r in rows])
    for shard, ii in byshard.items():
        zf = np.load(f"{OV}/{shard}")
        A = {k: zf[k] for k in ("obj_pos", "obj_quat", "base_pos", "base_quat", "phase")}
        for i in ii:
            r = rows[i]; s = int(r["frame_start"]); e = s + int(r["n_frames"])
            R_rel, p_rel = rel_traj(A["obj_pos"][s:e].astype(float), A["obj_quat"][s:e].astype(float),
                                    A["base_pos"][s:e].astype(float), A["base_quat"][s:e].astype(float))
            ph = A["phase"][s:e]
            SH[i] = short_rel(R_rel, p_rel, ph); T[i] = full_abs(R_rel, p_rel, ph)
    np.savez(_OUT, short=SH, traj=T, uid=uids, K=K); print(f"cached {_OUT}  short {SH.shape} full {T.shape}")
    # sanity: short frame 0 must be [identity-6d, 0]
    assert np.allclose(SH[:, 0, :6], np.array([1, 0, 0, 0, 1, 0], np.float32), atol=1e-4) and np.allclose(SH[:, 0, 6:], 0, atol=1e-5), "onset frame != identity"
    # VALIDATE: per-mode drift over the SHORT window (should match the diagnostic ordering rigid<slip<drop)
    oc = np.array([r["v2_outcome"] for r in rows])
    R0 = sixd_to_R(SH[:, 0, :6])
    print(f"\n{'mode':22s} {'k=K-1 drift°':>12s} {'transMM':>8s}")
    for m in ["RIGID", "TRANSIENT_SLIP", "PERSISTENT_SLIP", "LIFTED_DROPPED", "CLOSED_NEVER_LIFTED"]:
        g = oc == m
        dr = geodesic_deg(sixd_to_R(SH[g, -1, :6]), R0[g]); tr = np.linalg.norm(SH[g, -1, 6:9], axis=1) * 1000
        print(f"{m:22s} {np.median(dr):12.1f} {np.median(tr):8.1f}")


if __name__ == "__main__":
    main()
