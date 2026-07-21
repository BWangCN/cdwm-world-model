"""Phase-2 targets: convert every outcomes_v2 episode to the object-in-MOVING-gripper pose T_go(t) = (R_g(t)^T R_obj(t),
R_g(t)^T (p_obj-p_g)), then derive per-episode SUMMARY targets (Codex Q5). Cache uid-keyed to traj_summaries.npz.
Rigid -> T_go ~ constant; slip -> drifts; drop -> diverges/falls. Reference pose = mean T_go over the HOLD phase (2).

Summary vector (per episode): [ settle_6d(6) settle_trans_mm(3) | max_drift_deg(1) max_drift_mm(1) | slip_rate_deg_s(1) |
                                drop_bin(1) drop_time_norm(1) | held_frac_monitor(1) ]  = 15
Also validates: prints per-mode max_drift / settle-trans so we can confirm the target design before training.
"""
import os, sys, csv, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wm.common import quat_to_R, R_to_6d, geodesic_deg
OV = os.environ.get("CDWM_OV", "/misc/kcgscratch1/MengyeGroup/yy5259/datasets/cdwm-grasp-dataset/outcomes_v2")
_OUT = os.path.join(OV, "traj_summaries.npz")
DT = 0.032                                                    # s between frames (README)


def rel_traj(op, oq, bp, bq):                                 # world trajs -> object-in-gripper (R_rel (N,3,3), p_rel (N,3) meters)
    Ro = quat_to_R(oq); Rg = quat_to_R(bq)
    R_rel = np.einsum("nij,nik->njk", Rg, Ro)                 # Rg^T Ro
    p_rel = np.einsum("nij,ni->nj", Rg, op - bp)             # Rg^T (po-pg)
    return R_rel, p_rel


def summarize(op, oq, bp, bq, phase, held, clear_mm, is_drop):
    R_rel, p_rel = rel_traj(op, oq, bp, bq)
    hold = phase == 2
    ref = np.where(hold)[0] if hold.any() else np.arange(min(3, len(phase)))
    R_ref = R_rel[ref].mean(0); p_ref = p_rel[ref[0]] if len(ref) else p_rel[0]
    # re-orthonormalize R_ref (mean of rotations) via SVD
    U, _, Vt = np.linalg.svd(R_ref); R_ref = U @ Vt
    drift_deg = geodesic_deg(R_rel, np.broadcast_to(R_ref, R_rel.shape))
    drift_mm = np.linalg.norm(p_rel - p_ref, axis=1) * 1000
    mon = phase == 4
    slip_rate = 0.0
    if mon.sum() >= 3:                                        # robust drift slope during monitor (deg/s)
        t = np.arange(mon.sum()) * DT; slip_rate = float(np.polyfit(t, drift_deg[mon], 1)[0])
    # drop time = first monitor/lift frame where contact lost (held False) after lift, normalized by episode length
    lost = np.where((~held.astype(bool)) & (phase >= 3))[0]
    drop_time = float(lost[0] / len(phase)) if (is_drop and len(lost)) else 1.0
    held_frac = float(held[mon].mean()) if mon.any() else float(held.mean())
    settle_6d = R_to_6d(R_rel[-1]); settle_trans = p_rel[-1] * 1000
    return np.concatenate([settle_6d, settle_trans, [drift_deg.max(), drift_mm.max(), slip_rate,
                                                     float(is_drop), drop_time, held_frac]]).astype(np.float32)


def main():
    rows = list(csv.DictReader(open(f"{OV}/outcomes_index.csv")))
    byshard = collections.defaultdict(list)
    for i, r in enumerate(rows): byshard[r["shard"]].append(i)
    S = np.zeros((len(rows), 15), np.float32); uids = np.array([r["uid"] for r in rows])
    for shard, ii in byshard.items():
        zf = np.load(f"{OV}/{shard}")                        # materialize shard arrays ONCE (npz ignores mmap -> never slice a lazy npz in a loop)
        A = {k: zf[k] for k in ("obj_pos", "obj_quat", "base_pos", "base_quat", "phase", "held", "clear_mm")}
        for i in ii:
            r = rows[i]; s = int(r["frame_start"]); e = s + int(r["n_frames"])
            S[i] = summarize(A["obj_pos"][s:e].astype(float), A["obj_quat"][s:e].astype(float),
                             A["base_pos"][s:e].astype(float), A["base_quat"][s:e].astype(float),
                             A["phase"][s:e], A["held"][s:e], A["clear_mm"][s:e], r["v2_outcome"] == "LIFTED_DROPPED")
    np.savez(_OUT, summ=S, uid=uids); print(f"cached {_OUT}  shape {S.shape}")
    # VALIDATE: per-mode summary medians (design check)
    oc = np.array([r["v2_outcome"] for r in rows])
    print(f"\n{'mode':22s} {'maxDrift°':>9s} {'maxDrift_mm':>11s} {'settle_mm':>9s} {'slip°/s':>8s} {'drop%':>6s} {'heldFrac':>8s}")
    for m in ["RIGID", "TRANSIENT_SLIP", "PERSISTENT_SLIP", "LIFTED_DROPPED", "CLOSED_NEVER_LIFTED"]:
        g = S[oc == m]
        st = np.linalg.norm(g[:, 6:9], axis=1)
        print(f"{m:22s} {np.median(g[:,9]):9.1f} {np.median(g[:,10]):11.1f} {np.median(st):9.1f} {np.median(g[:,11]):8.2f} {100*g[:,12].mean():6.0f} {np.median(g[:,14]):8.2f}")


if __name__ == "__main__":
    main()
