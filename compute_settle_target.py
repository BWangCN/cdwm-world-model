"""Re-derive the grasps.npz-style H=32 CLOSING-SETTLE target from outcomes_v2 raw trajectories, so the CLEAN single
model trains the trajectory head (rigid episodes) + boolean head (all episodes) on ONE dataset (outcomes_v2). Per
episode: take the pre-LIFT settle window (phase < LIFT), resample to H frames, apply the grasps.npz target formula
(gripper-frame cumulative SE(3) delta since t0; target[0]=identity). Stores target (H,9) + base_rel (9) keyed by uid,
plus the v2_outcome label. Round-trip self-check (invert target -> world pose) asserts the formula matches the card.
    python compute_settle_target.py [--check_only]
"""
import os, sys, re, csv, glob, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wm.common import quat_to_R, R_to_6d, sixd_to_R

# v3 gripper dataset (self-contained: poses + achieved gripper channels). Rebuild the target purely from v3
# (v3 is NOT bit-identical to v1/v2 -> use v3 as the single canonical source; never join v3 gripper onto v2 poses).
V3 = os.environ.get("CDWM_V3", "/misc/kcgscratch1/MengyeGroup/yy5259/datasets/cdwm-grasp-dataset/gripper_v3/tier_a_outcomes_v2_aligned")
SETTLE = os.environ.get("CDWM_SETTLE", f"{V3}/settle_target.npz")
LIFT, H = 3, 32                                               # phase>=3 = lift; settle target length (matches grasps.npz)


def settle_target(op, oq, bp, bq, phase, driver):
    lift = np.where(phase >= LIFT)[0]
    end = int(lift[0]) if len(lift) else len(phase)          # settle window = [0, lift-onset)
    if end < 4: return None
    ks = np.linspace(0, end - 1, H).round().astype(int)      # H frames spanning the settle
    op, oq, bp, bq = op[ks].astype(np.float64), oq[ks].astype(np.float64), bp[ks].astype(np.float64), bq[ks].astype(np.float64)
    R0, p0, Rg = quat_to_R(oq[0]), op[0], quat_to_R(bq[0])   # t0 = first settle frame (gripper ~const pre-lift)
    tgt = np.zeros((H, 9), np.float32)
    for t in range(H):
        Rt = quat_to_R(oq[t]); dp = op[t] - p0; dR = Rt @ R0.T
        tgt[t] = np.concatenate([Rg.T @ dp, R_to_6d(Rg.T @ dR @ Rg)])
    base_rel = np.concatenate([R0.T @ (bp[0] - p0), R_to_6d(R0.T @ Rg)]).astype(np.float32)
    t0 = np.concatenate([oq[0], op[0], bq[0], bp[0]]).astype(np.float32)   # (14) raw t0: obj_quat,obj_pos,base_quat,base_pos
    # round-trip check: invert target -> world object pose, compare to the raw poses at the sampled frames
    resid = 0.0
    for t in (H // 2, H - 1):
        R_w = (Rg @ sixd_to_R(tgt[t, 3:]) @ Rg.T) @ R0; p_w = p0 + Rg @ tgt[t, :3]
        resid = max(resid, np.degrees(np.arccos(np.clip((np.trace(R_w.T @ quat_to_R(oq[t])) - 1) / 2, -1, 1))),
                    np.linalg.norm(p_w - op[t]) * 1000)       # deg or mm
    grip = driver[ks].astype(np.float32)                      # (H,) achieved closure (right_driver_joint rad, 0..0.8) at the target frames
    grip_phase = phase[ks].astype(np.int8)                    # (H,) phase per sampled frame (1=close 2=hold) -> phase-sliced grip eval
    return tgt, base_rel, t0, float(resid), grip, grip_phase


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--check_only", action="store_true"); a = ap.parse_args()
    idx = json.load(open(f"{V3}/index.json"))
    outc = {r["episode_id"]: r["v2_outcome"] for r in idx}    # episode_id -> label (ERR eps absent from shards)
    pools = sorted({r["pool"] for r in idx})
    TGT, BR, T0, UID, LAB, OBJ, GRIP, GPH, resids = [], [], [], [], [], [], [], [], []
    for pool in pools:
        z = np.load(f"{V3}/traj_{pool}.npz", allow_pickle=True)
        op, oq, bp, bq, ph, dr = z["obj_pos"], z["obj_quat"], z["base_pos"], z["base_quat"], z["phase"], z["driver_rad"]
        got = 0
        for eid, s, n in zip(z["ep_id"], z["ep_start"], z["ep_nframes"]):
            eid, s, n = str(eid), int(s), int(n); sl = slice(s, s + n)
            out = settle_target(op[sl], oq[sl], bp[sl], bq[sl], ph[sl], dr[sl])
            if out is None: continue
            tgt, br, t0, resid, grip, gph = out; resids.append(resid)
            if not a.check_only:
                TGT.append(tgt); BR.append(br); T0.append(t0); UID.append(eid)
                LAB.append(outc.get(eid, "UNK")); OBJ.append(re.sub(r"_g\d+$", "", eid)); GRIP.append(grip); GPH.append(gph)
            got += 1
        print(f"  {pool}: {got} episodes (resid p99 {np.percentile(resids,99):.3f})", flush=True)
        if a.check_only and len(resids) > 3000: break
    resids = np.array(resids)
    print(f"round-trip residual (deg-or-mm): med {np.median(resids):.4f} p99 {np.percentile(resids,99):.4f} max {resids.max():.4f}")
    if a.check_only: return
    np.savez(SETTLE, target=np.stack(TGT), base_rel=np.stack(BR), t0=np.stack(T0),
             uid=np.array(UID), v2_outcome=np.array(LAB), object_id=np.array(OBJ),
             grip_target=np.stack(GRIP), grip_phase=np.stack(GPH))
    print(f"wrote {SETTLE}  target {np.stack(TGT).shape}  grip {np.stack(GRIP).shape}  ({len(UID)} episodes)")


if __name__ == "__main__":
    main()
