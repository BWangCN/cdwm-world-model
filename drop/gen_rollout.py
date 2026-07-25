"""Generate K-step settling trajectories for the drop ROLLOUT WM. Re-simulate each gateb episode from com_sim
(the 3 mm-quantized CoM the sim used; deterministic) and record the object orientation at K evenly-spaced steps from
release to settle. Output: gateb/<obj>_roll_s0.npz with traj_quat (N,K,4) world-frame wxyz. Reuses drop_sweep
(hull + explicit-inertial CoM offset).

    python gen_rollout.py --obj 011_banana [--n 20]
"""
import os, sys, argparse
import numpy as np, mujoco
from scipy.spatial.transform import Rotation as R
HERE = os.path.dirname(os.path.abspath(__file__))
from drop.density.v2 import drop_sweep as DS
K = 16


def settle_traj(m, d, quat, V, drop_h, max_steps=4000):
    Rr = R.from_quat(np.r_[quat[1:], quat[0]]); pz = drop_h - Rr.apply(V)[:, 2].min()
    mujoco.mj_resetData(m, d); d.qpos[:3] = [0, 0, pz]; d.qpos[3:7] = quat; d.qvel[:] = 0
    mujoco.mj_forward(m, d); poses = [quat.copy()]; still = 0
    for _ in range(max_steps):
        mujoco.mj_step(m, d); poses.append(d.qpos[3:7].copy())
        if np.linalg.norm(d.qvel[:3]) < 0.006 and np.linalg.norm(d.qvel[3:6]) < 0.1047:
            still += 1
            if still >= 75: break
        else:
            still = 0
    poses = np.array(poses)                                   # (T+1, 4) world-frame wxyz, release -> rest
    idx = np.linspace(0, len(poses) - 1, K).astype(int)       # K evenly-spaced keyframes
    return poses[idx], (still >= 75)


def gen(obj, nmax=None):
    z = np.load(f"{HERE}/gateb/{obj}_gateb_s0.npz", allow_pickle=True)
    H = DS.hull(obj); rel, dh, com, bas = z["release_quat"], z["drop_h"], z["com"], z["basin"]
    n = len(rel) if nmax is None else min(nmax, len(rel))
    dcache = {}; trajs = []; keep = []
    for i in range(n):
        ai = int(com[i, 0]); delta = round(float(com[i, 1]) / 0.003) * 0.003   # com_sim (deterministic)
        key = (ai, delta)
        if key not in dcache: mdl = DS.build(H, delta, H["axes"][ai]); dcache[key] = (mdl, mujoco.MjData(mdl))
        mdl, d = dcache[key]
        tq, ok = settle_traj(mdl, d, rel[i], H["V"], float(dh[i]))
        if ok: trajs.append(tq); keep.append(i)
    trajs = np.array(trajs); keep = np.array(keep)
    outp = f"{HERE}/gateb/{obj}_roll_s0.npz"
    np.savez(outp, object=obj, traj_quat=trajs, release_quat=rel[keep], drop_h=dh[keep], com=com[keep], basin=bas[keep])
    print(f"[{obj}] {len(trajs)}/{n} settled trajs, K={K}, traj_quat {trajs.shape} -> {outp}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--obj", required=True); ap.add_argument("--lim", type=int, default=None)
    a = ap.parse_args(); gen(a.obj, a.lim)


if __name__ == "__main__":
    main()
