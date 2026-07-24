"""Verify the compact gateb dataset is a CONSISTENT recipe: re-simulate stored episodes from their recorded inputs
(release_quat, drop_h, com[hidden CoM]) using the object's point-cloud hull + the MuJoCo sim, and check the
regenerated resting pose / basin matches what we stored. If it matches, the 13 MB (inputs + outcome) + the point
clouds + this sim losslessly regenerate the full trajectories -> a reliable compact sharing format.

    python verify_gateb.py --objs 011_banana 050_medium_clamp --n 40
"""
import os, sys, argparse
import numpy as np, mujoco
from scipy.spatial.transform import Rotation as R
HERE = os.path.dirname(os.path.abspath(__file__))
from drop.density.v2 import drop_sweep as DS
from drop.gateb.generate_obj import stable_orientations, basin_of


def geod(qa, qb):
    dR = R.from_quat(np.r_[qb[1:], qb[0]]) * R.from_quat(np.r_[qa[1:], qa[0]]).inv()
    return float(np.degrees(dR.magnitude()))


def check(obj, n):
    z = np.load(f"{HERE}/{obj}_gateb_s0.npz", allow_pickle=True)
    H = DS.hull(obj); Rs, _ = stable_orientations(H)
    rel, dh, com, rest, bas = z["release_quat"], z["drop_h"], z["com"], z["rest_quat"], z["basin"]
    com_sim = z["com_sim"] if "com_sim" in z.files else com     # the quantized CoM the simulator actually used (exact recipe)
    n = min(n, len(rel)); mb_un = mb_rd = 0; errs = []
    for i in range(n):
        ai = int(com[i, 0])
        for tag, dl in [("stored_com", float(com[i, 1])), ("com_sim", float(com_sim[i, 1]))]:
            m = DS.build(H, dl, H["axes"][ai]); d = mujoco.MjData(m)
            q, ok = DS.settle(m, d, rel[i], H["V"], float(dh[i]), max_steps=4000)
            b = basin_of(q, Rs)
            if tag == "stored_com":
                mb_un += int(b == int(bas[i])); errs.append(geod(q, rest[i]))
            else:
                mb_rd += int(b == int(bas[i]))
    return dict(obj=obj, n=n, basin_match_stored_com=round(mb_un / n, 3),
               basin_match_com_sim=round(mb_rd / n, 3), median_rest_err_deg=round(float(np.median(errs)), 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objs", nargs="+", default=["011_banana", "050_medium_clamp", "033_spatula"])
    ap.add_argument("--n", type=int, default=40); a = ap.parse_args()
    print(f"MuJoCo {mujoco.__version__}")
    for o in a.objs:
        print(check(o, a.n))


if __name__ == "__main__":
    main()
