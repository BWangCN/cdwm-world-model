"""Physical-density realism check (Codex #2, bounded subset). The main Gate B uses a CONTROLLED explicit-inertial
CoM offset. Here we validate on the hammer (5 CoACD hulls) that a PHYSICAL per-hull density latent reproduces the
same effect: sample the head material density (hollow->steel), which physically moves the CoM, drop near the
stable-pose saddle, and check (a) multimodal rest basins, (b) the physical latent CAUSES the basin (randomized
head density -> I(basin; density) > 0). If yes, the controlled offset is a faithful proxy, not a synthetic artifact.

    python physical_density_check.py --n 1200
"""
import os, sys, argparse
import numpy as np, mujoco
from scipy.spatial.transform import Rotation as R
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pilot"))
import drop_ab as A                                          # hammer hulls, build(dens per-hull), drop(), HANDLE/HEAD
import boundary_multimodal as BM                             # release() near saddle, basin_diff()

HANDLE_D = 500.0                                             # hollow fiberglass handle (fixed)
MATS = (1000.0, 7800.0)                                      # head density range: hollow/plastic -> steel (randomized latent)


def dmi(a, b):
    a = a.astype(int); b = b.astype(int); mi = 0.0
    for x in np.unique(a):
        px = np.mean(a == x)
        for y in np.unique(b):
            py = np.mean(b == y); pxy = np.mean((a == x) & (b == y))
            if pxy > 0: mi += pxy * np.log2(pxy / (px * py))
    return mi


def cluster(quats, thr=30.0):
    lab = -np.ones(len(quats), int); K = 0
    for i in range(len(quats)):
        if lab[i] >= 0: continue
        lab[i] = K
        for j in range(i + 1, len(quats)):
            if lab[j] < 0 and BM.basin_diff(quats[i], quats[j]) < thr: lab[j] = K
        K += 1
    return lab, K


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=1200); a = ap.parse_args()
    rng = np.random.default_rng(0); dcache = {}
    rest, hd = [], []
    for _ in range(a.n):
        h = rng.uniform(*MATS)                               # PHYSICAL hidden latent = head material density (randomized)
        key = round(h / 200) * 200
        if key not in dcache:
            dens = {i: HANDLE_D for i in A.HANDLE}; dens.update({i: key for i in A.HEAD})
            m = A.build(dens); dcache[key] = (m, mujoco.MjData(m))
        m, d = dcache[key]
        theta = rng.uniform(88, 100); q_rel = BM.release(theta, rng, jitter=6.0)
        q_rest, ok, _ = A.drop(m, d, q_rel, rng.uniform(0.005, 0.05))
        if ok: rest.append(q_rest); hd.append(h)
    rest = np.array(rest); hd = np.array(hd)
    lab, K = cluster(rest); frac = np.sort(np.bincount(lab) / len(lab))[::-1]
    hbin = (hd > np.median(hd)).astype(int)                  # 2-bin physical latent
    mi = dmi(lab, hbin)
    print(f"physical-density hammer: {len(rest)}/{a.n} settled; head density U{MATS} kg/m^3 (randomized)")
    print(f"  rest basins: {K}, fractions {np.round(frac, 2)}  multimodal={(frac > 0.15).sum() >= 2}")
    print(f"  I(basin ; physical head-density) = {mi:.3f} bits  (>0 -> physical mass distribution CAUSES the basin)")
    for b in range(min(K, 3)):
        print(f"    basin {b}: n={int((lab == b).sum())}  mean head density {hd[lab == b].mean():.0f} kg/m^3")


if __name__ == "__main__":
    main()
