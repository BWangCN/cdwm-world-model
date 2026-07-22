"""Gate B step 1 — generate the near-boundary / hidden-CoM drop episodes (the frontier regime the baseline WM
TIES no-motion on). For a CoM-sensitive object we release near the stable-pose SADDLE (where V3 showed CoM
decides the fall) with a per-episode HIDDEN center-of-gravity, drop, and record the resting pose. The latent
(com_delta) is recorded so an oracle-latent model can be compared (Gate B success criterion 2).

Design decisions (pinned):
- CoM model: controlled explicit-<inertial> offset along the object's long axis, sampled per episode from a
  physically-motivated prior (head-heavy .. handle-heavy). This is the tractable first version; per-CoACD-hull
  material density is the physical hardening (needs per-object CoACD, deferred). Framed honestly as controlled.
- Sampler: near the balance orientation (theta ~ 95deg for the hammer, from density/v3v4). A trimesh stable-pose
  saddle sampler generalizes this to arbitrary objects (next step).
- Output: episodes (release_quat, drop_h, com_delta[latent], rest_quat) -> resting-pose DISTRIBUTION for a fixed
  visible observation. Multimodal by construction iff CoM controls the basin (verified here).

Starts on the hammer (proven reliable settle in density/v3v4).
    python generate.py --n 400 --out hammer_gateb.npz
"""
import os, sys, argparse
import numpy as np, mujoco
from scipy.spatial.transform import Rotation as R

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "density", "v3v4"))
import boundary_multimodal as BM                          # proven hammer sim: build(delta), A.drop, basin_diff, release, HALF, HEAD_SIGN


def cluster_basins(quats, thr=30.0):                      # yaw-invariant clustering of resting poses -> basin id
    labels = -np.ones(len(quats), int); K = 0
    for i in range(len(quats)):
        if labels[i] >= 0: continue
        labels[i] = K
        for j in range(i + 1, len(quats)):
            if labels[j] < 0 and BM.basin_diff(quats[i], quats[j]) < thr: labels[j] = K
        K += 1
    return labels, K


def generate(n=400, seed=0):
    rng = np.random.default_rng(seed)
    dcache = {}                                            # bucket compiled models by CoM offset (reuse across episodes)
    rel, dh, com, rest = [], [], [], []
    for _ in range(n):
        theta = rng.uniform(88, 100)                      # near the saddle (v3v4: peak sensitivity ~95deg)
        q_rel = BM.release(theta, rng, jitter=6.0)
        delta = BM.HEAD_SIGN * rng.uniform(-0.2 * BM.HALF, 0.7 * BM.HALF)   # HIDDEN CoM latent (head..handle)
        key = round(float(delta) / 0.005) * 0.005
        if key not in dcache:
            m = BM.build(key); dcache[key] = (m, mujoco.MjData(m))
        m, d = dcache[key]
        h = rng.uniform(0.005, 0.05)
        q_rest, ok, _ = BM.A.drop(m, d, q_rel, h)
        if not ok: continue
        rel.append(q_rel); dh.append(h); com.append(delta); rest.append(q_rest)
    return np.array(rel), np.array(dh), np.array(com), np.array(rest)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="hammer_gateb.npz"); a = ap.parse_args()
    rel, dh, com, rest = generate(a.n, seed=a.seed)
    labels, K = cluster_basins(rest)
    frac = np.sort(np.bincount(labels) / len(labels))[::-1]
    outp = os.path.join(HERE, a.out)
    np.savez(outp, release_quat=rel, drop_h=dh, com_delta=com, rest_quat=rest, basin=labels)
    print(f"generated {len(rel)}/{a.n} settled episodes  |  {K} rest basins, fractions {np.round(frac, 2)}")
    print(f"multimodal={(frac > 0.15).sum() >= 2}  |  latent com_delta range [{com.min()*1000:.0f},{com.max()*1000:.0f}]mm")
    # sanity: does the hidden CoM actually drive the basin? correlate com_delta with basin
    if K >= 2:
        for b in range(min(K, 3)):
            cb = com[labels == b]
            print(f"  basin {b}: n={int((labels==b).sum())}  mean com_delta {cb.mean()*1000:+.0f}mm")
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
