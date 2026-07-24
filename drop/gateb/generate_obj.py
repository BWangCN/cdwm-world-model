"""Gate B — geometry-consistent multi-object episode generator. Uses each object's POINT-CLOUD convex hull as
the collision geom (so the sim geometry == the cloud the WM sees), trimesh stable poses for a near-boundary
release sampler, and a per-episode HIDDEN CoM (explicit-inertial offset along a principal axis). Drops, records
the resting pose + latent. Reuses density/v2/drop_sweep (hull/build/settle) + trimesh.

    python generate_obj.py --obj 048_hammer --n 800 --seed 0     # writes gateb/<obj>_gateb_s<seed>.npz
"""
import os, sys, argparse
import numpy as np, mujoco, trimesh
from trimesh.poses import compute_stable_poses
from scipy.spatial.transform import Rotation as R

HERE = os.path.dirname(os.path.abspath(__file__))
from drop.density.v2 import drop_sweep as DS                                    # hull(), build(H,delta,axis), settle(m,d,quat,V,drop_h), basin_diff


def stable_orientations(H, thresh=0.02):
    m = trimesh.Trimesh(vertices=H["V"]).convex_hull       # V is centered -> center_mass ~ 0
    T, P = compute_stable_poses(m, center_mass=m.center_mass, n_samples=1, threshold=thresh)
    return [t[:3, :3] for t in T], P                       # resting-orientation rotations + probs


def basin_of(q_rest, stable_Rs):                           # nearest stable pose (yaw-invariant) -> basin id
    Rr = R.from_quat(np.r_[q_rest[1:], q_rest[0]])
    phi = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    best, bi = 1e9, -1
    for i, Rs in enumerate(stable_Rs):
        d = ((R.from_euler("z", phi) * Rr) * R.from_matrix(Rs).inv()).magnitude().min()
        if d < best: best, bi = d, i
    return bi


def gen(obj, n, seed):
    H = DS.hull(obj)
    if H is None: raise SystemExit(f"no point cloud for {obj}")
    Rs, P = stable_orientations(H)
    if len(Rs) < 2: raise SystemExit(f"{obj}: <2 stable poses, no boundary to sample")
    rng = np.random.default_rng(seed)
    dcache = {}
    rel, dh, com, rest, bas = [], [], [], [], []
    settled = 0
    for _ in range(n):
        Rbase = Rs[rng.integers(len(Rs))]                  # start near a stable pose ...
        hax = rng.normal(size=3); hax[2] = 0; hax /= np.linalg.norm(hax)
        tilt = rng.uniform(15, 55)                         # ... tilt toward a boundary (near-boundary release)
        q = (R.from_euler("z", rng.uniform(0, 2 * np.pi)) * R.from_rotvec(np.radians(tilt) * hax) * R.from_matrix(Rbase))
        x, y, z, w = q.as_quat(); q_rel = np.array([w, x, y, z])
        ai = rng.integers(3)                               # hidden CoM: offset along a random principal axis
        delta = rng.uniform(-0.35, 0.35) * H["half"][ai]
        key = (ai, round(float(delta) / 0.003) * 0.003)
        if key not in dcache:
            mdl = DS.build(H, key[1], H["axes"][ai]); dcache[key] = (mdl, mujoco.MjData(mdl))
        mdl, d = dcache[key]
        h = rng.uniform(0.005, 0.05)
        q_rest, ok = DS.settle(mdl, d, q_rel, H["V"], h, max_steps=4000)   # near-boundary tumbles need a longer window
        if not ok: continue
        settled += 1
        rel.append(q_rel); dh.append(h); com.append([ai, delta]); rest.append(q_rest); bas.append(basin_of(q_rest, Rs))
    return dict(rel=np.array(rel), dh=np.array(dh), com=np.array(com), rest=np.array(rest),
                basin=np.array(bas), n_stable=len(Rs), settle_rate=settled / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", required=True); ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    r = gen(a.obj, a.n, a.seed)
    outp = os.path.join(HERE, f"{a.obj}_gateb_s{a.seed}.npz")
    np.savez(outp, object=a.obj, release_quat=r["rel"], drop_h=r["dh"], com=r["com"],
             rest_quat=r["rest"], basin=r["basin"])
    u, c = np.unique(r["basin"], return_counts=True); frac = np.sort(c / c.sum())[::-1]
    print(f"[{a.obj}] settle {r['settle_rate']:.0%}  n_stable={r['n_stable']}  episodes={len(r['rel'])}")
    print(f"  rest-basin fractions {np.round(frac, 2)}  multimodal={(frac > 0.15).sum() >= 2}")
    print(f"  wrote {outp}")


if __name__ == "__main__":
    main()
