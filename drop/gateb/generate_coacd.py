"""Gate B — CoACD END-TO-END episode generator (Item 1). Same near-boundary release sampler and same per-episode
HIDDEN CoM (explicit-inertial offset along a principal axis) as generate_obj.py, but the collision geometry is the
real mesh's CoACD convex DECOMPOSITION (more faithful than the single point-cloud hull), and basins are defined by
the real MESH's stable poses. Reuses hull_vs_coacd (align_mesh/settle_robust/build_coacd) + drop_sweep.

End-to-end consistency (Codex): the WM input cloud is sampled from the REAL MESH surface (build_mesh_cloud.py),
NOT the CoACD surface — CoACD seams change sampling density and the WM could learn simulator artifacts. Both the
sim geometry and the WM cloud derive from one real mesh.

    python generate_coacd.py --obj 011_banana --n 1000 --seed 0   # writes gateb/<obj>_coacd_s<seed>.npz
    python generate_coacd.py --obj 011_banana --smoke
"""
import os, argparse, time
import numpy as np, mujoco, trimesh, coacd
from trimesh.poses import compute_stable_poses
from scipy.spatial.transform import Rotation as R

HERE = os.path.dirname(os.path.abspath(__file__))
from drop.density.v2 import drop_sweep as DS                            # hull(), axes/half, basin_diff, PCDIR
from drop.hull_vs_coacd import align_mesh, settle_robust, build_coacd   # CoACD machinery (the 3 gotchas solved)
from drop.gateb.generate_obj import basin_of
coacd.set_log_level("error")


def mesh_stable_orientations(m, thresh=0.02):
    """Stable resting orientations of the REAL mesh (the object's true resting modes) — the basin frame."""
    T, P = compute_stable_poses(m, center_mass=m.center_mass, n_samples=1, threshold=thresh)
    return [t[:3, :3] for t in T], P


def mesh_hull_frame(m):
    """Hull center + principal axes + half-extents of the ALIGNED mesh (self-consistent CoM frame, stored in the
    npz so the WM's grounded latent uses the EXACT frame the CoM was placed in — the mesh cloud is sampled from
    this same aligned mesh)."""
    V = np.asarray(m.convex_hull.vertices, float)
    hc = V.mean(0); _, _, vt = np.linalg.svd(V - hc)                    # principal axes (rows of vt)
    half = np.abs((V - hc) @ vt.T).max(0)                              # half-extent along each principal axis
    return hc.astype(np.float32), vt.astype(np.float32), half.astype(np.float32)


def gen(obj, n, seed, smoke=False, max_steps=8000):
    if DS.hull(obj) is None: raise SystemExit(f"no point cloud for {obj}")
    m = align_mesh(obj)                                                  # real mesh in the hull frame (same centering)
    parts = coacd.run_coacd(coacd.Mesh(np.asarray(m.vertices), np.asarray(m.faces)), threshold=0.06)
    hulls = [np.asarray(v, float) for v, f in parts]
    Vc = np.vstack(hulls); rc = float(np.linalg.norm(Vc - Vc.mean(0), axis=1).max())
    hc, axes, half = mesh_hull_frame(m)                                 # self-consistent CoM frame (mesh-derived)
    Rs, P = mesh_stable_orientations(m)                                 # basin frame = real-mesh stable poses
    if len(Rs) < 2: raise SystemExit(f"{obj}: <2 mesh stable poses, no boundary to sample")
    rng = np.random.default_rng(seed)
    dcache = {}
    rel, dh, com, rest, bas = [], [], [], [], []
    settled = 0; t0 = time.time()
    N = 60 if smoke else n
    for _ in range(N):
        Rbase = Rs[rng.integers(len(Rs))]                               # near a mesh stable pose ...
        hax = rng.normal(size=3); hax[2] = 0; hax /= np.linalg.norm(hax)
        tilt = rng.uniform(15, 55)                                      # ... tilt toward a boundary
        q = (R.from_euler("z", rng.uniform(0, 2 * np.pi)) * R.from_rotvec(np.radians(tilt) * hax) * R.from_matrix(Rbase))
        x, y, z, w = q.as_quat(); q_rel = np.array([w, x, y, z])
        ai = rng.integers(3)                                            # hidden CoM: offset along a mesh principal axis
        delta = rng.uniform(-0.35, 0.35) * float(half[ai])
        key = (ai, round(float(delta) / 0.003) * 0.003)
        if key not in dcache:
            mdl = build_coacd(hulls, key[1], axes[ai], rc); dcache[key] = (mdl, mujoco.MjData(mdl))
        mdl, d = dcache[key]
        h = rng.uniform(0.005, 0.05)
        q_rest, ok = settle_robust(mdl, d, q_rel, Vc, h, max_steps=max_steps)   # orientation-based settle (CoACD micro-rolls)
        if not ok: continue
        settled += 1
        rel.append(q_rel); dh.append(h); com.append([ai, delta]); rest.append(q_rest); bas.append(basin_of(q_rest, Rs))
    return dict(rel=np.array(rel), dh=np.array(dh), com=np.array(com), rest=np.array(rest),
                basin=np.array(bas), n_stable=len(Rs), n_hulls=len(hulls), settle_rate=settled / max(N, 1),
                sec_per_ep=(time.time() - t0) / max(N, 1), hc=hc, axes=axes, half=half)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", required=True); ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    r = gen(a.obj, a.n, a.seed, smoke=a.smoke)
    if a.smoke:
        u, c = np.unique(r["basin"], return_counts=True)
        print(f"[SMOKE {a.obj}] hulls={r['n_hulls']} n_stable={r['n_stable']} settle={r['settle_rate']:.0%} "
              f"sec/ep={r['sec_per_ep']:.2f} episodes={len(r['rel'])} basins={dict(zip(u.tolist(), c.tolist()))}")
        return
    outp = os.path.join(HERE, f"{a.obj}_coacd_s{a.seed}.npz")
    np.savez(outp, object=a.obj, release_quat=r["rel"], drop_h=r["dh"], com=r["com"],
             rest_quat=r["rest"], basin=r["basin"],
             hull_center=r["hc"], hull_axes=r["axes"], hull_half=r["half"])   # self-consistent CoM frame for the grounded latent
    u, c = np.unique(r["basin"], return_counts=True); frac = np.sort(c / c.sum())[::-1]
    print(f"[{a.obj}] CoACD hulls={r['n_hulls']} settle {r['settle_rate']:.0%} n_stable={r['n_stable']} "
          f"episodes={len(r['rel'])} sec/ep={r['sec_per_ep']:.2f}")
    print(f"  rest-basin fractions {np.round(frac, 2)}  multimodal={(frac > 0.15).sum() >= 2}")
    print(f"  wrote {outp}")


if __name__ == "__main__":
    main()
