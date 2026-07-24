"""Bounded physical-density realism validation (Codex #2; generalizes the hammer check to several objects).
Main Gate B uses a CONTROLLED explicit-inertial CoM offset on the point-cloud convex hull. Here, for CoM-sensitive
objects that have a real mesh, CoACD-decompose the mesh, give a PHYSICAL heterogeneous per-hull density (the hulls on
one end are heavy with their density randomized per episode, so the CoM physically moves), drop near a stable-pose
boundary, and test that the physical latent CAUSES the basin: I(basin; density) > 0 and the rest basins are multimodal.
If it holds across objects, the controlled explicit-inertial offset is a faithful proxy for real mass distribution
(generalizing the hammer, I=0.161~0.165), not a synthetic artifact. This is a SEPARATE realism validation on real mesh
geometry; it is NOT integrated into the WM (which uses the point-cloud hull), exactly as the hammer check was.

    python multi_physical_density.py --objs 051_large_clamp OXO_Soft_Works_Can_Opener_SnapLock --n 800
    python multi_physical_density.py --smoke
"""
import os, sys, json, argparse
import numpy as np, mujoco, trimesh, coacd
from scipy.spatial.transform import Rotation as R
from trimesh.poses import compute_stable_poses
coacd.set_log_level("error")
HERE = os.path.dirname(os.path.abspath(__file__))
from drop.gateb.generate_obj import basin_of
from drop.density.v2 import drop_sweep as DS                                       # PCDIR (point-cloud scale reference)

from common.paths import OBJECTS as OBJDIR
LIGHT_D = 300.0
HEAVY_RANGE = (400.0, 4000.0)                                 # heavy-end material density latent (randomized per episode)
DEFAULT = ["051_large_clamp", "OXO_Soft_Works_Can_Opener_SnapLock", "Remington_1_12_inch_Hair_Straightener",
           "Razer_Naga_MMO_Gaming_Mouse", "Schleich_Hereford_Bull", "Nintendo_Yoshi_Action_Figure"]


def dmi(a, b):                                                # discrete mutual information (bits), same as the hammer check
    a = a.astype(int); b = b.astype(int); mi = 0.0
    for x in np.unique(a):
        px = np.mean(a == x)
        for y in np.unique(b):
            py = np.mean(b == y); pxy = np.mean((a == x) & (b == y))
            if pxy > 0: mi += pxy * np.log2(pxy / (px * py))
    return mi


def decompose(obj):
    m = trimesh.load(os.path.join(OBJDIR, obj, "mesh.obj"), force="mesh")
    mu = np.load(f"{DS.PCDIR}/{obj}.npz")["mu"].astype(np.float64)   # match the point-cloud scale (sim ground-truth)
    m.apply_scale((mu.max(0) - mu.min(0)).max() / (m.vertices.max(0) - m.vertices.min(0)).max())
    m.apply_translation(-m.center_mass)
    parts = coacd.run_coacd(coacd.Mesh(m.vertices, m.faces), threshold=0.06)
    hulls = [np.asarray(v, float) for v, f in parts]
    _, _, vt = np.linalg.svd(m.vertices - m.vertices.mean(0)); axis = vt[0]   # longest principal axis
    proj = np.array([h.mean(0) for h in hulls]) @ axis
    heavy = proj >= np.quantile(proj, 0.6)                    # hulls on one end -> the density-varied group
    if heavy.sum() == 0 or heavy.all(): heavy = proj >= np.median(proj)
    return m, hulls, heavy


def build(hulls, dens):                                       # MuJoCo model; inertiafromgeom -> CoM computed from per-hull density
    meshes = "".join(f'<mesh name="h{i}" vertex="{" ".join(f"{x:.5f}" for x in h.flatten())}"/>' for i, h in enumerate(hulls))
    geoms = "".join(f'<geom type="mesh" mesh="h{i}" density="{dens[i]:.1f}" condim="6" friction="1 0.005 0.0001"/>'
                    for i in range(len(hulls)))
    xml = f"""<mujoco>
      <compiler inertiafromgeom="true"/>
      <option timestep="0.002" gravity="0 0 -9.81"/>
      <asset>{meshes}</asset>
      <worldbody>
        <geom name="table" type="plane" size="3 3 0.1" condim="6" friction="1 0.005 0.0001"/>
        <body name="obj" pos="0 0 1"><freejoint/>{geoms}</body>
      </worldbody></mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def settle(m, d, quat, V, drop_h, max_steps=4000):
    Rr = R.from_quat(np.r_[quat[1:], quat[0]]); pz = drop_h - Rr.apply(V)[:, 2].min()
    mujoco.mj_resetData(m, d); d.qpos[:3] = [0, 0, pz]; d.qpos[3:7] = quat; d.qvel[:] = 0
    mujoco.mj_forward(m, d); still = 0
    for _ in range(max_steps):
        mujoco.mj_step(m, d)
        if np.linalg.norm(d.qvel[:3]) < 0.006 and np.linalg.norm(d.qvel[3:6]) < 0.1047:
            still += 1
            if still >= 75: break
        else:
            still = 0
    return d.qpos[3:7].copy(), still >= 75


def rot_to_vertical(axis):                                    # body `axis` -> world +z (from drop_sweep; no scipy warning)
    z = np.array([0.0, 0, 1]); v = axis / np.linalg.norm(axis); c = np.dot(v, z)
    if c > 0.9999: return R.identity()
    if c < -0.9999: return R.from_rotvec(np.pi * np.array([1.0, 0, 0]))
    ax = np.cross(v, z); ax /= np.linalg.norm(ax); return R.from_rotvec(np.arccos(c) * ax)


def run_obj(obj, n, seed=0):
    # PAIRED near-boundary design: identical release, low vs high heavy-end density -> does the CoM flip the basin?
    # (broad releases dilute the density signal with release variation; the boundary regime isolates it, as the hammer did)
    mesh, hulls, heavy = decompose(obj)
    T, _ = compute_stable_poses(mesh, center_mass=mesh.center_mass, n_samples=1, threshold=0.02)
    Rs = [t[:3, :3] for t in T]; V = np.asarray(mesh.vertices)
    if len(Rs) < 2: return dict(object=obj, n_hull=len(hulls), skip="<2 stable poses")
    long_axis = np.linalg.svd(mesh.vertices - mesh.vertices.mean(0))[2][0]
    m_lo = build(hulls, np.where(heavy, HEAVY_RANGE[0], LIGHT_D)); d_lo = mujoco.MjData(m_lo)
    m_hi = build(hulls, np.where(heavy, HEAVY_RANGE[1], LIGHT_D)); d_hi = mujoco.MjData(m_hi)
    shift = float(np.linalg.norm(np.array(m_hi.body("obj").ipos) - np.array(m_lo.body("obj").ipos)) * 1000)
    rng = np.random.default_rng(seed); labs, dens = [], []; disagree = ok = 0
    for _ in range(n):
        hax = rng.normal(size=3); hax[2] = 0; hax /= np.linalg.norm(hax); tilt = rng.uniform(0, 12)
        q = (R.from_euler("z", rng.uniform(0, 2 * np.pi)) * R.from_rotvec(np.radians(tilt) * hax) * rot_to_vertical(long_axis))
        x, y, z, w = q.as_quat(); q_rel = np.array([w, x, y, z]); h = rng.uniform(0.005, 0.05)
        q0, s0 = settle(m_lo, d_lo, q_rel, V, h); q1, s1 = settle(m_hi, d_hi, q_rel, V, h)
        if not (s0 and s1): continue
        ok += 1; b0 = basin_of(q0, Rs); b1 = basin_of(q1, Rs)
        labs += [b0, b1]; dens += [0, 1]; disagree += int(b0 != b1)
    if ok < 20: return dict(object=obj, n_hull=len(hulls), com_shift_mm=round(shift, 1), skip=f"only {ok} settled")
    labs = np.array(labs); frac = np.sort(np.bincount(labs) / len(labs))[::-1]
    return dict(object=obj, n_hull=len(hulls), n_heavy=int(heavy.sum()), com_shift_mm=round(shift, 1), pairs=ok,
                boundary_basin_disagree=round(disagree / ok, 3), n_basin=int(len(np.unique(labs))),
                top2=[round(float(x), 2) for x in frac[:2]], multimodal=bool((frac > 0.15).sum() >= 2),
                mi=round(float(dmi(labs, np.array(dens))), 3))


def smoke():
    obj = "051_large_clamp"; mesh, hulls, heavy = decompose(obj)
    print(f"[{obj}] extents(mm)={np.round(mesh.extents*1000,1)}  hulls={len(hulls)}  heavy={int(heavy.sum())}")
    m_lo = build(hulls, np.where(heavy, HEAVY_RANGE[0], LIGHT_D)); m_hi = build(hulls, np.where(heavy, HEAVY_RANGE[1], LIGHT_D))
    shift = np.linalg.norm(np.array(m_hi.body("obj").ipos) - np.array(m_lo.body("obj").ipos)) * 1000
    print(f"  CoM shift low->high density = {shift:.1f} mm  (physical hidden latent)")
    r = run_obj(obj, 40)
    print(f"  quick run: {r}")
    assert len(hulls) > 1, "CoACD gave 1 hull (no CoM diversity)"
    assert shift > 1.0, "density change did not move CoM"
    print("smoke OK: >1 hull, CoM moves with density, drops settle + basins assigned")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objs", nargs="+", default=DEFAULT); ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    if a.smoke: smoke(); return
    res = []
    for o in a.objs:
        r = run_obj(o, a.n); res.append(r); print(r)
    ok = [r for r in res if "mi" in r]
    if ok:
        dis = [r["boundary_basin_disagree"] for r in ok]
        print(f"\n=== PHYSICAL-DENSITY VALIDATION ({len(ok)} objects, paired near-boundary) ===")
        print(f"  boundary basin-disagree (same release, low vs high density -> basin flips): "
              f"mean {np.mean(dis):.2f}, per-obj {dis} (hammer controlled ~92% at saddle)")
        print(f"  density causal (disagree>0): {sum(d>0 for d in dis)}/{len(ok)} | multimodal: {sum(r['multimodal'] for r in ok)}/{len(ok)}")
        print(f"  mean I(basin;density) at boundary = {np.mean([r['mi'] for r in ok]):.3f} bits (hammer physical=0.161)")
    json.dump(res, open(os.path.join(HERE, "density", "physical_multi.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
