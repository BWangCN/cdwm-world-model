"""Boundary-regime hull-vs-CoACD robustness study (see docs/00_overview.md "Geometry: the convex-hull scope").

The global concavity audit (concavity_audit.py) measures whether the hull looks right AT REST IN GENERAL. This
measures something different and more relevant: does the hull distort the SPECIFIC near-boundary Gate B episodes,
compared to a faithful CoACD convex-decomposition of the real mesh? Confound-free 3-arm design per episode --
identical release + identical explicit-inertial hidden CoM; only the collision geometry differs (single hull vs
CoACD multi-hull) -- plus a 1.5-degree release-wobble control that isolates ordinary near-boundary sensitivity
(the "floor") from a geometry effect (the "gap" beyond that floor). "Same resting pose" = yaw-invariant
orientation match (< 30 deg), not lossy stable-pose binning (that binning was shown to mis-rank several objects).

    python -m drop.hull_vs_coacd --obj <object> --n 250 --out gateb_runs/hull_vs_coacd/<object>.json
    python -m drop.hull_vs_coacd --smoke
"""
import os, json, argparse
import numpy as np, mujoco, trimesh, coacd
from scipy.spatial.transform import Rotation as R
from trimesh.poses import compute_stable_poses
from common.paths import OBJECTS, GATEB_DIR, PCDIR
from drop.density.v2 import drop_sweep as DS
from drop.gateb.generate_obj import stable_orientations, basin_of
coacd.set_log_level("error")


def align_mesh(obj):                                          # mesh -> the point-cloud hull frame (same centering as DS.hull)
    m = trimesh.load(f"{OBJECTS}/{obj}/mesh.obj", force="mesh")
    mu = np.load(f"{PCDIR}/{obj}.npz")["mu"].astype(np.float64)
    m.apply_scale((mu.max(0) - mu.min(0)).max() / (m.vertices.max(0) - m.vertices.min(0)).max())
    m.apply_translation(-np.asarray(m.convex_hull.vertices).mean(0))
    return m


def _geod(qa, qb):
    dR = R.from_quat(np.r_[qb[1:], qb[0]]) * R.from_quat(np.r_[qa[1:], qa[0]]).inv()
    return float(np.degrees(dR.magnitude()))


def settle_robust(m, d, quat, V, drop_h, max_steps=6000, win=250, tol=5.0):
    """Settled = ORIENTATION stops drifting (net rotation over `win` steps < tol) and not translating. Robust to
    the micro-rolling a decomposed surface shows near a raw velocity threshold; applied to both arms identically."""
    Rr = R.from_quat(np.r_[quat[1:], quat[0]]); pz = drop_h - Rr.apply(V)[:, 2].min()
    mujoco.mj_resetData(m, d); d.qpos[:3] = [0, 0, pz]; d.qpos[3:7] = quat; d.qvel[:] = 0; mujoco.mj_forward(m, d)
    hist = []
    for st in range(max_steps):
        mujoco.mj_step(m, d); hist.append(d.qpos[3:7].copy())
        if len(hist) > win and _geod(hist[-win], hist[-1]) < tol and np.linalg.norm(d.qvel[:3]) < 0.02:
            return hist[-1].copy(), True
    return d.qpos[3:7].copy(), False


def build_coacd(hulls, delta, axis, r, mass=0.2):             # same explicit-inertial as DS.build; only geometry differs
    meshes = "".join(f'<mesh name="h{i}" vertex="{" ".join(f"{x:.5f}" for x in h.flatten())}"/>' for i, h in enumerate(hulls))
    geoms = "".join(f'<geom type="mesh" mesh="h{i}" condim="6" friction="1 0.005 0.0001"/>' for i in range(len(hulls)))
    I0 = 0.4 * mass * r ** 2; off = delta * axis
    xml = f"""<mujoco><option timestep="0.002" gravity="0 0 -9.81"/>
      <asset>{meshes}</asset>
      <worldbody><geom name="table" type="plane" size="3 3 0.1" condim="6" friction="1 0.005 0.0001"/>
        <body name="obj" pos="0 0 1"><freejoint/>{geoms}
          <inertial pos="{off[0]:.5f} {off[1]:.5f} {off[2]:.5f}" mass="{mass}" diaginertia="{I0:.6f} {I0:.6f} {I0:.6f}"/>
        </body></worldbody></mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def _wobble(quat, deg, ax):
    qp = (R.from_rotvec(np.radians(deg) * ax) * R.from_quat(np.r_[quat[1:], quat[0]])).as_quat()
    return np.r_[qp[3], qp[:3]]


def run_obj(obj, N, seed=0):
    H = DS.hull(obj)
    m = align_mesh(obj)
    parts = coacd.run_coacd(coacd.Mesh(np.asarray(m.vertices), np.asarray(m.faces)), threshold=0.06)
    hulls = [np.asarray(v, float) for v, f in parts]; Vc = np.vstack(hulls)
    rc = float(np.linalg.norm(Vc - Vc.mean(0), axis=1).max())
    d = np.load(f"{GATEB_DIR}/{obj}_gateb_s0.npz", allow_pickle=True)
    relq, dh, com = d["release_quat"], d["drop_h"], d["com"]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(relq), min(N, len(relq)), replace=False)
    hc = {}; cc = {}; ok = same_geom = same_floor = 0; gdiff = []
    for j in idx:
        j = int(j); ai = int(com[j][0]); delta = round(float(com[j][1]) / 0.003) * 0.003
        key = (ai, round(delta, 3))
        if key not in hc:
            mh = DS.build(H, delta, H["axes"][ai]); hc[key] = (mh, mujoco.MjData(mh))
            mcd = build_coacd(hulls, delta, H["axes"][ai], rc); cc[key] = (mcd, mujoco.MjData(mcd))
        mh, dhh = hc[key]; mcd, dcc = cc[key]
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
        qh, sh = settle_robust(mh, dhh, relq[j], H["V"], float(dh[j]))
        qp, sp = settle_robust(mh, dhh, _wobble(relq[j], 1.5, ax), H["V"], float(dh[j]))
        qc, sc = settle_robust(mcd, dcc, relq[j], Vc, float(dh[j]))
        if not (sh and sp and sc):
            continue
        ok += 1; g = DS.basin_diff(qh, qc)
        gdiff.append(g); same_geom += int(g < 30); same_floor += int(DS.basin_diff(qh, qp) < 30)
    if ok < 20:
        return dict(object=obj, n_coacd_hull=len(hulls), pairs=ok, skip=f"only {ok} settled")
    return dict(object=obj, n_coacd_hull=len(hulls), pairs=ok, settle_rate=round(ok / len(idx), 2),
                agree_hull_vs_coacd=round(same_geom / ok, 3),
                agree_floor_1p5deg=round(same_floor / ok, 3),
                geom_gap=round((same_floor - same_geom) / ok, 3),
                median_basin_diff_deg=round(float(np.median(gdiff)), 1))


def smoke():
    obj = "021_bleach_cleanser"; H = DS.hull(obj); Rs_hull, _ = stable_orientations(H)
    d = np.load(f"{GATEB_DIR}/{obj}_gateb_s0.npz", allow_pickle=True)
    okrep = 0
    for j in range(15):
        ai = int(d["com"][j][0]); delta = round(float(d["com"][j][1]) / 0.003) * 0.003
        mdl = DS.build(H, delta, H["axes"][ai])
        q, s = DS.settle(mdl, mujoco.MjData(mdl), d["release_quat"][j], H["V"], float(d["drop_h"][j]), 4000)
        okrep += int(s and basin_of(q, Rs_hull) == int(d["basin"][j]))
    print(f"[{obj}] hull-arm reproduces stored basin on {okrep}/15 episodes (deterministic sanity)")
    r = run_obj(obj, 100); print("  CoACD run:", r)   # n=100: this object's settle_rate ~0.33, need >=20 pairs
    assert r.get("pairs", 0) >= 20 and "agree_hull_vs_coacd" in r
    print("smoke OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj"); ap.add_argument("--n", type=int, default=250); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None); a = ap.parse_args()
    if a.smoke:
        smoke(); return
    r = run_obj(a.obj, a.n); print(json.dumps(r))
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(r, open(a.out, "w"))


if __name__ == "__main__":
    main()
