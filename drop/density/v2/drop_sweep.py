"""V2 object-diversity sensitivity sweep (Gate A.5). Does moving the CoM change WHICH basin an object settles
into -- for the audit's high-transition objects, and NOT for deep-stable negative controls?

Codex-endorsed design: explicit <inertial> CoM offset = the clean ABLATION knob (mass + inertia held fixed;
CoM is the ONLY varied quantity, isotropic inertia so the comparison is confound-free). Offset is RELATIVE to
object scale (fraction of the long-axis half-extent). Geometry = convex hull of the WM's own point cloud
(= the support geometry that sets stable poses). Paired: identical release pose, CoM at centroid vs shifted.

Two regimes per object (parallels the hammer pilot):
  flat     -- settle to a stable pose, then tilt ~U[0,15] deg (the CORPUS's sampling)
  boundary -- long axis vertical (balanced on end) + small tilt (near-tipping)

Metric = basin-disagreement rate (do the two CoM conditions settle in different coarse basins?) + median
|Δ rest-orientation|. GO signal for the CoG contribution = high disagreement on high-transition objects,
low on negative controls. NOT a realism claim (ablation only); the dataset build uses per-hull density.

    python drop_sweep.py OBJ1 OBJ2 ...        # default: audit-selected set; writes v2_results.json
"""
import os, sys, json, numpy as np, mujoco
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R

from common.paths import PCDIR
REF_MASS = 0.2                       # kg, held FIXED across conditions (ablation)
OFFSET_FRAC = 0.35                   # CoM shift = 0.35 * long-axis half-extent (relative to object scale)

# audit-selected: high-transition (CoG-sensitive candidates) + deep-stable (negative controls)
DEFAULT = dict(
    high=["003_cracker_box", "Android_Figure_Chrome", "Razer_Naga_MMO_Gaming_Mouse",
          "Quercetin_500", "Prostate_Optimizer"],
    neg=["Schleich_African_Black_Rhino", "Thomas_Friends_Woodan_Railway_Henry",
         "Unmellow_Yellow_Tieks_Neon_Patent_Leather_Ballet_Flats"])


def hull(obj):
    f = f"{PCDIR}/{obj}.npz"
    if not os.path.exists(f): return None
    mu = np.load(f)["mu"].astype(np.float64)
    if len(mu) > 4000: mu = mu[np.random.default_rng(0).choice(len(mu), 4000, replace=False)]
    V = mu[ConvexHull(mu).vertices]
    V = V - V.mean(0)                                   # center
    _, _, vt = np.linalg.svd(V - V.mean(0)); axes = vt  # rows = principal axes (0 = longest)
    half = np.array([(V @ axes[i]).ptp() / 2 for i in range(3)])  # per-axis half extent
    r = np.linalg.norm(V, axis=1).max()
    return dict(V=V, axes=axes, half=half, r=r)


def build(H, delta, axis):                             # delta along principal `axis` (body frame)
    verts = " ".join(f"{x:.5f}" for x in H["V"].flatten())
    I0 = 0.4 * REF_MASS * H["r"] ** 2                   # isotropic, fixed -> only CoM varies
    off = delta * axis
    xml = f"""<mujoco>
      <option timestep="0.002" gravity="0 0 -9.81"/>
      <asset><mesh name="m" vertex="{verts}"/></asset>
      <worldbody>
        <geom name="table" type="plane" size="3 3 0.1" condim="6" friction="1 0.005 0.0001"/>
        <body name="obj" pos="0 0 1">
          <freejoint/>
          <geom type="mesh" mesh="m" condim="6" friction="1 0.005 0.0001"/>
          <inertial pos="{off[0]:.5f} {off[1]:.5f} {off[2]:.5f}" mass="{REF_MASS}"
                    diaginertia="{I0:.6f} {I0:.6f} {I0:.6f}"/>
        </body>
      </worldbody></mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def settle(m, d, quat, V, drop_h, max_steps=2000):
    Rr = R.from_quat(np.r_[quat[1:], quat[0]])
    pz = drop_h - (Rr.apply(V))[:, 2].min()
    mujoco.mj_resetData(m, d); d.qpos[:3] = [0, 0, pz]; d.qpos[3:7] = quat; d.qvel[:] = 0
    mujoco.mj_forward(m, d); still = 0
    for _ in range(max_steps):
        mujoco.mj_step(m, d)
        if np.linalg.norm(d.qvel[:3]) < 0.006 and np.linalg.norm(d.qvel[3:6]) < 0.1047:
            still += 1
            if still >= 75: break
        else: still = 0
    return d.qpos[3:7].copy(), still >= 75


def geod(qa, qb):
    dR = R.from_quat(np.r_[qb[1:], qb[0]]) * R.from_quat(np.r_[qa[1:], qa[0]]).inv()
    return float(np.degrees(dR.magnitude()))


_PHI = np.linspace(0, 2 * np.pi, 72, endpoint=False)
def basin_diff(qa, qb):                                # yaw-INVARIANT rest-pose difference: min over world-z spin
    Ra = R.from_quat(np.r_[qa[1:], qa[0]]); Rb = R.from_quat(np.r_[qb[1:], qb[0]])
    zsp = R.from_euler("z", _PHI)                      # world-frame yaw family (same face down = same basin)
    d = (zsp * Ra) * Rb.inv()
    return float(np.degrees(d.magnitude().min()))      # small => same basin (just yaw); large => different face down


def rot_to_vertical(axis):                             # body `axis` -> world +z (robust, no scipy warning)
    z = np.array([0.0, 0, 1]); v = axis / np.linalg.norm(axis)
    c = np.dot(v, z)
    if c > 0.9999: return R.identity()
    if c < -0.9999: return R.from_rotvec(np.pi * np.array([1.0, 0, 0]))
    ax = np.cross(v, z); ax /= np.linalg.norm(ax)
    return R.from_rotvec(np.arccos(c) * ax)


def stable_pose(m, d, H, rng):                         # drop from a random orientation -> a stable pose quat
    x, y, z, w = R.random(random_state=rng).as_quat()
    q, ok = settle(m, d, np.array([w, x, y, z]), H["V"], 0.05)
    return q if ok else np.array([1.0, 0, 0, 0])


def release(rng, H, regime, base_q):
    hax = rng.normal(size=3); hax[2] = 0; hax /= np.linalg.norm(hax)
    if regime == "flat":
        tilt = rng.uniform(0, 15); q0 = R.from_quat(np.r_[base_q[1:], base_q[0]])
    else:                                              # boundary: long axis -> vertical (balanced on end)
        tilt = rng.uniform(0, 12); q0 = rot_to_vertical(H["axes"][0])
    q = R.from_euler("z", rng.uniform(0, 2 * np.pi)) * R.from_rotvec(np.radians(tilt) * hax) * q0
    x, y, z, w = q.as_quat()
    return np.array([w, x, y, z]), rng.uniform(0.005, 0.080)


def sweep_obj(obj, N=30, thr=30.0):
    H = hull(obj)
    if H is None: return None
    m0 = build(H, 0.0, H["axes"][0]); d0 = mujoco.MjData(m0)
    rng = np.random.default_rng(0)
    base_q = stable_pose(m0, d0, H, rng)
    out = dict(object=obj, long_mm=round(float(H["half"][0]) * 2000, 1), axes={})
    # sweep the CoM offset along each principal axis (axis 0 long .. 2 short); folds in the V1 out-of-plane test
    for ai, aname in [(0, "long"), (1, "mid"), (2, "short")]:
        delta = OFFSET_FRAC * H["half"][ai]
        m1 = build(H, delta, H["axes"][ai]); d1 = mujoco.MjData(m1)
        shift_mm = float(np.linalg.norm(np.array(m1.body("obj").ipos) - np.array(m0.body("obj").ipos)) * 1000)
        rec = dict(shift_mm=round(shift_mm, 1))
        for regime in ["flat", "boundary"]:
            rng = np.random.default_rng(1000 + ai)     # SAME releases across axes for comparability
            diff = 0; dd = []; ok = 0
            for _ in range(N):
                q_rel, h = release(rng, H, regime, base_q)
                q0, s0 = settle(m0, d0, q_rel, H["V"], h)
                q1, s1 = settle(m1, d1, q_rel, H["V"], h)
                if not (s0 and s1): continue
                ok += 1; bd = basin_diff(q0, q1); diff += bd > thr; dd.append(bd)
            rec[regime] = dict(n=ok, basin_disagree=round(diff / ok, 3) if ok else None,
                               med_basindiff=round(float(np.median(dd)), 1) if dd else None)
        out["axes"][aname] = rec
    # headline = worst-case sensitivity across CoM directions, boundary regime
    bnd = [out["axes"][a]["boundary"]["basin_disagree"] for a in out["axes"]
           if out["axes"][a]["boundary"]["basin_disagree"] is not None]
    flt = [out["axes"][a]["flat"]["basin_disagree"] for a in out["axes"]
           if out["axes"][a]["flat"]["basin_disagree"] is not None]
    out["max_boundary_disagree"] = max(bnd) if bnd else None
    out["max_flat_disagree"] = max(flt) if flt else None
    return out


def main():
    objs = sys.argv[1:] or (DEFAULT["high"] + DEFAULT["neg"])
    neg = set(DEFAULT["neg"])
    N = int(os.environ.get("V2_N", "30"))
    results = []
    print(f"{'object':40s} {'kind':>5} {'flat_dis':>9} {'bnd_dis':>8}  (max over CoM directions; disagree = rest differs >30° yaw-inv)")
    for o in objs:
        r = sweep_obj(o, N=N)
        if r is None: print(f"{o:40s}  (no point cloud, skipped)"); continue
        results.append(r)
        kind = "NEG" if o in neg else "high"
        print(f"{o:40s} {kind:>5} {str(r['max_flat_disagree']):>9} {str(r['max_boundary_disagree']):>8}")
    outp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v2_results.json")
    json.dump(results, open(outp, "w"), indent=2); print(f"\nwrote {outp}")

if __name__ == "__main__":
    main()
