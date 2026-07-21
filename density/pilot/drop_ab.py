"""T4 A/B PILOT (go/no-go): does the object's center of gravity change where it comes to rest?

Compares, on the REAL YCB hammer mesh (elpis-lab CoACD, 5 hulls), two mass distributions dropped from
IDENTICAL release poses (paired):
  UNIFORM   — every hull the same density  -> CoM at the geometric centroid   (== the current drop corpus)
  HEAD_HEAVY— handle hull light + head hulls dense -> CoM toward the head, reproducing the real 665 g
              (handle ~0.5 g/cm^3 hollow-fiberglass + head ~4.2 g/cm^3 -> total ~0.665 kg, head-weighted)

For a sweep of release tilts we drop N paired episodes, settle under MuJoCo, and measure per condition:
  - net_rot_from_release (deg), rest "basin" (which body face ends up down).
Then report, per tilt, how often the two CoM conditions DISAGREE on the rest basin and the median
|Δ rest-orientation|. If CoM materially moves the outcome -> GO (build the dataset); if the effect is tiny
across all tilts -> NO-GO (Codex's shape-dominated risk).  See ../../notes/drop_density_todo.md (T4).

    python drop_ab.py            # prints CoM check + per-tilt effect table

ponytail: first-pass mini-harness, NOT the colleague's corpus-v1 harness. Physics matched to the dataset
README where it matters (condim=6, rolling mu=1e-4, zero-vel release, 4 s ceiling, at-rest thresholds).
"""
import os, numpy as np, mujoco
from scipy.spatial.transform import Rotation as R

HERE = os.path.dirname(os.path.abspath(__file__))
MESHDIR = os.path.join(HERE, "hammer")
NHULL = 5
HANDLE = [0]; HEAD = [1, 2, 3, 4]                 # from CoACD geometry inspection (see density/README.md)
REAL_MASS = 0.665                                  # YCB 048_hammer, kg

# --- densities (kg/m^3) ---
def uniform_density():                             # single value -> total REAL_MASS; CoM at centroid regardless
    vol = _total_volume()
    return {i: REAL_MASS / vol for i in range(NHULL)}

def head_heavy_density():                          # handle light + head dense; ~same total mass, CoM shifted
    dens = {i: 500.0 for i in HANDLE}              # hollow fiberglass handle
    vh = sum(_hull_volume(i) for i in HEAD)
    m_handle = sum(500.0 * _hull_volume(i) for i in HANDLE)
    d_head = (REAL_MASS - m_handle) / vh           # solve so total == REAL_MASS
    for i in HEAD: dens[i] = d_head
    return dens

# --- mesh volumes (scipy convex hull of each STL's verts) ---
_VCACHE = {}
def _verts(i):
    if i not in _VCACHE:
        fn = os.path.join(MESHDIR, f"textured_coacd_{i}.stl")
        with open(fn, "rb") as f:
            f.read(80); n = int(np.frombuffer(f.read(4), "<u4")[0])
            d = np.frombuffer(f.read(n * 50), np.uint8).reshape(n, 50)
        V = np.stack([np.frombuffer(d[k, 12:48].tobytes(), "<f4").reshape(3, 3) for k in range(n)]).reshape(-1, 3)
        _VCACHE[i] = np.unique(V, axis=0)
    return _VCACHE[i]
def _hull_volume(i):
    from scipy.spatial import ConvexHull
    return float(ConvexHull(_verts(i)).volume)
def _total_volume():
    return sum(_hull_volume(i) for i in range(NHULL))
def _all_verts():
    return np.concatenate([_verts(i) for i in range(NHULL)])

# --- MJCF ---
def build(dens):
    meshes = "".join(f'<mesh name="m{i}" file="textured_coacd_{i}.stl"/>' for i in range(NHULL))
    geoms = "".join(
        f'<geom name="h{i}" type="mesh" mesh="m{i}" density="{dens[i]:.3f}" '
        f'condim="6" friction="1 0.005 0.0001"/>' for i in range(NHULL))
    xml = f"""
<mujoco model="hammer_drop">
  <compiler angle="radian" meshdir="{MESHDIR}"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <asset>{meshes}</asset>
  <worldbody>
    <geom name="table" type="plane" size="2 2 0.1" pos="0 0 0" condim="6" friction="1 0.005 0.0001"/>
    <body name="obj" pos="0 0 0.5">
      <freejoint name="j"/>
      {geoms}
    </body>
  </worldbody>
</mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    return m

def com_local(m):
    return np.array(m.body("obj").ipos), float(m.body("obj").mass[0])

# --- one drop; returns (rest_quat_wxyz, settled_bool, net_rot_deg) ---
V_ALL = None
def drop(m, d, quat_rel, drop_h):
    global V_ALL
    if V_ALL is None: V_ALL = _all_verts()
    Rr = R.from_quat(np.r_[quat_rel[1:], quat_rel[0]])          # wxyz -> xyzw
    wz = (Rr.apply(V_ALL))[:, 2]                                 # world z of verts at this orientation, body@origin
    pz = drop_h - wz.min()                                       # lowest vertex sits drop_h above table
    mujoco.mj_resetData(m, d)
    d.qpos[:3] = [0, 0, pz]; d.qpos[3:7] = quat_rel; d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    still = 0
    for _ in range(2000):                                        # 4 s @ 2 ms
        mujoco.mj_step(m, d)
        v = np.linalg.norm(d.qvel[:3]); w = np.linalg.norm(d.qvel[3:6])
        if v < 0.006 and w < 0.1047:                             # <6 mm/s & <6 deg/s
            still += 1
            if still >= 75: break                                # sustained 150 ms
        else:
            still = 0
    q = d.qpos[3:7].copy()
    dR = R.from_quat(np.r_[q[1:], q[0]]) * Rr.inv()
    return q, still >= 75, float(np.degrees(dR.magnitude()))

def basin(quat_wxyz):                                            # which body face points UP at rest (6 bins)
    up_body = R.from_quat(np.r_[quat_wxyz[1:], quat_wxyz[0]]).inv().apply([0, 0, 1])
    ax = int(np.argmax(np.abs(up_body)))
    return ax * 2 + (0 if up_body[ax] > 0 else 1)

def sample_release(rng, tilt_deg, mode="flat"):
    # base "stable pose": flat = identity (hammer lies on its side); upright = long axis (y) rotated to vertical (z),
    # i.e. balanced on one end -> a metastable pose where CoM height decides which way it topples.
    base = R.identity() if mode == "flat" else R.from_euler("x", -np.pi / 2)
    hax = rng.normal(size=3); hax[2] = 0; hax /= np.linalg.norm(hax)   # random HORIZONTAL tilt axis
    q_tilt = R.from_rotvec(np.radians(tilt_deg) * hax)
    q_yaw = R.from_euler("z", rng.uniform(0, 2 * np.pi))
    q = (q_yaw * q_tilt * base)
    x, y, z, w = q.as_quat()
    return np.array([w, x, y, z]), rng.uniform(0.005, 0.080)     # wxyz, drop_h

def main():
    du, dh = uniform_density(), head_heavy_density()
    mu, md = build(du), build(dh)
    cu, mu_mass = com_local(mu); cd, md_mass = com_local(md)
    shift_mm = np.linalg.norm(cu - cd) * 1000
    print(f"UNIFORM   : mass={mu_mass:.3f} kg  CoM(local)={np.round(cu,4)}")
    print(f"HEAD_HEAVY: mass={md_mass:.3f} kg  CoM(local)={np.round(cd,4)}")
    print(f"CoM shift between conditions: {shift_mm:.1f} mm   (object long axis ~335 mm)\n")
    assert shift_mm > 5, "CoM barely moved -- density map not creating CoG diversity"

    du_d, dh_d = mujoco.MjData(mu), mujoco.MjData(md)
    for mode, tilts in [("flat", [10, 20, 30, 45, 60]),       # dataset regime (stable pose + small tilt)
                        ("upright", [0, 5, 10, 20, 30])]:     # near-tipping regime (balanced on one end)
        print(f"\n=== mode={mode} ===")
        print(f"{'tilt':>5} {'N':>4} {'basin_disagree':>15} {'med|net_rot U|':>15} {'med|net_rot H|':>15} {'med|Δrot|':>11}")
        for tilt in tilts:
            rng = np.random.default_rng(1234 + tilt)
            N = 24; disagree = 0; ru = []; rh = []; drot = []; ok = 0
            for _ in range(N):
                q_rel, h = sample_release(rng, tilt, mode)
                qu, su, nu = drop(mu, du_d, q_rel, h)
                qh, sh, nh = drop(md, dh_d, q_rel, h)
                if not (su and sh): continue
                ok += 1
                disagree += (basin(qu) != basin(qh))
                ru.append(nu); rh.append(nh)
                dR = R.from_quat(np.r_[qh[1:], qh[0]]) * R.from_quat(np.r_[qu[1:], qu[0]]).inv()
                drot.append(np.degrees(dR.magnitude()))
            if ok:
                print(f"{tilt:>5} {ok:>4} {disagree/ok:>14.0%} {np.median(ru):>14.1f}° {np.median(rh):>14.1f}° {np.median(drot):>10.1f}°")
    print("\nGO if basin_disagree / |Δrot| are large in the near-tipping (upright) regime -> CoM decides the outcome")
    print("there; if ~0 even upright -> NO-GO. flat = the current dataset's stable-pose+small-tilt sampling.")

if __name__ == "__main__":
    main()
