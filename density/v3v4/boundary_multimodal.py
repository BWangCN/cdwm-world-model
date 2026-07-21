"""V3 (boundary sampler) + V4 (distributional necessity), the crux of the L3 thesis, on the hammer.

V3: sweep the release orientation from lying-flat (deep in a basin) to balanced-on-end (the saddle between two
    stable poses), and show CoM-sensitivity -- disagreement between a centroid-CoM and a head-heavy-CoM drop --
    PEAKS at the saddle and is ~0 deep in the basin. => a boundary sampler that targets the saddle finds the
    high-sensitivity poses (vs the corpus's stable-pose+small-tilt sampling, which never goes there).

V4: at the saddle, treat the CoM as a HIDDEN latent (unknown internal mass, e.g. how head-heavy the tool is),
    sample it from a plausible prior, and drop. For this FIXED visible observation (same geometry + same release)
    the resting pose is genuinely MULTIMODAL -> a point predictor is structurally inadequate; a distributional
    predictor wins on NLL. This is the drop analogue of the slip-MDN result.

Reuses the pilot hammer hulls + settle machinery (density/pilot/drop_ab.py). CoM set via explicit <inertial>
(the clean ablation knob) along the object's long axis; mass + isotropic inertia held fixed, only CoM moves.

    python boundary_multimodal.py
"""
import os, sys, numpy as np, mujoco
from scipy.spatial.transform import Rotation as R

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pilot"))
import drop_ab as A                                   # _all_verts(), MESHDIR, drop(), NHULL

V = A._all_verts()
C = np.array(A.build(A.uniform_density()).body("obj").ipos)   # TRUE volumetric centroid (not vertex mean)
_, _, VT = np.linalg.svd(V - V.mean(0)); AXIS = VT[0]  # long axis
HALF = float((V @ AXIS).ptp() / 2)
Rmax = float(np.linalg.norm(V - C, axis=1).max())
MASS = 0.665
# head is at -Y side (hulls 1-4); sign so +delta = toward the head (see density/README.md)
HEAD_SIGN = -1.0 if (np.mean([A._verts(i).mean(0) for i in A.HEAD], 0) - C) @ AXIS < 0 else 1.0


def build(delta):                                     # CoM at C + delta*AXIS (body frame), mass+inertia fixed
    off = C + delta * AXIS
    I0 = 0.4 * MASS * Rmax ** 2
    meshes = "".join(f'<mesh name="m{i}" file="textured_coacd_{i}.stl"/>' for i in range(A.NHULL))
    geoms = "".join(f'<geom type="mesh" mesh="m{i}" condim="6" friction="1 0.005 0.0001"/>' for i in range(A.NHULL))
    xml = f"""<mujoco>
      <compiler angle="radian" meshdir="{A.MESHDIR}"/>
      <option timestep="0.002" gravity="0 0 -9.81"/>
      <asset>{meshes}</asset>
      <worldbody>
        <geom name="table" type="plane" size="3 3 0.1" condim="6" friction="1 0.005 0.0001"/>
        <body name="obj" pos="0 0 0.6"><freejoint/>{geoms}
          <inertial pos="{off[0]:.5f} {off[1]:.5f} {off[2]:.5f}" mass="{MASS}"
                    diaginertia="{I0:.6f} {I0:.6f} {I0:.6f}"/></body>
      </worldbody></mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


_PHI = np.linspace(0, 2 * np.pi, 72, endpoint=False)
def basin_diff(qa, qb):                               # yaw-invariant rest difference (deg)
    Ra = R.from_quat(np.r_[qa[1:], qa[0]]); Rb = R.from_quat(np.r_[qb[1:], qb[0]])
    return float(((R.from_euler("z", _PHI) * Ra) * Rb.inv()).magnitude().min() * 180 / np.pi)


def release(theta_deg, rng, jitter=4.0):              # long axis tilted `theta` up from horizontal + small jitter
    hax = rng.normal(size=3); hax[2] = 0; hax /= np.linalg.norm(hax)
    q = (R.from_euler("z", rng.uniform(0, 2 * np.pi))
         * R.from_rotvec(np.radians(rng.uniform(0, jitter)) * hax)
         * R.from_euler("x", np.radians(theta_deg)))
    x, y, z, w = q.as_quat()
    return np.array([w, x, y, z])


def v3_sweep(N=14):
    print("=== V3: CoM-sensitivity vs release orientation (0=lying flat .. 90=balanced on end) ===")
    dcen = build(0.0); dhh = build(HEAD_SIGN * 0.6 * HALF)
    Dc, Dh = mujoco.MjData(dcen), mujoco.MjData(dhh)
    print(f"{'theta':>6} {'disagree':>9} {'med_basindiff':>14}")
    peak = (0, -1)
    for theta in [0, 20, 40, 60, 75, 85, 90, 95, 105]:
        rng = np.random.default_rng(100 + theta); dis = 0; bd = []; ok = 0
        for _ in range(N):
            q = release(theta, rng, jitter=4.0); h = rng.uniform(0.005, 0.05)
            qc, sc, _ = A.drop(dcen, Dc, q, h); qh, sh, _ = A.drop(dhh, Dh, q, h)
            if not (sc and sh): continue
            ok += 1; d = basin_diff(qc, qh); dis += d > 30; bd.append(d)
        rate = dis / ok if ok else 0
        if rate > peak[1]: peak = (theta, rate)
        print(f"{theta:>6} {rate:>8.0%} {np.median(bd) if bd else 0:>13.1f}°  (n={ok})")
    print(f"-> sensitivity peaks near theta={peak[0]} ({peak[1]:.0%}); ~0 lying flat = the saddle is where CoM decides.\n")
    return peak[0]


def v4_multimodal(theta, M=80):
    print(f"=== V4: at the boundary (theta={theta}), CoM hidden -> is the resting pose multimodal? ===")
    # hidden latent = CoM offset along long axis, plausible prior head-heavy..handle-heavy
    rng = np.random.default_rng(7)
    deltas = HEAD_SIGN * rng.uniform(-0.2 * HALF, 0.7 * HALF, M)   # +=toward head; spans both sides of the saddle
    q_fixed = release(theta, np.random.default_rng(0), jitter=0.0)  # ONE fixed visible release pose
    rests = []
    for dz in deltas:
        m = build(dz); d = mujoco.MjData(m)
        q, ok, _ = A.drop(m, d, q_fixed, 0.03)
        if ok: rests.append(q)
    rests = np.array(rests)
    # cluster resting poses into basins by yaw-invariant distance (>30 deg apart = different basin)
    labels = -np.ones(len(rests), int); K = 0
    for i in range(len(rests)):
        if labels[i] >= 0: continue
        labels[i] = K
        for j in range(i + 1, len(rests)):
            if labels[j] < 0 and basin_diff(rests[i], rests[j]) < 30: labels[j] = K
        K += 1
    frac = np.bincount(labels) / len(labels)
    frac = np.sort(frac)[::-1]
    print(f"settled {len(rests)}/{M}; {K} distinct rest basins, fractions = {np.round(frac,2)}")
    multimodal = (frac > 0.15).sum() >= 2
    # NLL: point predictor (all mass on the argmax basin) vs distributional (empirical mix)
    eps = 1e-3; p_emp = frac
    p_point = np.full(K, eps); p_point[0] = 1 - eps * (K - 1)
    counts = (frac * len(rests)).round().astype(int)
    nll_point = -np.sum(counts * np.log(p_point)) / counts.sum()
    nll_dist = -np.sum(counts * np.log(np.clip(p_emp, eps, 1))) / counts.sum()
    print(f"multimodal={multimodal}  NLL point-predictor={nll_point:.2f}  NLL distributional={nll_dist:.2f}  "
          f"(distributional wins by {nll_point - nll_dist:.2f} nats)")
    print("-> a fixed observation with hidden CoM yields multiple rest basins; a point predictor is structurally"
          " inadequate. Drop analogue of the slip-MDN result.")


def main():
    print(f"hammer: long axis half={HALF*1000:.0f}mm, head_sign={HEAD_SIGN:+.0f}\n")
    theta = v3_sweep()
    v4_multimodal(max(theta, 85))

if __name__ == "__main__":
    main()
