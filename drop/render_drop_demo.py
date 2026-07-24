"""Demo videos for the drop paper. SAME object + SAME near-boundary release, SWEEP the hidden CoM -> the resting
BASIN flips. Illustrates why a distributional WM is needed here: an unobserved inertial latent (CoM) controls a
multimodal outcome. This is a controlled ILLUSTRATION (release fixed, CoM varied); the quantitative claims are the
CIs elsewhere. Re-simulates each drop with mujoco.Renderer (EGL), since the gateb .npz stores only release/CoM/rest.
Reuses density/v2/drop_sweep (hull/build/settle physics) + gateb/generate_obj (stable poses / basin_of).

    MUJOCO_GL=egl python render_drop_demo.py --objs 011_banana 050_medium_clamp Razer_Taipan_Black_Ambidextrous_Gaming_Mouse
    MUJOCO_GL=egl python render_drop_demo.py --smoke        # EGL render+encode path only, no search
"""
import os, sys, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, mujoco, imageio
from scipy.spatial.transform import Rotation as R
HERE = os.path.dirname(os.path.abspath(__file__))
from drop.density.v2 import drop_sweep as DS
from drop.gateb.generate_obj import stable_orientations, basin_of
try:
    from PIL import Image, ImageDraw, ImageFont
    HAVE_PIL = True
    try:
        FONT = ImageFont.truetype("DejaVuSans.ttf", 26)
    except Exception:
        import matplotlib
        FONT = ImageFont.truetype(os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf", "DejaVuSans.ttf"), 26)
except Exception:
    HAVE_PIL = False; FONT = None

OUT = os.path.join(HERE, "notes", "demo_videos"); os.makedirs(OUT, exist_ok=True)
W, H_ = 800, 600
DELTAS = [-0.35, -0.15, 0.15, 0.35]                          # CoM sweep as fraction of the axis half-extent


def build_render(H, delta, axis):                           # physics identical to DS.build; adds light/floor/color for rendering
    verts = " ".join(f"{x:.5f}" for x in H["V"].flatten())
    I0 = 0.4 * DS.REF_MASS * H["r"] ** 2; off = delta * axis
    xml = f"""<mujoco>
      <option timestep="0.002" gravity="0 0 -9.81"/>
      <visual><headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6"/><global offwidth="{W}" offheight="{H_}"/></visual>
      <asset>
        <mesh name="m" vertex="{verts}"/>
        <texture name="grid" type="2d" builtin="checker" rgb1=".18 .26 .34" rgb2=".10 .15 .20" width="512" height="512"/>
        <material name="grid" texture="grid" texrepeat="8 8" reflectance=".1"/>
      </asset>
      <worldbody>
        <light pos="0.3 0.3 1.2" dir="-0.2 -0.2 -1" diffuse="0.8 0.8 0.8"/>
        <geom name="table" type="plane" size="3 3 0.1" condim="6" friction="1 0.005 0.0001" material="grid"/>
        <body name="obj" pos="0 0 1">
          <freejoint/>
          <geom type="mesh" mesh="m" condim="6" friction="1 0.005 0.0001" rgba="0.92 0.60 0.20 1"/>
          <inertial pos="{off[0]:.5f} {off[1]:.5f} {off[2]:.5f}" mass="{DS.REF_MASS}"
                    diaginertia="{I0:.6f} {I0:.6f} {I0:.6f}"/>
        </body>
      </worldbody></mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def settle_frames(m, d, quat, V, drop_h, rnd=None, cam=None, max_steps=4000, every=10):
    Rr = R.from_quat(np.r_[quat[1:], quat[0]]); pz = drop_h - Rr.apply(V)[:, 2].min()
    mujoco.mj_resetData(m, d); d.qpos[:3] = [0, 0, pz]; d.qpos[3:7] = quat; d.qvel[:] = 0
    mujoco.mj_forward(m, d); frames = []; still = 0
    for step in range(max_steps):
        mujoco.mj_step(m, d)
        if rnd is not None and step % every == 0:
            rnd.update_scene(d, camera=cam); frames.append(rnd.render().copy())
        if np.linalg.norm(d.qvel[:3]) < 0.006 and np.linalg.norm(d.qvel[3:6]) < 0.1047:
            still += 1
            if still >= 75: break
        else:
            still = 0
    if rnd is not None and frames:                          # hold the resting frame ~0.5s
        frames += [frames[-1]] * 15
    return frames, d.qpos[3:7].copy()


def label(frames, text):
    if not (HAVE_PIL and frames): return frames
    out = []
    for f in frames:
        im = Image.fromarray(f); dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, W, 38], fill=(0, 0, 0)); dr.text((10, 6), text, fill=(255, 255, 255), font=FONT)
        out.append(np.asarray(im))
    return out


def camera(H):
    cam = mujoco.MjvCamera()                                   # frame the object tightly (was far -> object looked small/rough)
    cam.lookat[:] = [0, 0, max(0.02, 0.25 * H["half"][0])]; cam.distance = max(0.25, 4.5 * float(H["r"]))
    cam.elevation = -16; cam.azimuth = 90
    return cam


def find_demo(obj, n_try=80):                               # find a release+axis where sweeping CoM flips the basin
    H = DS.hull(obj)
    if H is None: return None
    Rs, _ = stable_orientations(H)
    if len(Rs) < 2: return None
    rng = np.random.default_rng(0)
    for _ in range(n_try):
        Rbase = Rs[rng.integers(len(Rs))]
        hax = rng.normal(size=3); hax[2] = 0; hax /= np.linalg.norm(hax)
        tilt = rng.uniform(15, 55); h = rng.uniform(0.005, 0.05)
        q = (R.from_euler("z", rng.uniform(0, 2 * np.pi)) * R.from_rotvec(np.radians(tilt) * hax) * R.from_matrix(Rbase))
        x, y, z, w = q.as_quat(); q_rel = np.array([w, x, y, z])
        for ai in range(3):
            res = []
            for fr in DELTAS:
                delta = fr * H["half"][ai]; m = DS.build(H, delta, H["axes"][ai]); d = mujoco.MjData(m)
                _, qr = settle_frames(m, d, q_rel, H["V"], h)
                res.append((delta, basin_of(qr, Rs)))
            if len({b for _, b in res}) >= 2:               # a basin flip across the CoM sweep -> good demo
                return dict(H=H, Rs=Rs, q_rel=q_rel, h=h, ai=ai, res=res)
    return None


def render_obj(obj):
    demo = find_demo(obj)
    if demo is None: print(f"[{obj}] no basin-flip release found, skip"); return None
    H, ai, q_rel, h = demo["H"], demo["ai"], demo["q_rel"], demo["h"]
    rnd = mujoco.Renderer(build_render(H, 0.0, H["axes"][ai]), height=H_, width=W); cam = camera(H)
    clips = []
    for delta, _ in demo["res"]:
        m = build_render(H, delta, H["axes"][ai]); d = mujoco.MjData(m)
        fr, qr = settle_frames(m, d, q_rel, H["V"], h, rnd, cam)
        b = basin_of(qr, demo["Rs"])
        clips += label(fr, f"{obj}   hidden CoM {delta*1000:+.0f}mm (axis {ai})  ->  basin {b}")
    p = os.path.join(OUT, f"{obj}_com_sweep.mp4")
    imageio.mimsave(p, clips, fps=30, macro_block_size=1)
    basins = [b for _, b in demo["res"]]
    print(f"[{obj}] wrote {p}  ({len(clips)} frames, basins across sweep={basins})")
    return p


def smoke():
    print(f"MUJOCO_GL={os.environ.get('MUJOCO_GL')}  PIL={HAVE_PIL}")
    H = DS.hull("011_banana"); rnd = mujoco.Renderer(build_render(H, 0.0, H["axes"][0]), height=H_, width=W)
    m = build_render(H, 0.0, H["axes"][0]); d = mujoco.MjData(m)
    fr, _ = settle_frames(m, d, np.array([1.0, 0, 0, 0]), H["V"], 0.1, rnd, camera(H), max_steps=200, every=10)
    fr = label(fr, "smoke")
    p = os.path.join(OUT, "_smoke.mp4"); imageio.mimsave(p, fr, fps=20, macro_block_size=1)
    assert os.path.getsize(p) > 0 and len(fr) > 3, "empty render"
    print(f"smoke OK: {len(fr)} frames, {os.path.getsize(p)} bytes -> {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objs", nargs="+", default=["011_banana", "050_medium_clamp",
                    "Razer_Taipan_Black_Ambidextrous_Gaming_Mouse"])
    ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    if a.smoke: smoke(); return
    for o in a.objs: render_obj(o)


if __name__ == "__main__":
    main()
