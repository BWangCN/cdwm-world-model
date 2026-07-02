"""STAGE 2 (env: CD-WM venv, MUJOCO_GL=egl). Direction-annotated WM prediction video on NORMAL near-success grasps.
Per grasp: re-sim the Robotiq gripper, the object follows the GROUND-TRUTH settle, and TWO in-scene arrows are drawn from
the object along its up-axis: GREEN = GT (teacher) tilt direction, ORANGE = WM-predicted tilt direction. The arrows lean the
SAME amount (magnitude) but in different directions (axis) — that divergence IS the ~dir_err. H.264/yuv420p/faststart mp4."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import sys, subprocess, numpy as np
for p in ["/data/Manipulation/out/2026-06-25_focused-187-augment", "/data/Manipulation/out/2026-06-16_anygrasp-stability-distribution",
          "/data/CD-WM/grasp_closing_sweep", "/data/CD-WM/sanity_closing"]:
    sys.path.insert(0, p)
import mujoco, imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation
import config_aug as C, config_dist as DCFG, dist_harness as Hh, sweep_common as S
from build_grasps import base_pose_from_grasp
DCFG.MS_MODELS = C.MS_MODELS

OUT = "/data/Manipulation/out/2026-07-01_wm-direction-videos"
Z = np.load(f"{OUT}/direction_grasps.npz", allow_pickle=True)
Hn = int(Z["H"]); NA, NC, NH = Hh.NA, Hh.NC, Hh.NH; TOTAL = NA + NC + NH
PW, PH = 430, 340; DECOR = int(mujoco.mjtCatBit.mjCAT_DECOR)
GT_RGBA = [0.10, 0.85, 0.28, 1.0]; WM_RGBA = [1.0, 0.50, 0.10, 1.0]
GT_C, WM_C = (30, 200, 70), (255, 130, 26)
UP = np.array([0.0, 0.0, 1.0])


def q2R(q): return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
def R2q(R): x = Rotation.from_matrix(R).as_quat(); return np.array([x[3], x[0], x[1], x[2]])
def font(s):
    try: return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", s)
    except Exception: return ImageFont.load_default()


def add_arrow(scn, p0, p1, rgba, w=0.008):
    if scn.ngeom >= scn.maxgeom: return
    gm = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(gm, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).flatten(), np.asarray(rgba, np.float32))
    mujoco.mjv_connector(gm, mujoco.mjtGeom.mjGEOM_ARROW, w, np.asarray(p0, np.float64), np.asarray(p1, np.float64))
    gm.category = DECOR; scn.ngeom += 1


def resim_render(i):
    obj = str(Z["object"][i]); ep = dict(np.load(str(Z["episodes"][i]), allow_pickle=True))
    P = ep["P"].astype(float); ap = ep["approach"].astype(float)
    base_final = ep["base_pos"][0].astype(float); bq = ep["base_quat"][0].astype(float)
    op = ep["obj_pos"].astype(float); oqt = ep["obj_quat"].astype(float)
    Rgb = q2R(bq); up0 = q2R(oqt[0]) @ UP                       # world up at grasp-established
    PRg = Z["PRg"][i].astype(float)                             # [H,3,3] WM cumulative gripper-frame rotation
    mjcf = C.MANIFEST[obj]["mjcf"]; rest = Hh.obj_geom(obj)[0]
    g = dict(object=obj, pose_id="d", _speed="slow", grasp_center=P.tolist(), rest_pos=rest.tolist(), rest_quat=[1, 0, 0, 0],
             base_pos=base_final.tolist(), base_quat=bq.tolist(), offset_norm=0.0, offset_from_com=[0, 0, 0])
    sc = f"/tmp/dirvid/{obj}"; os.makedirs(sc, exist_ok=True)
    m = mujoco.MjModel.from_xml_path(Hh.build_scene_fixed(g, Hh.FORCE, sc, mjcf, table_contact=True)); data = mujoco.MjData(m)
    oq, obid, bq_, _rp, _lp, obj_geoms, _gg = Hh._ids(m, obj)
    for gi in range(m.ngeom):                                   # GHOST the gripper so it doesn't occlude the object
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[gi]) or ""
        if nm not in (obj, "tabletop", "world"):
            m.geom_rgba[gi] = [0.55, 0.57, 0.62, 0.22]; m.geom_matid[gi] = -1
    og_matid = m.geom_matid[obj_geoms].copy(); og_rgba = m.geom_rgba[obj_geoms].copy()   # object's real texture (for WM pass)
    GT_TINT = np.array([0.16, 0.80, 0.34, 1.0])                                           # flat green for the GT ghost pass
    act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")
    mocapid = m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gmocap")]
    slide0 = base_final - C.STANDOFF_M * ap
    mujoco.mj_resetData(m, data)
    data.qpos[oq:oq+3] = rest; data.qpos[oq+3:oq+7] = [1, 0, 0, 0]
    data.qpos[bq_:bq_+3] = slide0; data.qpos[bq_+3:bq_+7] = bq
    data.mocap_pos[mocapid] = slide0; data.mocap_quat[mocapid] = bq; data.ctrl[act] = 0.0; mujoco.mj_forward(m, data)
    sset = sorted({int(round(NA + k*(TOTAL-1-NA)/Hn)) for k in range(Hn+1)}); snaps = {}
    for step in range(TOTAL):
        if step < NA:      mo = slide0 + (base_final-slide0)*((step+1)/NA); ct = 0.0
        elif step < NA+NC: mo = base_final; ct = 255.0*(step-NA)/NC
        else:              mo = base_final; ct = 255.0
        data.mocap_pos[mocapid] = mo; data.ctrl[act] = ct; mujoco.mj_step(m, data)
        if step in sset: snaps[step] = data.qpos.copy()
    snap = [snaps[s] for s in sset]
    rnd = mujoco.Renderer(m, height=PH, width=PW, max_geom=3000)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = op[0] + np.array([0, 0, 0.03]); cam.distance = 0.40; cam.azimuth = 130; cam.elevation = -52
    R0 = q2R(oqt[0]); frames, leanG2d, leanW2d = [], [], []
    for t in range(Hn):
        qp = snap[min(t, len(snap)-1)].copy(); qp[oq:oq+3] = op[t]
        qwm = R2q(Rgb @ PRg[t] @ Rgb.T @ R0)                   # WM-PREDICTED object orientation (world)
        # pass 1 — WM prediction, real texture (this is what you watch)
        m.geom_matid[obj_geoms] = og_matid
        for j, gid in enumerate(obj_geoms): m.geom_rgba[gid] = og_rgba[j]
        qp[oq+3:oq+7] = qwm; data.qpos[:] = qp; mujoco.mj_forward(m, data)
        rnd.update_scene(data, camera=cam); imgW = rnd.render().astype(np.float32)
        # pass 2 — GT reference, flat GREEN ghost
        for gid in obj_geoms: m.geom_matid[gid] = -1; m.geom_rgba[gid] = GT_TINT
        qp[oq+3:oq+7] = oqt[t]; data.qpos[:] = qp; mujoco.mj_forward(m, data)
        rnd.update_scene(data, camera=cam); imgG = rnd.render().astype(np.float32)
        frames.append((0.66*imgW + 0.34*imgG).clip(0, 255).astype(np.uint8))   # WM solid over faint GT ghost
        Rggt = Rgb.T @ (q2R(oqt[t]) @ R0.T) @ Rgb              # GT settle rotation in GRIPPER frame
        leanG2d.append((Rggt @ UP)[:2]); leanW2d.append((PRg[t] @ UP)[:2])   # compass: gripper-frame fall dir (GT, WM)
    rnd.close(); del m, data, rnd
    return frames, np.array(leanG2d), np.array(leanW2d)


def compass(dr, cx, cy, R, g, w):
    """2D gripper-frame tilt compass: GT (green) vs WM (orange) fall-direction; angle between ≈ dir_err."""
    dr.ellipse([cx-R, cy-R, cx+R, cy+R], fill=(18, 18, 22), outline=(120, 120, 128), width=1)
    dr.line([cx-R+2, cy, cx+R-2, cy], fill=(55, 55, 60)); dr.line([cx, cy-R+2, cx, cy+R-2], fill=(55, 55, 60))
    def arr(v, col):
        n = float(np.hypot(v[0], v[1]))
        if n < 0.02: return
        L = R*max(0.5, min(n/0.5, 0.93)); ex, ey = cx+(v[0]/n)*L, cy-(v[1]/n)*L   # floor length -> low-tilt dir still readable
        dr.line([cx, cy, ex, ey], fill=col, width=4); ang = np.arctan2(-v[1], v[0])
        for da in (2.6, -2.6):
            dr.line([ex, ey, ex-11*np.cos(ang+da), ey-11*np.sin(ang+da)], fill=col, width=4)
    arr(w, WM_C); arr(g, GT_C)


def main():
    M = len(Z["episodes"]); out = [resim_render(i) for i in range(M)]
    tiles = [o[0] for o in out]; lg = [o[1] for o in out]; lw = [o[2] for o in out]
    ncol = 5; nrow = (M + ncol - 1) // ncol
    CAP = 34; LEG = 30
    Wc = ncol*PW; Hc = LEG + nrow*(PH + CAP)
    tmp = f"{OUT}/_frames"; os.makedirs(tmp, exist_ok=True)
    frames = list(range(Hn)) + [Hn-1]*8
    for fi, t in enumerate(frames):
        cv = Image.new("RGB", (Wc, Hc), (16, 16, 18)); dr = ImageDraw.Draw(cv)
        dr.text((8, 7), "WM PREDICTION on NORMAL near-success grasps  ·  textured object = WM PREDICTION   ·   faint GREEN ghost = GT (teacher) reference  ·  "
                "corner compass:  ORANGE = WM,  GREEN = GT tilt direction (gripper frame; angle between = dir_err)  ·  "
                f"settle {t+1:2d}/{Hn}", font=font(15), fill=(235, 235, 235))
        for i in range(M):
            r, c = divmod(i, ncol); x = c*PW; y = LEG + r*(PH+CAP)
            cv.paste(Image.fromarray(tiles[i][t]), (x, y+CAP))
            de = float(Z["dir_err"][i]); ec = (120, 220, 120) if de < 15 else ((255, 150, 40) if de < 28 else (240, 80, 80))
            dr.text((x+5, y+2), f"{str(Z['split'][i])[:4]} · {str(Z['shape'][i])[:9]} · {str(Z['band'][i])} · GTtilt{float(Z['net_tilt'][i]):.0f}",
                    font=font(12), fill=(200, 200, 200))
            dr.text((x+5, y+17), f"dir_err {de:.0f}°   mag_err {float(Z['mag_err'][i]):.0f}°   {str(Z['object'][i])[:16]}", font=font(12), fill=ec)
            compass(dr, x+50, y+CAP+PH-50, 40, lg[i][t], lw[i][t])
        cv.save(f"{tmp}/{fi:04d}.png")
    out = f"{OUT}/wm_direction_grid.mp4"
    subprocess.run(["ffmpeg", "-y", "-r", "8", "-i", f"{tmp}/%04d.png", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-crf", "20", out], check=True, capture_output=True)
    imageio.imwrite(f"{OUT}/wm_direction_lastframe.png", np.asarray(Image.open(f"{tmp}/{len(frames)-1:04d}.png")))
    import shutil; shutil.rmtree(tmp, ignore_errors=True)
    print("wrote", out, f"{os.path.getsize(out)//1024}KB ; + wm_direction_lastframe.png")


if __name__ == "__main__":
    main()
