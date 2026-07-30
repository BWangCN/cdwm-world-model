"""Photoreal render of the kinematic-replay episode in MuJoCo (the project's proven renderer; EGL offscreen).
MuJoCo does pure kinematic FK (mj_forward, no dynamics) so there's no articulation explosion, and its renderer
is reliable here (unlike SAPIEN headless). Loads the FR3+2F-85 MJCF, adds a table/floor/camera/lights + the object
as a mocap box, and plays the trajectory log (qpos remapped ManiSkill->MuJoCo by joint name; object pose from log).
    MUJOCO_GL=egl python render_mujoco.py <obj_id>
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import sys, numpy as np, mujoco
from PIL import Image

OBJ = sys.argv[1] if len(sys.argv) > 1 else "006_mustard_bottle"
d = np.load(f"figures/vla_trajectory_{OBJ}.npz", allow_pickle=True)
qpos = d["qpos"]; objp = d["obj"]; half = [float(x) for x in d["obj_half"]]
jnames = [str(x) for x in d["joint_names"]]; phase = d["phase"].astype(str); N = len(qpos)
W = H = 1024

spec = mujoco.MjSpec.from_file("assets/fr3_2f85.xml")
spec.visual.headlight.ambient = [0.5, 0.5, 0.5]; spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
spec.visual.global_.offwidth = W; spec.visual.global_.offheight = H
wb = spec.worldbody
# place the FR3 base at the ManiSkill world pose (base at x=-0.5) so the logged world object poses line up
base = next(b for b in wb.bodies if b.name == "base"); base.pos = [-0.5, 0.0, 0.0]
# lights
wb.add_light(pos=[0.4, -0.4, 1.6], dir=[-0.25, 0.25, -1], diffuse=[0.7, 0.7, 0.7], specular=[0.03, 0.03, 0.03])
# checker floor material + a wood-toned table top (top surface at z=0)
tex = spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                       width=300, height=300, rgb1=[0.28, 0.28, 0.32], rgb2=[0.36, 0.36, 0.4])
spec.add_material(name="gridmat", textures=["", "grid"], texrepeat=[8, 8], reflectance=0.1)
wb.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], pos=[0, 0, -0.041], material="gridmat")
tt = wb.add_body(name="table_top", pos=[0, 0, -0.02])
tt.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.7, 0.55, 0.02], rgba=[0.62, 0.42, 0.24, 1])
PED_H = float(d["pedestal_h"]) if "pedestal_h" in d.files else 0.0   # support that lifts a low-grasp object (e.g. plate)
if PED_H > 1e-4:
    pb = wb.add_body(name="pedestal", pos=[0, 0, PED_H / 2])
    pb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.05, PED_H / 2], rgba=[0.35, 0.35, 0.38, 1])
# object: the REAL scanned mesh + texture (not a box proxy). Center the mesh AABB at the mocap origin so it
# lines up with the WM object pose (which treats the object as centered at the actor origin).
DSET = "/misc/kcgscratch1/MengyeGroup/yy5259/datasets/cdwm-grasp-dataset"
mdir = f"{DSET}/objects/{OBJ}"
import trimesh
tm = trimesh.load(f"{mdir}/mesh.obj", force="mesh", process=False)
cen = ((tm.bounds[0] + tm.bounds[1]) / 2).astype(float)
spec.add_mesh(name="objmesh", file=f"{mdir}/mesh.obj")
matname = None
texf = f"{mdir}/material_0.png"
if os.path.exists(texf):
    spec.add_texture(name="objtex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=texf)
    mat = spec.add_material(name="objmat")
    try: mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "objtex"
    except Exception:
        try: mat.textures = ["objtex"]
        except Exception: mat = None
    if mat is not None:                                     # matte object: kill blown-out specular highlights
        try: mat.specular = 0.05; mat.shininess = 0.05; mat.reflectance = 0.0
        except Exception: pass
    matname = "objmat" if mat is not None else None
ob = wb.add_body(name="cdwm_obj", mocap=True, pos=[0, 0, half[2]])
g = ob.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="objmesh", pos=(-cen).tolist())
if matname: g.material = matname
else: g.rgba = [0.85, 0.78, 0.62, 1]
# camera matching the SAPIEN view: eye -> target
eye = np.array([0.95, -0.85, 0.5]); tgt = np.array([-0.1, 0.0, 0.12])
fwd = tgt - eye; fwd /= np.linalg.norm(fwd)
right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right); up = np.cross(right, fwd)
def mat2quat(m):                                    # camera-to-world rotation -> wxyz quat (camera looks along -z)
    t = np.trace(m)
    if t > 0: s = np.sqrt(t + 1) * 2; q = [0.25 * s, (m[2,1]-m[1,2])/s, (m[0,2]-m[2,0])/s, (m[1,0]-m[0,1])/s]
    elif m[0,0] > m[1,1] and m[0,0] > m[2,2]: s = np.sqrt(1+m[0,0]-m[1,1]-m[2,2])*2; q = [(m[2,1]-m[1,2])/s, 0.25*s, (m[0,1]+m[1,0])/s, (m[0,2]+m[2,0])/s]
    elif m[1,1] > m[2,2]: s = np.sqrt(1+m[1,1]-m[0,0]-m[2,2])*2; q = [(m[0,2]-m[2,0])/s, (m[0,1]+m[1,0])/s, 0.25*s, (m[1,2]+m[2,1])/s]
    else: s = np.sqrt(1+m[2,2]-m[0,0]-m[1,1])*2; q = [(m[1,0]-m[0,1])/s, (m[0,2]+m[2,0])/s, (m[1,2]+m[2,1])/s, 0.25*s]
    return np.array(q)
cam = wb.add_camera(name="vla", pos=eye.tolist(), fovy=57)
cam.quat = mat2quat(np.column_stack([right, up, -fwd])).tolist()

model = spec.compile(); data = mujoco.MjData(model)
# ManiSkill qpos order -> MuJoCo qpos address, by joint name
adr = []
for nm in jnames:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm)
    adr.append(model.jnt_qposadr[jid] if jid >= 0 else -1)
adr = np.array(adr)
print(f"joint remap: matched {int((adr>=0).sum())}/{len(jnames)}", flush=True)
mocap = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cdwm_obj")]

# The log's 2F-85 gripper qpos is an INCONSISTENT linkage pose (ManiSkill set only the drivers; the passive
# coupler/spring/follower stayed at 0) -> fingers clip the object. Replace it with a physically-consistent pose
# (drivers + passive coupled) calibrated so the pads close to just outside the object width.
GJNT = ["2f85_right_driver_joint", "2f85_right_coupler_joint", "2f85_right_spring_link_joint", "2f85_right_follower_joint",
        "2f85_left_driver_joint", "2f85_left_coupler_joint", "2f85_left_spring_link_joint", "2f85_left_follower_joint"]
gadr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in GJNT]
HAVE_GRIP = False
try:
    gact = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "2f85_fingers_actuator")
    pads = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in ["2f85_right_silicone_pad", "2f85_left_silicone_pad"]]
    jaw_half = float(d["grip_half"]) if "grip_half" in d.files else float(min(half[0], half[1]))   # object width at the grasp point
    cr = model.actuator_ctrlrange[gact]
    cal = mujoco.MjData(model); closed_q = None; best = (1e9, np.zeros(8)); widest = (-1.0, np.zeros(8))
    for c in np.linspace(cr[0], cr[1], 60):
        mujoco.mj_resetData(model, cal); cal.ctrl[gact] = c
        for _ in range(40): mujoco.mj_step(model, cal)
        sep = float(np.linalg.norm(cal.xpos[pads[0]] - cal.xpos[pads[1]]) / 2)
        q8 = cal.qpos[gadr].copy()
        if sep > widest[0]: widest = (sep, q8)               # most-open pose
        if sep >= jaw_half + 0.009 and sep - (jaw_half + 0.009) < best[0]: best = (sep - (jaw_half + 0.009), q8)   # ~1cm clearance -> pads stay outside the object surface
    closed_q = best[1]; open_q = widest[1]
    HAVE_GRIP = True
    print(f"gripper calib: jaw_half={jaw_half:.3f}m -> closed pose set (open={np.round(open_q,2)})", flush=True)
except Exception as e:
    print("gripper calib skipped:", repr(e), flush=True)
GRASP_PH = {"grasp_close", "lift", "transport", "release"}

rnd = mujoco.Renderer(model, height=H, width=W)
frames = []
for i in range(N):
    for k, a in enumerate(adr):
        if a >= 0: data.qpos[a] = qpos[i][k]
    if HAVE_GRIP:                                            # override the inconsistent gripper qpos
        gq = closed_q if phase[i] in GRASP_PH else open_q
        for k, a in enumerate(gadr): data.qpos[a] = gq[k]
    data.mocap_pos[mocap] = objp[i][:3]
    data.mocap_quat[mocap] = objp[i][3:]                # wxyz (both SAPIEN + MuJoCo)
    mujoco.mj_forward(model, data)
    rnd.update_scene(data, camera="vla")
    frames.append(rnd.render().copy())

GW = 256                                            # smaller gif frames + palette optimization -> much smaller files
ims = [Image.fromarray(f).resize((GW, GW)).convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames]
ims[0].save(f"figures/mj_{OBJ}.gif", save_all=True, append_images=ims[1:], duration=90, loop=0, optimize=True)
strip = Image.new("RGB", (240 * 8, 240))
for j, i in enumerate(range(0, N, max(1, N // 8))[:8]): strip.paste(ims[i].resize((240, 240)), (j * 240, 0))
strip.save(f"figures/mj_strip_{OBJ}.png")
print(f"wrote figures/mj_{OBJ}.gif ({N} frames) | mean {np.mean(frames):.1f}", flush=True)
