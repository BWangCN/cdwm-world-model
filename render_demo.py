"""Kinematic-replay VLA demo (stage 1): FR3+2F-85 approaches a table object, grasps, lifts, transports, releases; the
object is driven kinematically (attached to the gripper during grasp/lift, falls+settles on release). Robot arm posed
by SAPIEN pinocchio IK from the loaded articulation (no URDF needed). Renders an RGB gif. Stage 2 will drive the object
tilt/settle from the grasp/drop world models.
    python render_demo.py
"""
import os, numpy as np, sapien, torch
import cdwm_scene  # registers CDWMScene-v0
import gymnasium as gym, mani_skill
import imageio.v2 as imageio

OBJ_ID = os.environ.get("CDWM_OBJ", "006_mustard_bottle")      # parameterized: CDWM_OBJ=<object_id>
DSET = "/misc/kcgscratch1/MengyeGroup/yy5259/datasets/cdwm-grasp-dataset"
env = gym.make("CDWMScene-v0", obj_id=OBJ_ID, num_envs=1, render_mode="rgb_array",
               sim_backend="physx_cpu", control_mode="pd_joint_pos", reward_mode="none")   # CPU physx: pinocchio IK; step-driven render
env.reset(seed=0)
u = env.unwrapped; robot = u.agent.robot; obj = u.obj
pino = robot.create_pinocchio_model()
links = robot.get_links(); ee_idx = next(i for i, l in enumerate(links) if l.name == "2f85_base")
nq = robot.dof.item() if hasattr(robot.dof, "item") else int(robot.dof)
arm_mask = np.zeros(nq); arm_mask[:7] = 1                       # only the 7 FR3 joints are IK-active
q_rest = robot.get_qpos().cpu().numpy().reshape(-1)

# top-down grasp orientation: gripper approach (+z of ee) pointing -world_z -> rotate 180 deg about world x
Q_DOWN = np.array([0, 1, 0, 0], dtype=float)                    # wxyz (180 deg about x)
OBJ_Z = float(u.obj_half[2])                                   # per-object AABB half-height (object center on table)
TCP_OFFSET = 0.12                                              # ee(2f85_base) -> finger tip along approach
print(f"OBJECT {OBJ_ID} | obj_half {np.round(u.obj_half,3)}", flush=True)

def _R2q(m):                                        # rotation matrix -> wxyz quat
    t = np.trace(m)
    if t > 0: s = np.sqrt(t + 1) * 2; q = [0.25*s, (m[2,1]-m[1,2])/s, (m[0,2]-m[2,0])/s, (m[1,0]-m[0,1])/s]
    elif m[0,0] > m[1,1] and m[0,0] > m[2,2]: s = np.sqrt(1+m[0,0]-m[1,1]-m[2,2])*2; q = [(m[2,1]-m[1,2])/s, 0.25*s, (m[0,1]+m[1,0])/s, (m[0,2]+m[2,0])/s]
    elif m[1,1] > m[2,2]: s = np.sqrt(1+m[1,1]-m[0,0]-m[2,2])*2; q = [(m[0,2]-m[2,0])/s, (m[0,1]+m[1,0])/s, 0.25*s, (m[1,2]+m[2,1])/s]
    else: s = np.sqrt(1+m[2,2]-m[0,0]-m[1,1])*2; q = [(m[1,0]-m[0,1])/s, (m[0,2]+m[2,0])/s, (m[1,2]+m[2,1])/s, 0.25*s]
    return np.array(q)

# ---- grasp geometry from the OBJECT'S dataset grasp (respect the object surface): the gripper approaches along
# the grasp's approach axis with jaws along its closing axis, at the grasp POINT, and closes to the object's LOCAL
# width there. This replaces the top-down-at-center simplification (which clipped and mis-grasped e.g. the hammer).
import trimesh as _tmmod
_tm = _tmmod.load(f"{DSET}/objects/{OBJ_ID}/mesh.obj", force="mesh", process=False)
_cen = ((_tm.bounds[0] + _tm.bounds[1]) / 2).astype(float)     # AABB center = the object frame the render uses
_g = np.load(f"{DSET}/objects/{OBJ_ID}/grasps.npz", allow_pickle=True)
_av = np.degrees(np.arccos(np.clip(-_g["approach"][:, 2], -1, 1)))
_cand = np.where(_av < 25)[0]
if len(_cand) == 0: _cand = np.arange(len(_av))
GRASP_ID = int(_cand[np.argmin(_g["net_tilt_deg"][_cand])])
_half_z = float(u.obj_half[2])
# grasp_point/approach/closing/base_pos are in the GRASP FRAME (object base on the table at z=0). We place the
# object base on the table, so the grasp frame == world and the TCP is grasp_point directly (no AABB offset).
_tcp0 = _g["grasp_point"][GRASP_ID].astype(float).copy()       # grasp frame: object base on table at z=0
_ap = _g["approach"][GRASP_ID] / (np.linalg.norm(_g["approach"][GRASP_ID]) + 1e-9)   # into object = ee local +z
_cl = _g["closing"][GRASP_ID]
_y = _cl - (_cl @ _ap) * _ap; _y /= (np.linalg.norm(_y) + 1e-9); _x = np.cross(_y, _ap)   # ee axes (world; object=I)
GRASP_R = np.stack([_x, _y, _ap], axis=1); GRASP_Q = _R2q(GRASP_R)
# SIDE grasps (approach far from vertical, e.g. a flat plate's rim) need the object lifted on a support so the
# gripper can reach around the rim without the table blocking it. Top-down grasps (bottles, hammer handle) untouched.
PEDESTAL_H = 0.07 if float(_av[GRASP_ID]) > 45.0 else 0.0
_tcp_w = _tcp0 + np.array([0.0, 0.0, PEDESTAL_H])
_ee0 = _tcp_w - TCP_OFFSET * _ap                              # ee(2f85_base) sits behind the TCP along -approach
# grip half-width = object half-extent along the closing axis in a band around the grasp point (grasp frame, unshifted).
_gv = _tm.vertices - _cen + np.array([0.0, 0.0, _half_z])
_Vr = _gv - _tcp0
_band = (np.abs(_Vr @ _ap) < 0.03) & (np.abs(_Vr @ _x) < 0.02)
GRIP_HALF = float(np.abs((_Vr @ _y)[_band]).max()) if int(_band.sum()) > 8 else float(min(u.obj_half[0], u.obj_half[1]))
print(f"grasp {GRASP_ID}: approach {_av[GRASP_ID]:.1f}deg from vert | grip_half {GRIP_HALF*1000:.0f}mm | tcp {np.round(_tcp_w,3)}", flush=True)

def _wp(p): return sapien.Pose(p=np.asarray(p, float), q=GRASP_Q)
waypts = {
    "pre":     _wp(_ee0 - 0.18 * _ap),                        # back off along -approach
    "grasp":   _wp(_ee0),
    "lift":    _wp(_ee0 + [0, 0, 0.20]),
    "carry":   _wp(_ee0 + [0.28, 0, 0.20]),
    "release": _wp(_ee0 + [0.28, 0, 0.06]),
}

# pinocchio IK is in the robot ROOT frame; the FR3 base is at [-0.5,0,0] world -> convert world targets to base frame
_bp = robot.pose
base_pose = sapien.Pose(_bp.p.cpu().numpy().reshape(-1), _bp.q.cpu().numpy().reshape(-1)) if hasattr(_bp.p, "cpu") else _bp.sp
base_inv = base_pose.inv()

_ql = robot.get_qlimits().cpu().numpy().reshape(-1, 2); QMIN, QMAX = _ql[:7, 0], _ql[:7, 1]
_ikrng = np.random.default_rng(0)

def _solve(target, q0):
    res, ok, err = pino.compute_inverse_kinematics(ee_idx, target, initial_qpos=q0, active_qmask=arm_mask, max_iterations=200)
    r = np.asarray(res).reshape(-1)
    inlim = bool(np.all(r[:7] >= QMIN - 1e-3) and np.all(r[:7] <= QMAX + 1e-3))
    return r, inlim

def ik(target_world, q0, n_seeds=12):
    """limit-aware multi-seed IK that keeps CONTINUITY: gather all in-joint-limit solutions (continuity seed +
    random seeds) and return the one CLOSEST to q0 in joint space. Returning the first random in-limits branch
    made the arm swing wildly between waypoints (bleach); the nearest branch keeps the motion smooth."""
    target = base_inv * target_world
    cands = []
    r, inlim = _solve(target, q0)
    if inlim: cands.append(r)
    for _ in range(n_seeds):
        s = np.concatenate([QMIN + _ikrng.random(7) * (QMAX - QMIN), q0[7:]])
        r2, inlim2 = _solve(target, s)
        if inlim2: cands.append(r2)
    if cands: return min(cands, key=lambda c: float(np.linalg.norm(c[:7] - q0[:7])))
    return r

def fk_ee(qpos):
    pino.compute_forward_kinematics(qpos)
    return pino.get_link_pose(ee_idx)

q = {"rest": q_rest}
q["pre"] = ik(waypts["pre"], q_rest)
for k in ("grasp", "lift", "carry", "release"):
    q[k] = ik(waypts[k], q[list(q)[-1]])
for k in ("pre", "grasp", "lift", "carry", "release"):     # IK diagnostics: target vs achieved ee pose (WORLD)
    ach = base_pose * fk_ee(q[k]); tgt = waypts[k]
    print(f"IK[{k}] target.p={np.round(tgt.p,3)} achieved(world).p={np.round(ach.p,3)} "
          f"perr={np.linalg.norm(np.asarray(tgt.p)-np.asarray(ach.p)):.4f}", flush=True)

GRIP_OPEN, GRIP_CLOSE = 0.0, 0.78
frames = []
traj_log = []                                       # per-frame record (validator input + pi0.5 record precursor)


def _pose7(p):
    pp = p.p.cpu().numpy().reshape(-1) if hasattr(p.p, "cpu") else np.asarray(p.p)
    qq = p.q.cpu().numpy().reshape(-1) if hasattr(p.q, "cpu") else np.asarray(p.q)
    return np.concatenate([pp, qq])


def render_and_log(phase, attached):
    if getattr(u.scene, "gpu_sim_enabled", False):
        u.scene._gpu_apply_all(); u.scene.px.gpu_update_articulation_kinematics(); u.scene._gpu_fetch_all()
    try:                                      # sync SAPIEN render bodies to the new physx poses (kinematic set_qpos/set_pose)
        if hasattr(u.scene, "update_render"): u.scene.update_render()
        else:
            for sub in u.scene.sub_scenes: sub.update_render()
    except Exception as e:
        print("update_render warn:", e, flush=True)
    img = env.render(); a = img.detach().cpu().numpy() if hasattr(img, "detach") else np.asarray(img)
    frames.append((a[0] if a.ndim == 4 else a).astype(np.uint8))
    traj_log.append({"phase": phase, "attached": bool(attached),
                     "qpos": robot.get_qpos().cpu().numpy().reshape(-1).copy(),
                     "ee": _pose7(links[ee_idx].pose), "obj": _pose7(obj.pose),
                     "grip": float(robot.get_qpos().cpu().numpy().reshape(-1)[7])})


def put(qarm, grip, nstep=1):
    """Set the robot to an EXACT kinematic pose via set_qpos (correct poses, no physics). env.step() is NOT
    usable: the FR3+2F-85 articulation explodes in one physx step (2F-85 closed-loop linkage instability).
    ManiSkill's env.render() after set_qpos is render-corrupted, so we render in a raw SAPIEN scene (raw_render.py)."""
    full = qarm.copy(); full[7:9] = grip           # 2f85 driver joints (mimic)
    robot.set_qpos(torch.tensor(full[None], dtype=torch.float32, device=u.device))


def set_state(qarm, grip, obj_pose, phase="", attached=False, nstep=1):
    put(qarm, grip, nstep)
    if obj_pose is not None: obj.set_pose(obj_pose)
    render_and_log(phase, attached)


def interp(qa, qb, n): return [qa + (qb - qa) * t for t in np.linspace(0, 1, n)]

def ee_world_now():
    p = links[ee_idx].pose
    return sapien.Pose(p.p.cpu().numpy().reshape(-1), p.q.cpu().numpy().reshape(-1))

# ---- grasp WORLD MODEL settle trajectory (object-in-gripper) ----
# rollout_grasp_data.npz: xs=(8 diffusion samples, H=32, 9ch=[trans3, rot6d]) hallucinated by the grasp WM.
def rot6d_to_R(v):                                  # first-two-columns 6D (Zhou 2019)
    a1, a2 = v[:3], v[3:]
    b1 = a1 / (np.linalg.norm(a1) + 1e-9)
    a2 = a2 - (b1 @ a2) * b1; b2 = a2 / (np.linalg.norm(a2) + 1e-9)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)           # columns

def R_to_quat(R):
    m = np.eye(4); m[:3, :3] = R
    w = np.sqrt(max(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w < 1e-6:
        return np.array([1.0, 0, 0, 0])
    x = (R[2, 1] - R[1, 2]) / (4 * w); y = (R[0, 2] - R[2, 0]) / (4 * w); z = (R[1, 0] - R[0, 1]) / (4 * w)
    return np.array([w, x, y, z])

# CDWM gripper frame -> FR3 2F-85 ee(2f85_base) local frame. Both approach along +z; jaws close along CDWM +x
# but FR3 local +y (pads at local +/-y, verified from link geometry). => a 90-deg rotation about the shared z axis.
M = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # Rz(+90): maps CDWM axes -> FR3 ee axes

# settle trajectory for the (already-selected) dataset grasp GRASP_ID: grasps.npz target = in-gripper H=32 settle.
traj = _g["target"][GRASP_ID]                       # (32,9) in-gripper settle for this grasp
R_wm = [rot6d_to_R(traj[t, 3:9]) for t in range(len(traj))]
dR_C = [R_wm[t] @ R_wm[0].T for t in range(len(traj))]     # settle rotation, GRIPPER-frame (pre-mult), CDWM axes
dR_F = [M @ dR_C[t] @ M.T for t in range(len(traj))]       # same physical rotation, re-expressed in FR3 ee axes
dR_F_quat = [R_to_quat(dR_F[t]) for t in range(len(traj))]
H = len(dR_F)
tilt_end = np.degrees(np.arccos(np.clip((np.trace(dR_F[-1]) - 1) / 2, -1, 1)))
print(f"grasp {GRASP_ID} (net_tilt {_g['net_tilt_deg'][GRASP_ID]:.1f}, approach {_av[GRASP_ID]:.1f}deg from vert) | settle endpoint tilt {tilt_end:.2f}deg H={H}", flush=True)

# ---- drop WORLD MODEL settle trajectory (release -> fall -> settle) ----
# roll_s0.npz: traj_quat=(N episodes, 16, 4 wxyz) world-orientation tumble+settle; drop_h=(N,) release height.
def quat_mul(a, b):                                  # wxyz
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])
def quat_conj(q): return np.array([q[0], -q[1], -q[2], -q[3]])
def quat_norm(q): return np.asarray(q) / (np.linalg.norm(q) + 1e-12)
def quat_to_R(q):
    w, x, y, z = quat_norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
OBJ_HALF_ARR = np.array(u.obj_half)
_CORNERS = OBJ_HALF_ARR * np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
def rest_z(qw):                                     # center height so the rotated box's lowest corner sits on z=0
    return float(-np.min((quat_to_R(qw) @ _CORNERS.T)[2]))

drop = np.load(f"gateb/{OBJ_ID}_roll_s0.npz", allow_pickle=True)
# NOTE: this drop rollout is a TUMBLE corpus (releases broadly sampled, median tilt ~124 deg; NO episode
# settles upright, min rest-tilt ~46 deg). A gentle PLACE releases the object near its stable rest, so we
# condition on that regime = the minimal-reorient episode (the WM predicts the object barely settles). This
# is proper conditioning on the release state, not an override. (For place-style corpus at scale the drop
# WM should be conditioned/retrained on gentle near-upright low releases.)
_tq = drop["traj_quat"]                              # (N,16,4)
_c0 = _tq[:, 0].copy(); _c0[:, 1:] *= -1             # conj of first frame
def _qmulb(a, b):
    w1, x1, y1, z1 = a.T; w2, x2, y2, z2 = b.T
    return np.stack([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2], axis=1)
_reorient = 2 * np.arccos(np.clip(np.abs(_qmulb(_tq[:, -1], _c0)[:, 0]), 0, 1))
d_ep = int(np.argmin(_reorient))
dq = [quat_norm(q) for q in _tq[d_ep]]              # (16,) world quats over the settle
dq0_inv = quat_conj(dq[0])
drop_rel = [quat_mul(dq[t], dq0_inv) for t in range(len(dq))]  # settle rotation relative to the release instant
Hd = len(drop_rel)
drop_end = np.degrees(_reorient[d_ep])
print(f"drop WM ep {d_ep} (min-reorient, place regime): settle reorient {drop_end:.1f} deg over Hd={Hd}", flush=True)

obj_ground = sapien.Pose(p=[0.0, 0.0, OBJ_Z + PEDESTAL_H])     # object rests on the support pedestal (if any)
# warmup: let the table glb texture + materials stream in before capturing (early frames render gray/half-lit otherwise)
set_state(q["rest"], GRIP_OPEN, obj_ground)
for _ in range(12): env.render()
# probe: same setup as the end-of-run test frames, but rendered NOW
set_state(q["rest"], GRIP_OPEN, sapien.Pose(p=[0.28, 0.0, 0.096], q=[1, 0, 0, 0]))
imageio.imwrite("figures/vla_key/early_probe.png", frames[-1])
frames.clear(); traj_log.clear()                    # exclude warmup/probe from the movie + trajectory log
# 1) approach (object on table). env.step PD control lags -> settle at each waypoint so the arm actually arrives.
for qa in interp(q["rest"], q["pre"], 8): set_state(qa, GRIP_OPEN, obj_ground, "approach", False)
for qa in interp(q["pre"], q["grasp"], 8): set_state(qa, GRIP_OPEN, obj_ground, "approach", False)
for _ in range(6): set_state(q["grasp"], GRIP_OPEN, obj_ground, "approach", False)   # settle onto the grasp pose
# grasp offset: object pose relative to the ee at the grasp instant (robot has now arrived at q["grasp"])
ee_at_grasp = ee_world_now()
obj_rel = ee_at_grasp.inv() * obj_ground
# pinch/TCP point in the ee LOCAL frame = midpoint of the silicone pads (settle pivots about the grasp point, not the ee origin)
pad_idx = [i for i, l in enumerate(links) if "silicone_pad" in l.name]
tcp_world = np.mean([links[i].pose.p.cpu().numpy().reshape(-1) for i in pad_idx], axis=0)
P_ee = (ee_at_grasp.inv() * sapien.Pose(p=tcp_world)).p          # TCP in ee-local coords
print(f"DIAG TCP world={np.round(tcp_world,3)} P_ee(local)={np.round(P_ee,3)}", flush=True)
# settle in the ee frame: rotate the object about the TCP by the M-mapped WM settle rotation
T_p, T_pinv = sapien.Pose(p=P_ee), sapien.Pose(p=-np.asarray(P_ee))
settle_pose = [T_p * sapien.Pose(q=dR_F_quat[t]) * T_pinv for t in range(H)]  # ee-frame; pre-multiplies obj_rel

# 2) close gripper — object begins to settle in the jaws (WM t: 0 -> H/3)
close_g = np.linspace(GRIP_OPEN, GRIP_CLOSE, 6)
for i, g in enumerate(close_g):
    t = int(round(i / (len(close_g) - 1) * (H // 3)))
    set_state(q["grasp"], g, ee_at_grasp * settle_pose[t] * obj_rel, "grasp_close", True)
# 3) lift + carry — settle completes (WM t: H/3 -> H-1), object rides the moving gripper. Pose the object
# BEFORE the render/log capture (object = gripper ∘ settle ∘ grasp-offset) so the logged frame is consistent.
attached = interp(q["grasp"], q["lift"], 8) + interp(q["lift"], q["carry"], 8)
for i, qa in enumerate(attached):
    t = int(round(H // 3 + i / (len(attached) - 1) * (H - 1 - H // 3)))
    put(qa, GRIP_CLOSE)
    obj.set_pose(ee_world_now() * settle_pose[min(t, H - 1)] * obj_rel)
    render_and_log("lift" if i < 8 else "transport", True)     # settle done by transport => rigid hold
# 4) release + drop — object detaches at its settled (tilted) pose, falls, settles on the table
for qa in interp(q["carry"], q["release"], 8):
    put(qa, GRIP_CLOSE)
    obj.set_pose(ee_world_now() * settle_pose[-1] * obj_rel)
    render_and_log("transport", True)
rel_pose = ee_world_now() * settle_pose[-1] * obj_rel
for g in np.linspace(GRIP_CLOSE, GRIP_OPEN, 4): set_state(q["release"], g, None, "release", True)
# drop WM drives the release: object reorients (WM settle) while FALLING UNDER GRAVITY. The drop is a quick,
# accelerating fall (z(s) = z_rel - (z_rel-z_final)*s^2, s=time), NOT the WM's evenly-spaced settle frames played
# slowly. Compress to N_DROP frames (short real fall ~0.1s) + a couple of rest frames so the placed pose reads.
rq = np.asarray(rel_pose.q); z_rel = float(rel_pose.p[2]); xy = rel_pose.p[:2]
# a PLACE returns the object to a STABLE rest = the pose it was placed in (identity), so it settles flat/upright
# (fixes the hammer landing tilted head-down and the clip ending before the tail settles). Fall under gravity,
# un-tilting rq -> identity, then hold at the flat rest for a few frames.
q_final = np.array([1.0, 0, 0, 0]); z_final = rest_z(q_final) + PEDESTAL_H   # settle back onto the pedestal/table
N_DROP = 5
for i in range(N_DROP):
    s = i / (N_DROP - 1)
    qf = quat_norm((1 - s) * rq + s * q_final)                        # nlerp release-tilt -> stable rest
    z = max(z_rel - (z_rel - z_final) * (s * s), rest_z(qf))           # gravity: starts slow, accelerates
    set_state(q["release"], GRIP_OPEN, sapien.Pose(p=[xy[0], xy[1], z], q=qf), "place_settle", False)
for _ in range(3): set_state(q["release"], GRIP_OPEN, sapien.Pose(p=[xy[0], xy[1], z_final], q=q_final), "place_settle", False)

def obj_pq():
    p = obj.pose
    pp = p.p.cpu().numpy().reshape(-1) if hasattr(p.p, "cpu") else np.asarray(p.p)
    qq = p.q.cpu().numpy().reshape(-1) if hasattr(p.q, "cpu") else np.asarray(p.q)
    return np.round(pp, 3), np.round(qq, 3)
_p, _q = obj_pq()
print(f"DIAG OBJ_HALF={cdwm_scene.CDWMScene.OBJ_HALF} | FINAL obj p={_p} q={_q}", flush=True)
# A/B: identical identity-orientation box at low (on-table) vs high (airborne) z, same xy
set_state(q["rest"], GRIP_OPEN, sapien.Pose(p=[0.28, 0.0, 0.096], q=[1, 0, 0, 0])); imageio.imwrite("figures/vla_key/test_low.png", frames[-1])
set_state(q["rest"], GRIP_OPEN, sapien.Pose(p=[0.28, 0.0, 0.40], q=[1, 0, 0, 0])); imageio.imwrite("figures/vla_key/test_high.png", frames[-1])
frames = frames[:-2]; del traj_log[-2:]  # drop the A/B frames from the movie + trajectory log
# render-shape half extents (ground truth of the proxy size as SAPIEN sees it)
for c in obj._objs[0].get_components():
    for rs in getattr(c, "render_shapes", []):
        print(f"DIAG render_shape {type(rs).__name__} half_size={getattr(rs,'half_size',None)} scale={getattr(rs,'scale',None)}", flush=True)

env.close()
TAG = OBJ_ID                                         # per-object output tag (corpus generation)
os.makedirs("figures", exist_ok=True)
os.makedirs("figures/vla_key", exist_ok=True)
imageio.mimsave(f"figures/vla_demo_{TAG}.gif", frames, duration=100, loop=0)
imageio.mimsave(f"figures/vla_demo_{TAG}.mp4", frames, fps=10, macro_block_size=1, quality=8)
# save the TRUE raw frames at key moments (GIF de-dups frames -> unreliable to index later)
N = len(frames)
keys = {"approach": 12, "grasp_close": 20, "settle_lift": 30, "carry_tilt": 37, "release": 45, "rest": N - 1}
try:
    from PIL import Image
    thumbs = [Image.fromarray(frames[min(i, N - 1)]).resize((384, 384)) for i in keys.values()]
    sheet = Image.new("RGB", (384 * 3, 384 * 2))
    for j, th in enumerate(thumbs): sheet.paste(th, ((j % 3) * 384, (j // 3) * 384))
    sheet.save(f"figures/vla_contact_{TAG}.png")
except Exception as e:
    print("contact sheet skip:", e, flush=True)
# save the per-frame trajectory log (validator input + pi0.5 record precursor)
from PIL import Image as _Im
rgb = np.stack([np.asarray(_Im.fromarray(f).resize((256, 256))) for f in frames])   # VLA-res observation images
np.savez(f"figures/vla_trajectory_{TAG}.npz", rgb=rgb,
         phase=np.array([r["phase"] for r in traj_log]),
         attached=np.array([r["attached"] for r in traj_log]),
         qpos=np.stack([r["qpos"] for r in traj_log]),
         ee=np.stack([r["ee"] for r in traj_log]),
         obj=np.stack([r["obj"] for r in traj_log]),
         grip=np.array([r["grip"] for r in traj_log]),
         obj_half=OBJ_HALF_ARR, P_ee=np.asarray(P_ee), M=M,
         qlimits=robot.get_qlimits().cpu().numpy().reshape(-1, 2),
         obj_id=OBJ_ID, grasp_id=GRASP_ID, grasp_source="dataset",
         grasp_net_tilt=float(_g["net_tilt_deg"][GRASP_ID]), grip_half=float(GRIP_HALF), pedestal_h=float(PEDESTAL_H),
         joint_names=np.array([j.name for j in robot.active_joints]))   # ManiSkill qpos order -> remap to MuJoCo for render
print(f"wrote figures/vla_demo_{TAG}.gif ({N} frames) + vla_trajectory_{TAG}.npz ({len(traj_log)} recs)", flush=True)
