"""Grasp-MTL rollout, MuJoCo replay (accurate + textured). Renders the real Robotiq 2F-85 closing on the real object
mesh, side by side: LEFT = ground truth (object posed by the GT settle trajectory, gripper at the GT achieved driver_rad),
RIGHT = model (object posed by the model's predicted trajectory, gripper at the model's PREDICTED driver_rad). Both use the
real gripper kinematics (the 4-bar linkage is solved by physics and cached, then set kinematically) and the real object,
so open/close and object pose are physically faithful. Picks a well-predicted rigid held-out episode.
    MUJOCO_GL=egl python render_grasp_mtl_roll.py [--smoke]     -> figures/rollout_grasp_mtl.{mp4,gif}
"""
import os, sys, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch, mujoco, imageio
from lxml import etree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from my_dataset_grasp_mtl import GraspMTLDS, GRIP_MAX
from wm.grasp_mtl import GraspMTL
from wm.dit import cosine_acp, ddim_sample
from wm.common import sixd_to_R, geodesic_deg, quat_to_R

MEN = "/home/yy5259/.cache/robot_descriptions/mujoco_menagerie/robotiq_2f85"
HFOBJ = "/misc/kcgscratch1/MengyeGroup/yy5259/datasets/cdwm-grasp-dataset/objects"
W, Hh = 640, 640
GJOINTS = ["right_driver_joint", "right_coupler_joint", "right_spring_link_joint", "right_follower_joint",
           "left_driver_joint", "left_coupler_joint", "left_spring_link_joint", "left_follower_joint"]
MOUNT_Z = 0.007                                                    # base_mount z offset in the menagerie model = data gripper-frame origin


def wxyz(Rm):
    from scipy.spatial.transform import Rotation as Rot
    q = Rot.from_matrix(Rm).as_quat()                             # xyzw
    return np.r_[q[3], q[:3]]


def build_scene_xml(obj, mesh_center, textured=True):
    """menagerie 2f85 + the object as a visual mocap body (kinematic). meshdir absolute so STLs resolve.
    The object geom is offset by -mesh_center so the mesh centroid sits at the body origin (-> anchor centroid at the pinch)."""
    root = etree.parse(f"{MEN}/2f85.xml").getroot()
    root.find("compiler").set("meshdir", f"{MEN}/assets")
    vis = etree.SubElement(root, "visual")
    etree.SubElement(vis, "headlight", ambient="0.5 0.5 0.5", diffuse="0.7 0.7 0.7", specular="0.1 0.1 0.1")
    etree.SubElement(vis, "global", offwidth=str(W), offheight=str(Hh))
    etree.SubElement(vis, "quality", shadowsize="4096")
    asset = root.find("asset")
    md = f"{HFOBJ}/{obj}"
    if textured:                                                 # real texture (mesh.obj carries vt uvs)
        etree.SubElement(asset, "texture", name="otx", type="2d", file=f"{md}/material_0.png")
        etree.SubElement(asset, "material", name="omat", texture="otx", specular="0.2", shininess="0.3")
    else:
        etree.SubElement(asset, "material", name="omat", rgba="0.85 0.60 0.25 1", specular="0.3", shininess="0.4")
    etree.SubElement(asset, "mesh", name="omesh", file=f"{md}/mesh.obj")
    wb = root.find("worldbody")
    etree.SubElement(wb, "light", pos="0.2 -0.3 0.6", dir="-0.3 0.4 -1", diffuse="0.5 0.5 0.5")
    ob = etree.SubElement(wb, "body", name="obj", mocap="true", pos="0 0 0.12")
    off = " ".join(f"{-x:.6f}" for x in mesh_center)
    etree.SubElement(ob, "geom", type="mesh", mesh="omesh", material="omat", pos=off, contype="0", conaffinity="0", group="0")
    return etree.tostring(root, pretty_print=True).decode()


def calibrate(dev="cpu"):
    """Sweep the actuator; for each settled driver_rad record the full 8-joint config -> interp table (physics solves the linkage)."""
    m = mujoco.MjModel.from_xml_path(f"{MEN}/2f85.xml"); d = mujoco.MjData(m)
    adr = [m.joint(j).qposadr[0] for j in GJOINTS]; drv = m.joint("right_driver_joint").qposadr[0]
    angs, qs = [], []
    for c in np.linspace(0, 255, 40):
        d.ctrl[:] = c
        for _ in range(400): mujoco.mj_step(m, d)
        angs.append(float(d.qpos[drv])); qs.append(d.qpos[adr].copy())
    angs, qs = np.array(angs), np.array(qs)                       # (40,), (40,8)
    return lambda a: np.array([np.interp(np.clip(a, angs.min(), angs.max()), angs, qs[:, i]) for i in range(8)])


def grasp_orientation(t0):
    """Object orientation in the gripper frame at settle t0 (translation-independent, so robust to the data's base-frame offset)."""
    R0, Rg = quat_to_R(t0[:4]), quat_to_R(t0[7:11])
    return Rg.T @ R0                                             # object -> gripper-frame rotation


def pick_episodes(model, ds, dev, k=3, pool=96, objects=None):
    """Top-k well-predicted rigid held-out episodes (low endpoint geo, visible grip swing), one per distinct object.
    If `objects` (list of names) is given, restrict to those objects (best-geo episode each), in that order."""
    rigid = np.nonzero(ds.is_rigid.astype(bool))[0]
    swing = ds.grip[rigid].std(1); cand = rigid[swing > 0.10]     # need an open->close swing to see
    if objects is not None:
        cand = np.array([i for i in cand if str(ds.obj[i]) in set(objects)])
    rng = np.random.default_rng(0); cand = rng.permutation(cand)[:(pool if objects is None else len(cand))]
    sd = torch.tensor(ds.stats["std"], device=dev); mu = torch.tensor(ds.stats["mean"], device=dev)
    acp = cosine_acp(1000, device=dev); scored = []
    with torch.no_grad():
        for j in cand:
            b = ds[int(j)]
            cond = model.encode(*[b[kk][None].to(dev) for kk in ("pts", "base_rel", "closing", "table")])
            x = ddim_sample(model, cond.repeat_interleave(8, 0), 32, acp, steps=25, device=dev)[:, -1] * sd + mu
            gt = (b["target"][-1].to(dev) * sd + mu)
            geo = float(np.median(geodesic_deg(sixd_to_R(x[:, 3:].cpu().numpy()),
                                               np.repeat(sixd_to_R(gt[3:].cpu().numpy())[None], 8, 0))))
            scored.append((int(j), str(ds.obj[int(j)]), geo))
    scored.sort(key=lambda t: t[2]); out, seen = [], set()          # best-first, one per object (with a real mesh)
    for j, o, g in scored:
        if o in seen or not os.path.isfile(f"{HFOBJ}/{o}/mesh.obj"): continue
        seen.add(o); out.append((j, o, g))
        if len(out) == k: break
    if objects is not None:                                          # keep requested order
        rank = {o: i for i, o in enumerate(objects)}; out.sort(key=lambda t: rank.get(t[1], 99))
    return out


def rollout(model, ds, j, dev):
    sd, mu = ds.stats["std"], ds.stats["mean"]
    b = ds[j]
    with torch.no_grad():
        cond = model.encode(*[b[k][None].to(dev) for k in ("pts", "base_rel", "closing", "table")])
        xs = ddim_sample(model, cond, 32, cosine_acp(1000, device=dev), steps=50, device=dev)[0].cpu().numpy() * sd + mu
        gp = model.predict_grip(cond).clamp(0, 1)[0].cpu().numpy() * GRIP_MAX     # predicted driver_rad (H,)
    gt = b["target"].numpy() * sd + mu
    gt_grip = b["grip_tgt"].numpy() * GRIP_MAX
    return b, gt, xs, gt_grip, gp


def body_pose(anchor, R_og, traj, k):
    """world (pos,quat) for the mocap object at step k. Centroid anchored at the pinch (anchor); orientation = grasp
    orientation R_og with the trajectory delta on top; position shifted by the (small) settle-translation delta."""
    Rd = sixd_to_R(traj[k, 3:]) @ sixd_to_R(traj[0, 3:]).T        # rotation delta from t0 (identity at k=0)
    td = traj[k, :3] - traj[0, :3]                                # settle translation delta (gripper frame ~ world)
    return anchor + td, Rd @ R_og


def render_seq(m, d, rnd, cam, qfn, anchor, R_og, traj, grip):
    adr = [m.joint(jn).qposadr[0] for jn in GJOINTS]
    frames = []
    for k in range(len(traj)):
        d.qpos[adr] = qfn(float(grip[k]))                        # gripper at driver_rad[k]
        pw, Rw = body_pose(anchor, R_og, traj, k)
        d.mocap_pos[0] = pw; d.mocap_quat[0] = wxyz(Rw)
        mujoco.mj_forward(m, d); rnd.update_scene(d, camera=cam)
        frames.append(np.rot90(rnd.render(), 2).copy())          # 180deg: gripper points DOWN (top-down grasp), no mirror (label reads upright)
    return frames


def compose(gtf, prf, obj, geo, half=True):
    """side-by-side GT|MODEL, crop dead (black) rows, optional 0.5 downscale, then draw labels (crisp after resize)."""
    from PIL import Image, ImageDraw
    comp = [np.concatenate([g, p], 1) for g, p in zip(gtf, prf)]
    stack = np.stack(comp).max(0)                                # brightest over time -> content rows
    rows = np.where(stack.max(2).max(1) > 18)[0]
    y0, y1 = max(0, rows.min() - 8), min(comp[0].shape[0], rows.max() + 8)
    out = []
    for f in comp:
        im = Image.fromarray(f[y0:y1])
        if half: im = im.resize((im.width // 2, im.height // 2), Image.LANCZOS)
        im = im.crop((0, 0, im.width // 2 * 2, im.height // 2 * 2))  # even dims (libx264 yuv420p requires even H/W)
        dr = ImageDraw.Draw(im); w = im.width
        short = obj.replace("_", " ")[:20].rstrip()             # keep it inside the left half (avoid running into the MODEL label)
        dr.text((6, 4), f"GT  {short}  {geo:.1f}deg", fill=(235, 235, 235))
        dr.text((w // 2 + 6, 4), "MODEL (predicted)", fill=(235, 235, 235))
        out.append(np.array(im))
    return out


def render_one(model, ds, j, obj, geo, qfn, dev, smoke=False):
    import trimesh
    b, gt, xs, gt_grip, gp = rollout(model, ds, j, dev)
    print(f"  {obj}: geo {geo:.2f} deg | grip GT {gt_grip[-1]:.3f} pred {gp[-1]:.3f} rad", flush=True)
    mesh = trimesh.load(f"{HFOBJ}/{obj}/mesh.obj", force="mesh"); mesh_center = np.asarray(mesh.centroid, float)
    R_og = grasp_orientation(ds.t0[j])
    try:
        m = mujoco.MjModel.from_xml_string(build_scene_xml(obj, mesh_center, textured=True))
    except Exception as e:
        print(f"  texture load failed ({e}); solid color", flush=True)
        m = mujoco.MjModel.from_xml_string(build_scene_xml(obj, mesh_center, textured=False))
    d = mujoco.MjData(m); rnd = mujoco.Renderer(m, height=Hh, width=W)
    mujoco.mj_forward(m, d); anchor = d.site("pinch").xpos.copy()   # grasp center = pinch site world pos
    cam = mujoco.MjvCamera(); cam.lookat[:] = anchor + np.array([0, 0, 0.02])
    cam.distance = 0.22; cam.elevation = -8; cam.azimuth = 130
    if smoke:
        fs = render_seq(m, d, rnd, cam, qfn, anchor, R_og, gt[[0, -1]], gt_grip[[0, -1]])
        imageio.imwrite("figures/_smoke_grasp_mtl.png", np.concatenate(fs, 1)); print("wrote figures/_smoke_grasp_mtl.png"); return None
    gtf = render_seq(m, d, rnd, cam, qfn, anchor, R_og, gt, gt_grip)
    prf = render_seq(m, d, rnd, cam, qfn, anchor, R_og, xs, gp)
    return compose(gtf, prf, obj, geo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--list", action="store_true"); ap.add_argument("--objects", default=None); a = ap.parse_args()
    objs = a.objects.replace("+", ",").split(",") if a.objects else None   # '+' delimiter too (sbatch --export splits on ',')
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0); np.random.seed(0)
    ck = torch.load("gated_runs/grasp_mtl/best.pt", map_location=dev)
    model = GraspMTL(n_feat=int(ck["n_feat"]), H=int(ck["H"]), D=int(ck["dim"]), depth=int(ck["depth"])).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    ds = GraspMTLDS("val", stats=ck["stats"]); os.makedirs("figures", exist_ok=True)
    if a.list:                                                   # print well-predicted candidate objects, no render
        for j, o, g in pick_episodes(model, ds, dev, k=20): print(f"  {g:5.2f} deg  {o}", flush=True)
        return
    qfn = calibrate()                                             # gripper linkage cache (object-independent)

    eps = pick_episodes(model, ds, dev, k=1 if a.smoke else a.n, objects=objs)
    print("episodes:", [(o, round(g, 2)) for _, o, g in eps], flush=True)
    for i, (j, obj, geo) in enumerate(eps, 1):
        try:
            frames = render_one(model, ds, j, obj, geo, qfn, dev, smoke=a.smoke)
        except Exception as e:
            print(f"  SKIP {obj}: {e}", flush=True); continue
        if frames is None: return
        tag = "" if i == 1 else f"_{i}"                          # primary keeps the base name (dev-log embed)
        imageio.mimsave(f"figures/rollout_grasp_mtl{tag}.mp4", frames, fps=8, codec="libx264", quality=8, macro_block_size=1)
        imageio.mimsave(f"figures/rollout_grasp_mtl{tag}.gif", frames[::2], duration=160, loop=0)
        print(f"wrote figures/rollout_grasp_mtl{tag}.{{mp4,gif}}", flush=True)


if __name__ == "__main__":
    main()
