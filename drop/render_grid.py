"""3-row x [ground truth | model imagination | remarks] grid of drop demos, rendered as DETAILED TEXTURED meshes.
Three held-out mesh-having objects; per row: left = MuJoCo physics settle (GT), middle = the rollout WM's imagined
settle (grounded arm) for the SAME release, right = a remark (model basin samples). Physics runs on the point-cloud
hull (what the WM sees); appearance is the full textured mesh.
    MUJOCO_GL=egl python render_grid.py
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, mujoco, imageio, torch
from scipy.spatial.transform import Rotation as R, Slerp
from PIL import Image, ImageDraw, ImageFont
HERE = os.path.dirname(os.path.abspath(__file__))
from drop.density.v2 import drop_sweep as DS
from drop.gateb.generate_obj import stable_orientations, basin_of
from drop.mesh_render_test import mesh_model, W, Hh
from drop.drop_diffusion import GateBDiT, ddim_sample, cosine_acp
from drop.drop_net import sixd_to_R
from drop.my_dataset_roll import RollDS, K
try:
    FONT = ImageFont.truetype("DejaVuSans.ttf", 24); FBIG = ImageFont.truetype("DejaVuSans.ttf", 32)
except Exception:
    FONT = FBIG = ImageFont.load_default()

OBJS = ["Schleich_Spinosaurus_Action_Figure", "Schleich_Hereford_Bull", "051_large_clamp"]
NICE = {"Schleich_Spinosaurus_Action_Figure": "Spinosaurus", "Schleich_Hereford_Bull": "Bull", "051_large_clamp": "Large clamp"}
ARM, NF, LM, N = "grounded_oracle", 17, "grounded", 60


def wxyz(q): return np.r_[q[3], q[:3]]


def mesh_cam(meshV):
    r = float(np.linalg.norm(meshV, axis=1).max()); cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, 0.3 * r]; cam.distance = max(0.4, 4.0 * r); cam.elevation = -16; cam.azimuth = 90
    return cam


def render_seq(mdl, rnd, cam, meshV, quats):                  # slerp quats over N frames; mesh lowest vertex on the floor
    Rk = R.from_quat([np.r_[q[1:], q[0]] for q in quats]); sl = Slerp(np.linspace(0, 1, len(quats)), Rk)
    d = mujoco.MjData(mdl); frames = []
    for t in np.linspace(0, 1, N):
        Rc = sl(t); pz = -Rc.apply(meshV)[:, 2].min()
        d.qpos[:3] = [0, 0, pz]; d.qpos[3:7] = wxyz(Rc.as_quat()); mujoco.mj_forward(mdl, d)
        rnd.update_scene(d, camera=cam); frames.append(rnd.render().copy())
    return frames


def settle_hull_quats(H, quat, drop_h, com):                  # real physics settle on the hull -> K keyframe quats (GT)
    ai, delta = int(com[0]), round(float(com[1]) / 0.003) * 0.003
    m = DS.build(H, delta, H["axes"][ai]); d = mujoco.MjData(m)
    pz = drop_h - R.from_quat(np.r_[quat[1:], quat[0]]).apply(H["V"])[:, 2].min()
    mujoco.mj_resetData(m, d); d.qpos[:3] = [0, 0, pz]; d.qpos[3:7] = quat; d.qvel[:] = 0; mujoco.mj_forward(m, d)
    qs = [quat.copy()]; still = 0
    for _ in range(4000):
        mujoco.mj_step(m, d); qs.append(d.qpos[3:7].copy())
        if np.linalg.norm(d.qvel[:3]) < 0.006 and np.linalg.norm(d.qvel[3:6]) < 0.1047:
            still += 1
            if still >= 75: break
        else:
            still = 0
    idx = np.linspace(0, len(qs) - 1, K).astype(int); return [qs[i] for i in idx]


def band(frames, txt):
    out = []
    for f in frames:
        im = Image.fromarray(f); dr = ImageDraw.Draw(im); dr.rectangle([0, 0, W, 34], fill=(0, 0, 0))
        dr.text((10, 4), txt, fill=(255, 255, 255), font=FONT); out.append(np.asarray(im))
    return out


def remark(obj, basins):
    im = Image.new("RGB", (W // 2, Hh), (15, 18, 24)); dr = ImageDraw.Draw(im)
    dr.text((18, Hh // 2 - 70), NICE[obj], fill=(255, 220, 120), font=FBIG)
    dr.text((18, Hh // 2 - 10), f"model basin samples:", fill=(210, 215, 225), font=FONT)
    dr.text((18, Hh // 2 + 22), f"  {basins}", fill=(255, 255, 255), font=FONT)
    dr.text((18, Hh // 2 + 66), "same release,", fill=(150, 165, 190), font=FONT)
    dr.text((18, Hh // 2 + 96), "diverging futures", fill=(150, 165, 190), font=FONT)
    return np.asarray(im)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"; os.environ["CDWM_GATEB_LATENT"] = LM
    te = RollDS("test"); mean = torch.tensor(te.stats["mean"], device=dev); std = torch.tensor(te.stats["std"], device=dev)
    model = GateBDiT(n_feat=NF, use_latent=False, H=K).to(dev)
    model.load_state_dict(torch.load(f"roll_runs/{ARM}/best.pt", map_location=dev)); model.eval()
    acp = cosine_acp(1000, device=dev); rows = []
    for obj in OBJS:
        H = DS.hull(obj); Rs, _ = stable_orientations(H)
        mdl_mesh, _sc, _mt, _off = mesh_model(obj); meshV = (np.asarray(_mt.vertices) - _mt.vertices.mean(0)) * _sc
        rnd = mujoco.Renderer(mdl_mesh, height=Hh, width=W); cam = mesh_cam(meshV)
        idxs = np.where(te.obj == obj)[0]; chosen = None
        with torch.no_grad():
            for j in idxs[:40]:
                j = int(j); b = te[j]
                cond = model.cond(*[b[k][None].to(dev) for k in ("pts", "base_rel", "closing", "table")], None)
                x0 = ddim_sample(model, cond.repeat_interleave(3, 0), K, acp, steps=25, device=dev)
                dR = sixd_to_R((x0 * std + mean)[..., :6]).cpu().numpy(); Rrel = te.R_rel[j]
                wq = [[wxyz(R.from_matrix(dR[s, k] @ Rrel).as_quat()) for k in range(K)] for s in range(3)]
                basins = [basin_of(wq[s][-1], Rs) for s in range(3)]; chosen = (j, wq, basins)
                if len(set(basins)) >= 2: break
        j, wq, basins = chosen
        gt = band(render_seq(mdl_mesh, rnd, cam, meshV,
                             settle_hull_quats(H, wxyz(R.from_matrix(te.R_rel[j]).as_quat()), float(te.dh[j]), te.com[j])),
                  f"{NICE[obj]}  —  ground truth (physics)")
        mo = band(render_seq(mdl_mesh, rnd, cam, meshV, wq[0]), f"{NICE[obj]}  —  model imagines")
        rem = remark(obj, basins)
        rows.append([np.concatenate([gt[i], mo[i], rem], axis=1) for i in range(N)])
        print(f"[{obj}] basins {basins} diverge={len(set(basins)) >= 2}")
    grid = [np.concatenate([rows[r][i] for r in range(len(rows))], axis=0) for i in range(N)] + \
           [np.concatenate([rows[r][-1] for r in range(len(rows))], axis=0)] * 10
    p = f"{HERE}/notes/demo_videos/drop_grid.mp4"; imageio.mimsave(p, grid, fps=20, macro_block_size=1)
    print(f"wrote {p} ({len(grid)} frames, {grid[0].shape})")


if __name__ == "__main__":
    main()
