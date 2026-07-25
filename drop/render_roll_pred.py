"""Render the drop rollout WM's IMAGINED settling vs MuJoCo ground truth. For a held-out object + a near-boundary
release: (top) the real physics re-sim; (then) several samples from the trained GateBDiT(H=K) rollout, each rendered
as the object descending + tumbling per the MODEL's predicted orientation trajectory to its predicted rest, labeled
with the resulting basin. Boundary releases -> the samples diverge into different basins = the model imagining
multiple settling futures. This is MODEL OUTPUT (unlike the physics-counterfactual demos).

    MUJOCO_GL=egl python render_roll_pred.py [--obj <name>]
"""
import os, sys, argparse, glob
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, mujoco, imageio, torch
from scipy.spatial.transform import Rotation as R, Slerp
HERE = os.path.dirname(os.path.abspath(__file__))
from drop.density.v2 import drop_sweep as DS
from drop.gateb.generate_obj import stable_orientations, basin_of
from drop.render_drop_demo import build_render, camera, W, H_, label, settle_frames, OUT
from drop.drop_diffusion import GateBDiT, ddim_sample, cosine_acp
from drop.drop_net import sixd_to_R
from drop.my_dataset_roll import RollDS, K

ARM, NF, USE_LAT, LM, NS = "grounded_oracle", 17, False, "grounded", 3


def wxyz(q_xyzw): return np.r_[q_xyzw[3], q_xyzw[:3]]
def rel_quat(Rrel): return wxyz(R.from_matrix(Rrel).as_quat())


def model_frames(H, wq, drop_h, rnd, cam, nf=48):             # object descends (height drop_h->0) + slerps through the K predicted orientations
    V = H["V"]; Rk = R.from_quat([np.r_[q[1:], q[0]] for q in wq]); sl = Slerp(np.linspace(0, 1, len(wq)), Rk)
    frames = []
    for t in np.linspace(0, 1, nf):
        # near-boundary drops are only cm above the table: land fast, then let the tumble + settle play out ON the floor
        Rc = sl(t); h = drop_h * max(0.0, 1 - 6 * t); pz = h - Rc.apply(V)[:, 2].min()
        m = build_render(H, 0.0, H["axes"][0]); d = mujoco.MjData(m)
        d.qpos[:3] = [0, 0, max(pz, -0.001)]; d.qpos[3:7] = wxyz(Rc.as_quat())
        mujoco.mj_forward(m, d); rnd.update_scene(d, camera=cam); frames.append(rnd.render().copy())
    return frames + [frames[-1]] * 12


def render_obj(obj, te, model, mean, std, acp, dev):          # GT + NS model rollouts for one HELD-OUT object -> mp4
    idxs = np.where(te.obj == obj)[0]
    H = DS.hull(obj); Rs, _ = stable_orientations(H)
    rnd = mujoco.Renderer(build_render(H, 0, H["axes"][0]), height=H_, width=W); cam = camera(H)
    last = None                                               # prefer a release where the NS samples diverge into >=2 basins
    with torch.no_grad():
        for j in idxs[:40]:
            j = int(j); b = te[j]
            cond = model.cond(*[b[k][None].to(dev) for k in ("pts", "base_rel", "closing", "table")],
                              b["latent"][None].to(dev) if USE_LAT else None)
            x0 = ddim_sample(model, cond.repeat_interleave(NS, 0), K, acp, steps=25, device=dev)
            dR = sixd_to_R((x0 * std + mean)[..., :6]).cpu().numpy()
            Rrel = te.R_rel[j]
            wq = [[wxyz(R.from_matrix(dR[s, k] @ Rrel).as_quat()) for k in range(K)] for s in range(NS)]
            basins = [basin_of(wq[s][-1], Rs) for s in range(NS)]
            last = (j, wq, basins)
            if len(set(basins)) >= 2: break
    j, wq, basins = last
    m = DS.build(H, round(float(te.com[j][1]) / 0.003) * 0.003, H["axes"][int(te.com[j][0])]); d = mujoco.MjData(m)
    gt, _ = settle_frames(m, d, rel_quat(te.R_rel[j]), H["V"], float(te.dh[j]), rnd, cam)
    clips = label(gt, f"{obj}   GROUND TRUTH (MuJoCo physics)")
    for s in range(NS):
        clips += label(model_frames(H, wq[s], float(te.dh[j]), rnd, cam),
                       f"{obj}   MODEL rollout sample {s+1}  ->  basin {basins[s]}")
    p = os.path.join(OUT, f"{obj}_roll_pred.mp4"); imageio.mimsave(p, clips, fps=30, macro_block_size=1)
    print(f"[{obj}] ep {j} basins {basins} diverge={len(set(basins)) >= 2} -> {p}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--objs", nargs="+", default=None); a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; os.environ["CDWM_GATEB_LATENT"] = LM
    te = RollDS("test"); mean = torch.tensor(te.stats["mean"], device=dev); std = torch.tensor(te.stats["std"], device=dev)
    model = GateBDiT(n_feat=NF, use_latent=USE_LAT, H=K).to(dev)
    model.load_state_dict(torch.load(f"roll_runs/{ARM}/best.pt", map_location=dev)); model.eval()
    acp = cosine_acp(1000, device=dev)
    objs = a.objs or sorted(set(te.obj), key=lambda o: -len(set(te.basin[te.obj == o])))[:3]   # top-3 multimodal held-out
    print("MODEL rollout (held-out test objects):", objs)
    for obj in objs:
        render_obj(obj, te, model, mean, std, acp, dev)


if __name__ == "__main__":
    main()
