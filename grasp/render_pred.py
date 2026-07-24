"""Predicted-vs-ground-truth rollout clips (one per outcome). Loads the short-window rollout WM (roll_full.pt), samples
N predicted K-step rollouts for a representative TEST episode per outcome (rigid / transient-slip / drop), and renders the
GT object (cloud + blue oriented bbox) vs the model's sampled bboxes (orange) in the gripper frame (motion relative to
lift-onset; gripper fixed). Shows the model's predicted DISTRIBUTION over the near-term future vs what actually happened.
    python render_pred.py   ->  figures/rollout_pred_{rigid,slip,drop}.mp4
"""
import os, sys, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from grasp.my_dataset_trajfull import TrajFullDS
from grasp.dit_rollout import DiTRollout
from common.dit import cosine_acp, ddim_sample
from common.utils import sixd_to_R, geodesic_deg
from common.paths import OUTCOMES as OV
os.makedirs("figures", exist_ok=True)
BOX_E = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
N = 8


def load_cloud(obj, n=1500, seed=0):
    mu = np.load(f"{OV}/point_clouds/{obj}.npz")["mu"].astype(float)
    return mu[np.random.default_rng(seed).choice(len(mu), min(n, len(mu)), replace=False)]


def corners(lo, hi):
    x0,y0,z0 = lo; x1,y1,z1 = hi
    return np.array([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]])


def faces(c): return [[c[i] for i in f] for f in [(0,1,2,3),(4,5,6,7),(0,1,5,4),(2,3,7,6),(1,2,6,5),(0,3,7,4)]]


def gripper(xhw, yhw, zc, zhw):
    t = 0.007; ye = min(yhw*0.8, 0.016); fl = min(zhw*0.9, 0.028); zb, zt = zc-fl, zc+fl*0.4
    return (corners([-xhw-t,-ye,zb],[-xhw,ye,zt]), corners([xhw,-ye,zb],[xhw+t,ye,zt]), corners([-xhw-t,-ye,zt],[xhw+t,ye,zt+0.010]))


def xf(R, p, V): return (R @ V.T).T + p                         # apply pose (R,p) to points V


def frame_img(mu_g, gt_box, pred_boxes, grip, title, lim, cmn, cmx):
    fig = plt.figure(figsize=(4.4, 4.7)); ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(Poly3DCollection(sum([faces(j) for j in grip], []), facecolor="0.4", alpha=.22, edgecolor="0.2", lw=.4))
    for pb in pred_boxes:
        ax.add_collection3d(Line3DCollection([[pb[a], pb[b]] for a, b in BOX_E], colors="#D55E00", lw=.8, alpha=.45))
    ax.scatter(mu_g[:,0], mu_g[:,1], mu_g[:,2], s=2, c=mu_g[:,2], cmap="viridis", vmin=cmn, vmax=cmx, alpha=.4, linewidths=0)
    ax.add_collection3d(Line3DCollection([[gt_box[a], gt_box[b]] for a, b in BOX_E], colors="#0072B2", lw=1.9))
    ax.text2D(0.02, 0.985, "cloud + blue box = ground truth · orange = model samples · gray = gripper", transform=ax.transAxes, fontsize=5.5, va="top", color="0.3")
    ax.set_title(title, fontsize=9)
    for f, l in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], lim): f(l)
    ax.set_box_aspect((1,1,1)); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.view_init(18, -60)
    fig.tight_layout(); fig.canvas.draw()
    w, h = fig.canvas.get_width_height(); img = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(h, w, 4)[..., :3].copy()
    plt.close(fig); return img


def render(gt, xs, mu, title, out):
    Kf = len(gt); Rg = sixd_to_R(gt[:, :6]); pg = gt[:, 6:9]
    Rp = sixd_to_R(xs[..., :6]); pp = xs[..., 6:9]              # (N,K,3,3),(N,K,3)
    cc = corners(mu.min(0), mu.max(0))
    clouds = [xf(Rg[k], pg[k], mu) for k in range(Kf)]
    gtb = [xf(Rg[k], pg[k], cc) for k in range(Kf)]
    predb = [[xf(Rp[i, k], pp[i, k], cc) for i in range(xs.shape[0])] for k in range(Kf)]
    allp = np.concatenate([c for c in clouds] + [b for fb in predb for b in fb])
    ctr = allp.mean(0); rad = max(0.07, np.abs(allp - ctr).max() * 1.05); lim = [(ctr[i]-rad, ctr[i]+rad) for i in range(3)]
    c0 = clouds[0]; grip = gripper(np.abs(c0[:,0]).max()*1.02, np.abs(c0[:,1]).max(), c0[:,2].mean(), (c0[:,2].max()-c0[:,2].min())*0.5)
    cmn, cmx = c0[:,2].min(), c0[:,2].max()
    frames = [frame_img(clouds[k], gtb[k], predb[k], grip, f"{title}   step {k+1}/{Kf}", lim, cmn, cmx) for k in range(Kf)]
    import imageio; imageio.mimsave(f"{out}.mp4", frames, fps=4, codec="libx264", quality=8); print(f"wrote {out}.mp4")


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    P = dict(np.load("roll_runs/roll_full/test_pred.npz", allow_pickle=True)); Kf = int(P["K"]); Td = int(P["T"]); tmean = P["tmean"]; tstd = P["tstd"]
    m = DiTRollout(H=Kf).to(dev); m.load_state_dict(torch.load("roll_runs/roll_full/best.pt", map_location=dev)); m.eval()
    acp = cosine_acp(Td, device=dev)
    ds = TrajFullDS("test", modality="full", kind="short"); y5 = ds.labels()[0]
    shorts = np.stack([ds.T[ds.uid2t[r["uid"]]] for r in ds.rows])
    drift = geodesic_deg(sixd_to_R(shorts[:, -1, :6]), sixd_to_R(shorts[:, 0, :6]))
    with torch.no_grad():
        for mode, name, tag in [(0, "RIGID", "rigid"), (1, "TRANSIENT_SLIP", "slip"), (3, "LIFTED_DROPPED", "drop")]:
            sel = [j for j in range(len(ds)) if y5[j] == mode]; d = drift[sel]; pick = sel[int(np.argmin(np.abs(d - np.median(d))))]
            b = ds[pick]
            cond = m.encode(b["pts"][None].to(dev), b["base_rel"][None].to(dev), b["closing"][None].to(dev), b["table"][None].to(dev))
            xs = ddim_sample(m.dit, cond.repeat_interleave(N, 0), Kf, acp, steps=25, device=dev).cpu().numpy() * tstd + tmean
            render(b["target_raw"].numpy(), xs, load_cloud(b["object"]), f"{name}  ·  {b['object']}\nGT (blue) vs model samples (orange)", f"figures/rollout_pred_{tag}")


if __name__ == "__main__":
    main()
