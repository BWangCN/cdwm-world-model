"""Render representative rollout clips (rigid / transient-slip / drop) from the outcomes_v2 shards in the WORLD frame:
the floating parallel-jaw mount lifts off a table and the object holds / tilts / tumbles with it. (The dataset has no
gripper mesh — the gripper is a pose-only "floating parallel-jaw mount" per the README — so the jaws are drawn from the
mount pose base_pos/base_quat, which MOVES over the lift.)
    python render_rollouts.py   ->  figures/rollout_{rigid,slip,drop}.mp4
Per frame: points = object 3DGS gaussians; blue box = object oriented bbox; RGB = object body axes; dark ⊓ = parallel-jaw
mount (lifts); faint square = table. Frame 1 = grasp/settle start.
"""
import os, sys, csv, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wm.common import quat_to_R, sixd_to_R, geodesic_deg, R_to_6d
HF = "/misc/kcgscratch1/MengyeGroup/yy5259/datasets/cdwm-grasp-dataset"; OV = f"{HF}/outcomes_v2"
PPZ = 0.1159; H = 40
os.makedirs("figures", exist_ok=True)
BOX_E = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]


def load_cloud(obj, n=2500, seed=0):
    mu = np.load(f"{OV}/point_clouds/{obj}.npz")["mu"].astype(float)
    return mu[np.random.default_rng(seed).choice(len(mu), min(n, len(mu)), replace=False)]


def corners(lo, hi):
    x0,y0,z0 = lo; x1,y1,z1 = hi
    return np.array([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]])


def faces(c):
    return [[c[i] for i in f] for f in [(0,1,2,3),(4,5,6,7),(0,1,5,4),(2,3,7,6),(1,2,6,5),(0,3,7,4)]]


def jaw_mount(xhw, yhw, zc, zhw):                              # thin finger rods + base bar in the MOUNT frame (clear ⊓ around ±x)
    t = 0.007; ye = min(yhw*0.8, 0.016); fl = min(zhw*0.9, 0.028)
    zb, zt = zc - fl, zc + fl*0.4
    return (corners([-xhw-t,-ye,zb],[-xhw,ye,zt]), corners([xhw,-ye,zb],[xhw+t,ye,zt]), corners([-xhw-t,-ye,zt],[xhw+t,ye,zt+0.010]))


def frame_img(cloud, obb, axc, axv, jaw, table, title, lim, cmn, cmx):
    fig = plt.figure(figsize=(4.3, 4.7)); ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(Poly3DCollection([table], facecolor="0.85", alpha=.20))                          # table
    ax.add_collection3d(Poly3DCollection(sum([faces(j) for j in jaw], []), facecolor="0.35", alpha=.32, edgecolor="0.15", lw=.5))   # gripper (thin ⊓)
    ax.scatter(cloud[:,0], cloud[:,1], cloud[:,2], s=3, c=cloud[:,2], cmap="viridis", vmin=cmn, vmax=cmx, alpha=.6, linewidths=0)
    ax.add_collection3d(Line3DCollection([[obb[a], obb[b]] for a, b in BOX_E], colors="#0072B2", lw=1.1))
    for v, col in zip(axv, ["#D55E00", "#009E73", "#0033AA"]): ax.plot(*zip(axc, v), color=col, lw=2.6)
    ax.text2D(0.02, 0.985, "points=object · box=bbox · RGB=axes · gray=parallel-jaw mount (lifts) · plane=table", transform=ax.transAxes, fontsize=5.6, va="top", color="0.3")
    ax.set_title(title, fontsize=9)
    for f, l in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], lim): f(l)
    ax.set_box_aspect((1,1,1)); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.view_init(24, -60)
    fig.tight_layout(); fig.canvas.draw()
    w, h = fig.canvas.get_width_height(); img = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(h, w, 4)[..., :3].copy()
    plt.close(fig); return img


def episode(r):                                               # world poses over the close->lift->outcome window, resampled to H
    zf = np.load(f"{OV}/{r['shard']}"); s = int(r["frame_start"]); e = s + int(r["n_frames"])
    ph = zf["phase"][s:e]; w = np.where(ph >= 2)[0]; w = w if len(w) >= 2 else np.arange(e - s)
    idx = np.clip(np.round(np.linspace(w[0], w[-1], H)).astype(int), 0, e - s - 1)
    return (zf["obj_pos"][s:e][idx].astype(float), quat_to_R(zf["obj_quat"][s:e][idx].astype(float)),
            zf["base_pos"][s:e][idx].astype(float), quat_to_R(zf["base_quat"][s:e][idx].astype(float)))


def render(r, mu, title, out):
    op, Ro, bp, Rb = episode(r)
    cloudw = [op[t] + mu @ Ro[t].T for t in range(H)]                                   # object gaussians in world
    cc = corners(mu.min(0), mu.max(0)); ctr = mu.mean(0); asz = (mu.max(0) - mu.min(0)).max() * 0.45
    obbw = [op[t] + cc @ Ro[t].T for t in range(H)]; axcw = [op[t] + ctr @ Ro[t].T for t in range(H)]
    axvw = [[axcw[t] + Ro[t][:, k] * asz for k in range(3)] for t in range(H)]
    mm0 = (op[0] + mu @ Ro[0].T - bp[0]) @ Rb[0]                                         # object in mount frame at t0 -> jaw size
    jm = jaw_mount(np.abs(mm0[:,0]).max()*1.03, np.abs(mm0[:,1]).max(), mm0[:,2].mean(), (mm0[:,2].max()-mm0[:,2].min())*0.5)
    jaww = [[bp[t] + jc @ Rb[t].T for jc in jm] for t in range(H)]                       # jaws follow the moving mount
    drift = geodesic_deg(sixd_to_R(R_to_6d(Ro[-1])[None])[0], Ro[0])
    allp = np.concatenate(cloudw + [bp]); m = allp.mean(0); rad = max(0.1, np.abs(allp - m).max())
    lim = [(m[i]-rad, m[i]+rad) for i in range(3)]; ztab = min(op[0,2], cloudw[0][:,2].min())
    tb = [[m[0]-rad, m[1]-rad, ztab], [m[0]+rad, m[1]-rad, ztab], [m[0]+rad, m[1]+rad, ztab], [m[0]-rad, m[1]+rad, ztab]]
    cmn, cmx = cloudw[0][:,2].min(), cloudw[0][:,2].max()
    frames = [frame_img(cloudw[t], obbw[t], axcw[t], axvw[t], jaww[t], tb, f"{title}\n(end-drift {drift:.0f}°)   frame {t+1}/{H}", lim, cmn, cmx) for t in range(H)]
    import imageio; imageio.mimsave(f"{out}.mp4", frames, fps=10, codec="libx264", quality=8); print(f"wrote {out}.mp4")


def main():
    z = np.load(f"{OV}/traj_full.npz", allow_pickle=True); T = z["traj"]; uid = z["uid"]; u2i = {u: i for i, u in enumerate(uid)}
    rows = list(csv.DictReader(open(f"{OV}/outcomes_index.csv"))); byuid = {r["uid"]: r for r in rows}
    R0 = sixd_to_R(T[:, 0, :6]); RL = sixd_to_R(T[:, -1, :6]); drift = geodesic_deg(RL, R0)
    for mode, tag in [("RIGID", "rigid"), ("TRANSIENT_SLIP", "slip"), ("LIFTED_DROPPED", "drop")]:
        idx = [u2i[r["uid"]] for r in rows if r["v2_outcome"] == mode and r["uid"] in u2i]
        d = drift[idx]; pick = idx[int(np.argmin(np.abs(d - np.median(d))))]; r = byuid[uid[pick]]
        render(r, load_cloud(r["object_id"]), f"{mode}  ·  {r['object_id']}", f"figures/rollout_{tag}")


if __name__ == "__main__":
    main()
