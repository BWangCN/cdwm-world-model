"""Corpus diffusion-rollout demo: the model IMAGINES the settle on the natural corpus (the regime upgrade of the
Gate-B-only rollout grid). Same 3 held-out episodes as the primary corpus demo. Per row:
  LEFT  = recorded corpus trajectory replayed (real free-fall/impact/settle, no re-simulation);
  MID x2 = two of S=8 diffusion samples: a DETERMINISTIC ballistic fall is prepended for context (render-only,
           labeled -- the model does NOT generate the fall; orientation is constant and height closed-form
           pre-contact), then the sampled contact->settle anchors slerped at floor level;
  RIGHT = remark (GT outcome + the outcome tally over all S samples; for the CoQ10 tip episode the samples
           DIVERGE -- some settle back, some tip -- the distributional answer where the point model ties no-motion).
Sample-display policy (disclosed): CoQ10 shows one settling + one tipping sample when both exist among the S;
other objects show the first two samples. All S outcomes are tallied in the remark.
    MUJOCO_GL=egl python render_droproll_demo.py
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, mujoco, imageio, torch
from scipy.spatial.transform import Rotation as R, Slerp
from PIL import Image, ImageDraw
import drop.render_grid as G
import drop.render_corpus_demo as CD                         # OBJS/NICE/load_episode/replay_frames reused
from drop.mesh_render_test import mesh_model, W, Hh
from drop.my_dataset_drop import DropRollDS
from drop.drop_diffusion import GateBDiT
from common.dit import cosine_acp, ddim_sample
from drop.drop_net import sixd_to_R
from common.paths import MODELS, REPO
S, GRAV = 16, 9.81   # S imagined futures per episode (only 2 are rendered; S sets the tally + which two are shown)


def imagined_frames(mdl, rnd, cam, meshV, R_rel, Rw_anchors, drop_h, n_settle=26, fps=15):
    """Deterministic ballistic fall (constant release orientation, closed-form height; render-only) then the
    sampled anchors slerped at floor level."""
    d = mujoco.MjData(mdl); frames = []
    Tf = np.sqrt(2 * max(drop_h, 1e-4) / GRAV)
    n_fall = max(3, int(round(Tf * fps * 4)))                # 4x slow-motion so the short fall is visible
    pz0 = -R_rel.apply(meshV)[:, 2].min()
    for t in np.linspace(0, Tf, n_fall, endpoint=False):
        z = drop_h - 0.5 * GRAV * t ** 2
        d.qpos[:3] = [0, 0, pz0 + max(z, 0)]; d.qpos[3:7] = G.wxyz(R_rel.as_quat()); mujoco.mj_forward(mdl, d)
        rnd.update_scene(d, camera=cam); frames.append(rnd.render().copy())
    Rk = R.from_matrix(np.stack([Rw.as_matrix() for Rw in Rw_anchors]))
    sl = Slerp(np.arange(len(Rw_anchors)), Rk)
    for t in np.linspace(0, len(Rw_anchors) - 1, n_settle):
        Rc = sl(t); pz = -Rc.apply(meshV)[:, 2].min()
        d.qpos[:3] = [0, 0, pz]; d.qpos[3:7] = G.wxyz(Rc.as_quat()); mujoco.mj_forward(mdl, d)
        rnd.update_scene(d, camera=cam); frames.append(rnd.render().copy())
    return frames


def remark(obj, gt_tipped, outcomes, cut, shown):
    im = Image.new("RGB", (int(W * 0.72), Hh), (15, 18, 24)); dr = ImageDraw.Draw(im)
    y = Hh // 2 - 130
    dr.text((18, y), CD.NICE[obj], fill=(255, 220, 120), font=G.FBIG); y += 52
    tally = f"{int((~outcomes).sum())} settle / {int(outcomes.sum())} tip"
    lines = [(f"actual outcome: {'TIPPED' if gt_tipped else 'settles back'}", (210, 215, 225)),
             (f"imagined futures (S={len(outcomes)}): {tally}", (140, 235, 160)),
             (f"(tip = net rotation > {cut:.0f} deg, data-derived)", (150, 165, 190)),
             (f"showing samples #{shown[0]+1} and #{shown[1]+1}", (150, 165, 190))]
    if gt_tipped and 0 < outcomes.sum() < len(outcomes):
        lines.append(("same release, diverging futures --", (255, 220, 120)))
        lines.append(("the distribution covers the tip", (255, 220, 120)))
        lines.append(("(34/36 CoQ10 tip episodes diverge)", (150, 165, 190)))
    for txt, col in lines:
        dr.text((18, y), txt, fill=col, font=G.FONT); y += 36
    return np.asarray(im)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)                              # reproducible diffusion draws
    st = np.load(f"{MODELS}/norm_stats/roll_corpus_stats.npz")
    stats = {"mean": st["mean"], "std": st["std"]}
    te = DropRollDS("test", stats=stats)
    mean = torch.tensor(stats["mean"], device=dev); std = torch.tensor(stats["std"], device=dev)
    model = GateBDiT(n_feat=13, use_latent=False, H=te.K).to(dev)
    model.load_state_dict(torch.load(f"{MODELS}/roll_corpus.pt", map_location=dev)); model.eval()
    acp = cosine_acp(1000, device=dev)
    # data-derived tip cutoff from CoQ10's GT net-rotation distribution (largest gap), as in train_droproll
    from drop.train_drop import geodesic
    co = [j for j in range(len(te)) if te.rows[j]["object"] == "CoQ10_BjTLbuRVt1t"]
    g0 = np.sort([float(te.netrot[j]) for j in co]); gi = int(np.diff(g0).argmax()); cut = float((g0[gi] + g0[gi + 1]) / 2)
    rows = []
    for obj in CD.OBJS:
        cand = sorted((j for j in range(len(te)) if te.rows[j]["object"] == obj and te.y_bt[j] == 1),
                      key=lambda j: (te.rows[j]["shard"], int(te.rows[j]["idx"])))
        if cand:
            j = cand[0]
        else:
            js = sorted((k for k in range(len(te)) if te.rows[k]["object"] == obj), key=lambda k: float(te.netrot[k]))
            j = js[len(js) // 2]
        r = te.rows[j]; z = CD.load_episode(r["shard"], r["idx"])
        op, oq = np.asarray(z["obj_pos"], float), np.asarray(z["obj_quat"], float)
        b = te[j]
        with torch.no_grad():
            cond = model.cond(*[b[k][None].to(dev) for k in ("pts", "base_rel", "closing", "table")], None)
            x0 = ddim_sample(model, cond.repeat_interleave(S, 0), te.K, acp, steps=25, device=dev)
            dR = sixd_to_R((x0 * std + mean)[..., :6])                        # (S,K,3,3)
        R_rel = R.from_matrix(te.R_rel[j].astype(np.float64))
        net = np.array([np.degrees(R.from_matrix(dR[s, -1].cpu().numpy()).magnitude()) for s in range(S)])
        tipped = net > cut
        gt_tipped = float(te.netrot[j]) > cut
        if gt_tipped and tipped.any() and (~tipped).any():                    # show the divergence when it exists
            shown = (int(np.nonzero(~tipped)[0][0]), int(np.nonzero(tipped)[0][0]))
        else:
            shown = (0, 1)
        mdl, _sc, _mt, _off = mesh_model(obj); meshV = (np.asarray(_mt.vertices) - _mt.vertices.mean(0)) * _sc
        rnd = mujoco.Renderer(mdl, height=Hh, width=W); cam = G.mesh_cam(meshV)
        gt = G.band(CD.replay_frames(mdl, rnd, cam, meshV, op, oq),
                    f"{CD.NICE[obj]}  —  recorded trajectory (ground truth)")
        cols = [gt]
        for si, tag in zip(shown, ("A", "B")):
            Rw = [R.from_matrix(dR[si, k].cpu().numpy()) * R_rel for k in range(te.K)]
            fr = imagined_frames(mdl, rnd, cam, meshV, R_rel, Rw, float(te.rows[j]["drop_h"]))
            cols.append(G.band(fr, f"imagined settle {tag} (fall = deterministic, render-only)"))
        n = max(len(c) for c in cols)
        cols = [c + [c[-1]] * (n - len(c)) for c in cols]
        rem = remark(obj, gt_tipped, tipped, cut, shown)
        rows.append([np.concatenate([cols[0][i], cols[1][i], cols[2][i], rem], axis=1) for i in range(n)])
        print(f"[{obj}] gt_tipped={gt_tipped} sampled net={np.round(net,1)} tally={int(tipped.sum())}/{S} tip shown={shown}")
    nmax = max(len(r_) for r_ in rows)
    grid = [np.concatenate([r_[min(i, len(r_) - 1)] for r_ in rows], axis=0) for i in range(nmax)]
    grid += [grid[-1]] * 12
    p = f"{REPO}/drop/figures/drop_roll_corpus.mp4"; imageio.mimsave(p, grid, fps=15, macro_block_size=1)
    print(f"wrote {p} ({len(grid)} frames, {grid[0].shape})")


if __name__ == "__main__":
    main()
