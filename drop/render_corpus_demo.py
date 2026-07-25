"""PRIMARY drop demo, faithful to the corpus formulation (release above table -> free-fall -> impact -> settle).
Three held-out objects; per row: LEFT = the RECORDED corpus trajectory replayed (obj_pos/obj_quat @31.25Hz -- real
free-fall, impact, settle; no re-simulation), MIDDLE = the baseline endpoint WM's PREDICTED resting pose (from the
cloud + release only, i.e. before the drop happens), RIGHT = remark (release tilt, outcome, model vs no-motion
geodesic error, object-level median). Selection policy (disclosed, uniform): basin-transition episodes exist for
exactly ONE held-out object (CoQ10, 44 tips; the corpus is shape-dominated) -> CoQ10 shows its FIRST (by shard,idx)
transition episode; objects with no transitions show their MEDIAN-net-rotation episode (a typical settle). Errors
reported as-is plus the object-level median. The CoQ10 tip is the honest tail case: the point model cannot predict
which way it falls (ties no-motion) -- exactly the failure that motivates the Gate B distributional extension.
    MUJOCO_GL=egl python -m drop.render_corpus_demo
"""
import os, io, tarfile
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, mujoco, imageio, torch
from scipy.spatial.transform import Rotation as R, Slerp
from PIL import Image, ImageDraw
import drop.render_grid as G                                 # mesh_cam / band / wxyz / FONT / FBIG
from drop.mesh_render_test import mesh_model, W, Hh
from drop.my_dataset_drop import DropDS
from drop.drop_net import DropNet, sixd_to_R
from common.paths import DROP_CORPUS, MODELS, REPO

# Representative trio (per-object medians, model vs no-motion): 9/12 held-out objects are clear wins -> two typical
# wins (rubiks 1.9 vs 7.8, apple 4.6 vs 7.5) + the honest tail case (CoQ10 21.7 vs 6.8, the only transition object).
OBJS = ["077_rubiks_cube", "013_apple", "CoQ10_BjTLbuRVt1t"]
NICE = {"077_rubiks_cube": "Rubik's cube", "013_apple": "Apple", "CoQ10_BjTLbuRVt1t": "Supplement bottle"}
UP = 3                                                       # slerp upsampling factor for smooth playback


def load_episode(shard, idx):                                # exactly precompute_drop's member indexing
    with tarfile.open(f"{DROP_CORPUS}/train_corpus/{shard}") as tf:
        m = tf.getmembers()[int(idx)]
        return np.load(io.BytesIO(tf.extractfile(m).read()), allow_pickle=True)


def replay_frames(mdl, rnd, cam, meshV, op, oq):
    """Replay the recorded trajectory. Orientation = recorded obj_quat; position re-anchored to the render frame:
    xy relative to the final rest point, z offset so the mesh rests exactly on the floor at the final pose (the
    corpus asset origin differs from our mesh center; only the anchor moves, the recorded motion is untouched)."""
    Rk = R.from_quat([np.r_[q[1:], q[0]] for q in oq])
    sl = Slerp(np.arange(len(oq)), Rk); tt = np.linspace(0, len(oq) - 1, len(oq) * UP)
    pz_rest = -Rk[-1].apply(meshV)[:, 2].min()
    z_off = pz_rest - op[-1, 2]
    d = mujoco.MjData(mdl); frames = []
    for t in tt:
        k = int(round(t)); Rc = sl(t)
        xy = op[k, :2] - op[-1, :2]
        pz = max(op[k, 2] + z_off, -Rc.apply(meshV)[:, 2].min())   # penetration guard (asset-origin mismatch)
        d.qpos[:3] = [xy[0], xy[1], pz]; d.qpos[3:7] = G.wxyz(Rc.as_quat()); mujoco.mj_forward(mdl, d)
        rnd.update_scene(d, camera=cam); frames.append(rnd.render().copy())
    return frames


def static_frames(mdl, rnd, cam, meshV, Rw, n):              # one pose, resting on the floor
    d = mujoco.MjData(mdl); pz = -Rw.apply(meshV)[:, 2].min()
    d.qpos[:3] = [0, 0, pz]; d.qpos[3:7] = G.wxyz(Rw.as_quat()); mujoco.mj_forward(mdl, d)
    rnd.update_scene(d, camera=cam); f = rnd.render().copy()
    return [f] * n


def remark(obj, tilt, geo, geo0, med, n, tipped):
    im = Image.new("RGB", (int(W * 0.72), Hh), (15, 18, 24)); dr = ImageDraw.Draw(im)
    y = Hh // 2 - 140
    dr.text((18, y), NICE[obj], fill=(255, 220, 120), font=G.FBIG); y += 52
    ok = geo < 15                                            # outcome-scale success, not merely edging no-motion
    lines = [(f"release: stable pose + {tilt:.0f} deg tilt", (210, 215, 225)),
             (f"outcome: {'TIPPED (rare tail case)' if tipped else 'settles back'}", (210, 215, 225)),
             (f"model error: {geo:.1f} deg", (140, 235, 160) if ok else (235, 160, 140)),
             (f"no-motion baseline: {geo0:.1f} deg", (150, 165, 190))]
    if med is not None:
        lines.append((f"object median: {med:.1f} deg (n={n})", (150, 165, 190)))
    if tipped:
        lines += [("the point model cannot predict the tip", (235, 160, 140)),
                  ("-> the Gate B distributional WM", (255, 220, 120))]
    for txt, col in lines:
        dr.text((18, y), txt, fill=col, font=G.FONT); y += 36
    return np.asarray(im)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    te = DropDS("test", modality="full", frame="release")
    model = DropNet().to(dev)
    model.load_state_dict(torch.load(f"{MODELS}/baseline/full_release.pt", map_location=dev)); model.eval()
    tp_path = f"{REPO}/drop_runs/full_release/test_pred.npz"     # per-object medians (regenerate: python -m drop.eval_drop)
    tp = np.load(tp_path, allow_pickle=True) if os.path.exists(tp_path) else None
    rows = []
    for obj in OBJS:
        cand = sorted((j for j in range(len(te)) if te.rows[j]["object"] == obj and te.y_bt[j] == 1),
                      key=lambda j: (te.rows[j]["shard"], int(te.rows[j]["idx"])))
        if cand:
            j = cand[0]                                     # FIRST settled basin_transition episode (CoQ10 only)
        else:                                               # no transitions for this object -> the MEDIAN-net-rotation
            js = sorted((k for k in range(len(te)) if te.rows[k]["object"] == obj),
                        key=lambda k: float(te.netrot[k]))  #   episode = a TYPICAL settle
            j = js[len(js) // 2]
        r = te.rows[j]
        z = load_episode(r["shard"], r["idx"])
        op, oq = np.asarray(z["obj_pos"], float), np.asarray(z["obj_quat"], float)
        b = te[j]
        with torch.no_grad():
            _, rot = model(*[b[k][None].to(dev) for k in ("pts", "base_rel", "closing", "table")])
            dR = sixd_to_R(rot)[0].cpu().numpy()
            dRg = sixd_to_R(b["rest6d"][None].to(dev))[0].cpu().numpy()
        geo = np.degrees(np.arccos(np.clip((np.trace(dR.T @ dRg) - 1) / 2, -1, 1)))
        geo0 = np.degrees(np.arccos(np.clip((np.trace(dRg) - 1) / 2, -1, 1)))          # no-motion: dR = I
        med = float(np.median(tp["geo"][tp["obj"] == obj])) if tp is not None else None
        nobj = int((tp["obj"] == obj).sum()) if tp is not None else 0
        R_rel = R.from_matrix(te.R_rel[j].astype(np.float64))
        R_pred_world = R.from_matrix(dR) * R_rel                                       # dR @ R_rel
        mdl, _sc, _mt, _off = mesh_model(obj); meshV = (np.asarray(_mt.vertices) - _mt.vertices.mean(0)) * _sc
        rnd = mujoco.Renderer(mdl, height=Hh, width=W); cam = G.mesh_cam(meshV)
        gt = G.band(replay_frames(mdl, rnd, cam, meshV, op, oq),
                    f"{NICE[obj]}  —  recorded corpus trajectory (free-fall / impact / settle)")
        mo = G.band(static_frames(mdl, rnd, cam, meshV, R_pred_world, len(gt)),
                    f"{NICE[obj]}  —  model predicts the resting pose (before the drop)")
        rem = remark(obj, float(r["tilt_deg"]), geo, geo0, med, nobj, bool(cand))
        rows.append([np.concatenate([gt[i], mo[i], rem], axis=1) for i in range(len(gt))])
        print(f"[{obj}] shard={r['shard']} idx={r['idx']} frames={len(oq)} tilt={float(r['tilt_deg']):.1f} "
              f"geo={geo:.1f} nomotion={geo0:.1f} med={med}")
    grid = [np.concatenate([r_[min(i, len(r_) - 1)] for r_ in rows], axis=0) for i in range(max(len(r_) for r_ in rows))]
    grid += [grid[-1]] * 12
    p = f"{REPO}/drop/figures/drop_corpus_demo.mp4"; imageio.mimsave(p, grid, fps=15, macro_block_size=1)
    print(f"wrote {p} ({len(grid)} frames, {grid[0].shape})")


if __name__ == "__main__":
    main()
