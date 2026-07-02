"""Shared eval core for the convergence diagnosis. best-of-8 DDIM predict for ANY checkpoint path; per-band + per-shape
dir_err/mag_err (EVAL_STANDARD wm.metrics); mechanism (gripper-frame tilt-direction circular-corr + WM-vs-closing-axis) on
the NORMAL near-success held-out subset. env gaussianobject."""
import os, sys
import numpy as np, torch
from torch.utils.data import DataLoader
import importlib
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from my_dataset import load_rows                          # rows are frame-agnostic
from wm.dit import DiTWM, cosine_acp, ddim_sample
from wm.common import sixd_to_R, quat_to_R, geodesic_deg
import wm.metrics as MET
BANDS = ["2-5", "5-15", "15-30", "30+"]; SHAPES = ["convex", "bottle-can", "concave", "handled", "other"]
UP = np.array([0.0, 0.0, 1.0])


@torch.no_grad()
def predict(ckpt_path, sd, mu, rows, dev, N=8, steps=50, ds_mod="my_dataset"):
    ChunkDS = importlib.import_module(ds_mod).ChunkDS       # frame-swappable: my_dataset | my_dataset_frame
    ck = torch.load(ckpt_path, map_location=dev)
    H = int(ck["H"]); model = DiTWM(n_feat=int(ck.get("n_feat", 12)), H=H, D=int(ck["dim"]), depth=int(ck["depth"])).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    dl = DataLoader(ChunkDS(rows, H, stats={"mean": mu, "std": sd}, fixed_subsample=True), 128, shuffle=False, num_workers=3)
    conds, gts, shp = [], [], []
    for b in dl:
        conds.append(model.encode(b["pts"].to(dev), b["base_rel"].to(dev), b["closing"].to(dev), b["table"].to(dev)))
        gts.append(b["target"].numpy()); shp += list(b["shape"])
    G = np.concatenate(gts) * sd + mu; cond = torch.cat(conds); M = len(G); acp = cosine_acp(int(ck["T"]), device=dev)
    Rg = sixd_to_R(G[:, -1, 3:]); tg = G[:, -1, :3]; best = np.full(M, 1e9); Rp = np.tile(np.eye(3), (M, 1, 1)).astype(float); tp = np.zeros((M, 3))
    for _ in range(N):
        x0 = np.concatenate([ddim_sample(model, cond[i:i+256], H, acp, steps=steps, device=dev).cpu().numpy() for i in range(0, M, 256)]) * sd + mu
        Rpn = sixd_to_R(x0[:, -1, 3:]); g = geodesic_deg(Rpn, Rg); bt = g < best; best[bt] = g[bt]; Rp[bt] = Rpn[bt]; tp[bt] = x0[bt, -1, :3]
    return dict(Rp=Rp, Rg=Rg, tp=tp, tg=tg, shape=np.array(shp))


def bandrow(suite, b, fld):
    r = next((x for x in suite["primary_stratified"] if x["band"] == b), None)
    return r[fld]["median"] if r and r[fld]["median"] is not None else None


def per_band_shape(P):
    s = MET.stratified(P["Rp"], P["Rg"], t_pred=P["tp"], t_gt=P["tg"], strata={"shape": P["shape"]})
    return dict(dir={b: bandrow(s["all"], b, "dir_err") for b in BANDS},
                mag={b: bandrow(s["all"], b, "mag_err") for b in BANDS},
                geo=s["all"]["overall_geo"]["median"],
                dir_shape={sh: {b: bandrow(s["shape"].get(sh, {"primary_stratified": []}), b, "dir_err") for b in BANDS} for sh in SHAPES})


def circ(gaz, paz):
    gb, pb = gaz - np.angle(np.mean(np.exp(1j*gaz))), paz - np.angle(np.mean(np.exp(1j*paz)))
    return float(np.sum(np.sin(gb)*np.sin(pb)) / (np.sqrt(np.sum(np.sin(gb)**2)*np.sum(np.sin(pb)**2)) + 1e-12))


def ho_meta(ho):
    out = []
    for r in ho:
        z = np.load(r["episode"], allow_pickle=True)
        out.append((float(geodesic_deg(quat_to_R(z["obj_quat"][0].astype(float)), np.eye(3))), z["closing"].astype(float), z["base_quat"][0].astype(float)))
    return out


def mechanism(P, meta):
    _, mag, der, mag_gt = MET.decompose(P["Rp"], P["Rg"])
    est = np.array([m[0] for m in meta]); clv = np.array([m[1] for m in meta]); bq = np.array([m[2] for m in meta])
    near = (est < 60) & np.isfinite(der) & (mag <= 8) & (mag_gt >= 5) & (mag_gt < 30)
    G, Q = P["Rg"][near], P["Rp"][near]
    gaz = np.arctan2((G @ UP)[:, 1], (G @ UP)[:, 0]); paz = np.arctan2((Q @ UP)[:, 1], (Q @ UP)[:, 0])
    Rgb = np.array([quat_to_R(q) for q in bq[near]]); clg = np.einsum("nij,nj->ni", np.transpose(Rgb, (0, 2, 1)), clv[near]); claz = np.arctan2(clg[:, 1], clg[:, 0])
    spr = lambda a: round(float(np.degrees(np.sqrt(-2*np.log(max(np.hypot(np.mean(np.cos(a)), np.mean(np.sin(a))), 1e-9))))), 1)
    return dict(n=int(near.sum()), circ_corr=round(circ(gaz, paz), 3),
                wm_vs_closing=round(float(np.median(np.degrees(np.abs(np.angle(np.exp(1j*(paz-claz))))))), 1),
                gt_vs_closing=round(float(np.median(np.degrees(np.abs(np.angle(np.exp(1j*(gaz-claz))))))), 1),
                wm_spread=spr(paz), gt_spread=spr(gaz))
