"""Honest eval for the HF-trained arms (fixes the shipped protocol's two inflations: test-set checkpoint selection +
oracle best-of-8-only). For an arm's run dir:
  - split heldout_object into OBJECT-DISJOINT val/test (50/50 by object),
  - SELECT the checkpoint by val overall geo (primary metric per wm/metrics.py),
  - report TEST per-band geo (primary) + dir_err + mag_err, at BOTH N=1 (deployable) and N=8 (oracle coverage),
  - alongside two trivial baselines: no-motion (identity) and train-median-pose.
Primary metric = geo (direction-AND-magnitude geodesic). dir_err/mag_err are the decomposition.

  python eval_hf.py --run wm_runs/colorless_s0/local_hf --arch dit_local --ds_mod my_dataset_hf_local
  python eval_hf.py --ckpt checkpoints/base700_s0.pt --arch dit --ds_mod my_dataset_hf   # shipped baseline (external anchor)
"""
import os, sys, glob, argparse, json
import numpy as np, torch
from common.dit import DiTWM, cosine_acp, ddim_sample
from common.dit_local import DiTWMLocal, DiTWMCov, DiTWMLR
from common.utils import sixd_to_R, geodesic_deg
import common.metrics as MET
import importlib
from torch.utils.data import DataLoader
BANDS = ["2-5", "5-15", "15-30", "30+"]


def build_model(ck, dev):
    arch = ck.get("arch", "dit"); Model = {"dit": DiTWM, "dit_local": DiTWMLocal, "dit_cov": DiTWMCov, "dit_lr": DiTWMLR}[arch]
    m = Model(n_feat=int(ck["n_feat"]), H=int(ck["H"]), D=int(ck["dim"]), depth=int(ck["depth"])).to(dev)
    m.load_state_dict(ck["model"]); m.eval(); return m


def obj_split(rows, frac=0.5, seed=0):
    objs = sorted({r["object"] for r in rows}); rng = np.random.default_rng(seed); rng.shuffle(objs)
    nval = int(len(objs) * frac); val = set(objs[:nval])
    return [r for r in rows if r["object"] in val], [r for r in rows if r["object"] not in val]


@torch.no_grad()
def encode_gts(model, ds, dev):
    dl = DataLoader(ds, 128, shuffle=False, num_workers=4); conds, gts, shp = [], [], []
    for b in dl:
        conds.append(model.encode(b["pts"].to(dev), b["base_rel"].to(dev), b["closing"].to(dev), b["table"].to(dev)))
        gts.append(b["target"].numpy()); shp += list(b["shape"])
    return torch.cat(conds), np.concatenate(gts), np.array(shp)


@torch.no_grad()
def predict(model, cond, G, dev, acp, H, N, sd, mu):
    G = G * sd + mu; M = len(G); Rg = sixd_to_R(G[:, -1, 3:]); tg = G[:, -1, :3]
    best = np.full(M, 1e9); Rp = np.tile(np.eye(3), (M, 1, 1)).astype(float); tp = np.zeros((M, 3))
    for _ in range(N):
        x0 = np.concatenate([ddim_sample(model, cond[i:i+256], H, acp, steps=50, device=dev).cpu().numpy()
                             for i in range(0, M, 256)]) * sd + mu
        Rpn = sixd_to_R(x0[:, -1, 3:]); g = geodesic_deg(Rpn, Rg)
        bt = g < best; best[bt] = g[bt]; Rp[bt] = Rpn[bt]; tp[bt] = x0[bt, -1, :3]
    return Rp, Rg, tp, tg


def band_table(Rp, Rg, tp, tg, shp):
    s = MET.stratified(Rp, Rg, t_pred=tp, t_gt=tg, strata={"shape": shp})["all"]
    row = {}
    for r in s["primary_stratified"]:
        row[r["band"]] = dict(geo=r["geo"]["median"], dir=r["dir_err"]["median"], mag=r["mag_err"]["median"], n=r["n"])
    row["overall_geo"] = s["overall_geo"]["median"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="wm_runs/colorless_sX/TAG (uses ckpt/ep*.pt series + best.pt)")
    ap.add_argument("--ckpt", default=None, help="single checkpoint (skips selection); e.g. shipped baseline")
    ap.add_argument("--arch", default="dit"); ap.add_argument("--ds_mod", default="my_dataset_hf")
    ap.add_argument("--norm", default=None, help="norm_stats.npz (default: run/norm_stats.npz or checkpoints/target_norm_stats.npz)")
    ap.add_argument("--cap", type=int, default=0, help="cap #objects per val/test (debug only; 0=all)")
    a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    DS = importlib.import_module(a.ds_mod)
    rows = DS.load_rows("heldout_object"); val, test = obj_split(rows)
    if a.cap:
        vk = sorted({r["object"] for r in val})[:a.cap]; tk = sorted({r["object"] for r in test})[:a.cap]
        val = [r for r in val if r["object"] in set(vk)]; test = [r for r in test if r["object"] in set(tk)]
    print(f"heldout {len(rows)} rows -> val {len(val)} (obj {len({r['object'] for r in val})}) / test {len(test)} (obj {len({r['object'] for r in test})})", flush=True)
    norm = a.norm or (f"{a.run}/norm_stats.npz" if a.run else "checkpoints/target_norm_stats.npz")
    st = np.load(norm); sd, mu = st["std"], st["mean"]

    # candidate checkpoints
    if a.ckpt:
        cands = [a.ckpt]
    else:
        cands = sorted(glob.glob(f"{a.run}/ckpt/ep*.pt")) + ([f"{a.run}/ckpt/best.pt"] if os.path.exists(f"{a.run}/ckpt/best.pt") else [])
    assert cands, "no checkpoints found"
    H = int(torch.load(cands[0], map_location="cpu")["H"]); acp = cosine_acp(int(torch.load(cands[0], map_location='cpu')["T"]), device=dev)
    DSval = DS.ChunkDS(val, H, stats={"mean": mu, "std": sd}, fixed_subsample=True)
    DStest = DS.ChunkDS(test, H, stats={"mean": mu, "std": sd}, fixed_subsample=True)

    # --- select checkpoint by val overall geo (N=1 for speed) ---
    sel, sel_geo = cands[0], 1e9
    if len(cands) > 1:
        for c in cands:
            m = build_model(torch.load(c, map_location=dev), dev); cond, G, shp = encode_gts(m, DSval, dev)
            Rp, Rg, tp, tg = predict(m, cond, G, dev, acp, H, 1, sd, mu)
            geo = float(np.median(MET.relrot_geodesic_deg(Rp, Rg)))
            if geo < sel_geo: sel_geo, sel = geo, c
            print(f"  val geo(N=1) {geo:6.2f}  {os.path.basename(c)}", flush=True)
    print(f"SELECTED {os.path.basename(sel)} (val geo N=1 {sel_geo:.2f})", flush=True)

    # --- evaluate selected on TEST at N=1 and N=8 ---
    m = build_model(torch.load(sel, map_location=dev), dev); cond, G, shp = encode_gts(m, DStest, dev)
    out = {"run": a.run or a.ckpt, "arch": a.arch, "selected": os.path.basename(sel), "val_geo_n1": round(sel_geo, 2), "test": {}}
    for N in (1, 8):
        Rp, Rg, tp, tg = predict(m, cond, G, dev, acp, H, N, sd, mu)
        out["test"][f"N{N}"] = band_table(Rp, Rg, tp, tg, shp)
    # --- trivial baselines on TEST (same Rg/tg via N=1 pass) ---
    Rp1, Rg, tp1, tg = predict(m, cond, G, dev, acp, H, 1, sd, mu)
    ident = band_table(np.tile(np.eye(3), (len(Rg), 1, 1)).astype(float), Rg, np.zeros_like(tg), tg, shp)  # no-motion
    trmed = median_pose(DS, sd, mu, H)
    trrow = band_table(np.tile(trmed, (len(Rg), 1, 1)), Rg, np.zeros_like(tg), tg, shp)
    out["baseline_no_motion"] = ident; out["baseline_train_median"] = trrow

    print("\n=== TEST per-band (median), primary=geo | " + (a.run or a.ckpt) + " ===")
    hdr = f"{'band':6s} {'n':>4s} | {'geo N1':>7s} {'geo N8':>7s} | {'dir N1':>7s} {'dir N8':>7s} | {'mag N1':>7s} || {'noMot geo':>9s} {'trMed geo':>9s}"
    print(hdr)
    for b in BANDS:
        n1, n8 = out["test"]["N1"].get(b), out["test"]["N8"].get(b)
        if not n1: continue
        print(f"{b:6s} {n1['n']:4d} | {fmt(n1['geo'])} {fmt(n8['geo'])} | {fmt(n1['dir'])} {fmt(n8['dir'])} | {fmt(n1['mag'])} || "
              f"{fmt(ident[b]['geo']):>9s} {fmt(trrow[b]['geo']):>9s}")
    print(f"overall geo: N1 {out['test']['N1']['overall_geo']}  N8 {out['test']['N8']['overall_geo']}  | no-motion {ident['overall_geo']}  train-median {trrow['overall_geo']}")
    jp = f"eval_hf_{(a.run or a.ckpt).replace('/','_').replace('.pt','')}.json"
    json.dump(out, open(jp, "w"), indent=1); print("wrote", jp)


def fmt(x): return f"{x:7.2f}" if isinstance(x, (int, float)) else f"{'-':>7s}"


def median_pose(DS, sd, mu, H, n=1500):
    rows = DS.load_rows("train"); rng = np.random.default_rng(0); sel = rng.choice(len(rows), min(n, len(rows)), replace=False)
    ds = DS.ChunkDS([rows[i] for i in sel], H, stats={"mean": mu, "std": sd}, fixed_subsample=True)
    last = np.stack([ds[i]["target"].numpy()[-1] for i in range(len(ds))]) * sd + mu   # (n,9) physical
    med6 = np.median(last[:, 3:], 0)
    return sixd_to_R(med6[None])[0]


if __name__ == "__main__":
    main()
