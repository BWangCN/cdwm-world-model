"""Evaluate ONE arm (tag) on the new protocol: for each seed, SELECT the checkpoint by held-out dir_err (dominant bands,
best-of-8) among plateau checkpoints, then full held-out per-band+shape dir_err/mag_err/geo + mechanism, plus train per-band
dir_err. Aggregate mean±std over seeds -> eval_<tag>.json. Frame-swappable via --ds_mod. Run: eval_arm.py --tag frame --ds_mod my_dataset_frame."""
import os, sys, json, glob, re, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_lib import predict, per_band_shape, mechanism, ho_meta, load_rows, BANDS, SHAPES
DOM = ["5-15", "15-30"]
SEL_EPS = [520, 560, 600, 640, 680, 700]                  # plateau-region candidates for dir_err-selection


def agg(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return [round(float(np.mean(v)), 1), round(float(np.std(v)), 1), len(v)] if v else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--ds_mod", default="my_dataset")
    ap.add_argument("--wm_root", default="wm_runs"); ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; seeds = [int(x) for x in a.seeds.split(",")]
    ho = load_rows("heldout_object"); tr = load_rows("train")
    rng = np.random.default_rng(0); tr = [tr[i] for i in rng.choice(len(tr), min(3000, len(tr)), replace=False)]
    meta = ho_meta(ho); per = []
    for s in seeds:
        d = f"{a.wm_root}/colorless_s{s}/{a.tag}"; ckd = f"{d}/ckpt"
        if not os.path.exists(f"{ckd}/best.pt"): print(f"skip s{s} (no ckpt)"); continue
        st = np.load(f"{d}/norm_stats.npz"); sd, mu = st["std"], st["mean"]
        # select checkpoint by held-out dominant-band dir_err
        cands = [f"{ckd}/ep{e:04d}.pt" for e in SEL_EPS if os.path.exists(f"{ckd}/ep{e:04d}.pt")] + [f"{ckd}/best.pt"]
        best_score, sel = 1e9, None
        for c in cands:
            r = per_band_shape(predict(c, sd, mu, ho, dev, ds_mod=a.ds_mod))
            sc = np.mean([r["dir"][b] for b in DOM])
            if sc < best_score: best_score, sel, sel_r = sc, c, r
        print(f"  {a.tag} s{s}: selected {os.path.basename(sel)} (dom dir {best_score:.1f})", flush=True)
        Ph = predict(sel, sd, mu, ho, dev, ds_mod=a.ds_mod); rh = per_band_shape(Ph); mech = mechanism(Ph, meta)
        Pt = predict(sel, sd, mu, tr, dev, ds_mod=a.ds_mod); rt = per_band_shape(Pt)
        per.append(dict(seed=s, sel=os.path.basename(sel), held_dir=rh["dir"], held_mag=rh["mag"], geo=rh["geo"],
                        held_dir_shape=rh["dir_shape"], train_dir=rt["dir"], mech=mech))
    res = dict(tag=a.tag, ds_mod=a.ds_mod, n_seeds=len(per), per_seed=[{k: p[k] for k in ("seed", "sel")} for p in per])
    for fld in ("held_dir", "held_mag", "train_dir"):
        res[fld] = {b: agg([p[fld][b] for p in per]) for b in BANDS}
    res["held_dir_shape"] = {sh: {b: agg([p["held_dir_shape"][sh][b] for p in per]) for b in BANDS} for sh in SHAPES}
    res["geo"] = agg([p["geo"] for p in per])
    res["mech"] = {k: agg([p["mech"][k] for p in per]) for k in ["circ_corr", "wm_vs_closing", "gt_vs_closing", "wm_spread", "gt_spread"]}
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"eval_{a.tag}.json"), "w"), indent=1)
    print(f"\n{a.tag}: held dir_err", {b: res['held_dir'][b] for b in BANDS})
    print(f"{a.tag}: circ_corr {res['mech']['circ_corr']}  wm_vs_closing {res['mech']['wm_vs_closing']} (GT {res['mech']['gt_vs_closing']})")
    print(f"wrote eval_{a.tag}.json")


if __name__ == "__main__":
    main()
