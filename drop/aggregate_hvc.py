"""Aggregate drop/hull_vs_coacd.py array outputs (gateb_runs/hull_vs_coacd/*.json) and re-stratify the core Gate B
result (gateb_runs/eval_pred_reldisjoint.npz: distribution>point, hidden-CoM-causal) by the boundary-regime
geometry gap, to check whether the mechanism holds under more faithful collision geometry.
    python -m drop.aggregate_hvc
"""
import glob, json, os
import numpy as np
from common.paths import REPO

CONCAVE = {"Schleich_Hereford_Bull", "Schleich_Therizinosaurus_ln9cruulPqc", "Schleich_African_Black_Rhino", "022_windex_bottle"}


def agg(rows, name):
    if not rows:
        print(f"  {name:20s} (none)"); return
    a = np.array([r["agree_hull_vs_coacd"] for r in rows]); f = np.array([r["agree_floor_1p5deg"] for r in rows])
    g = np.array([r["geom_gap"] for r in rows]); md = np.array([r["median_basin_diff_deg"] for r in rows])
    print(f"  {name:20s} n={len(rows):2d} | hull-vs-CoACD agree {a.mean():.2f} | 1.5deg floor {f.mean():.2f} | "
          f"geom_gap {g.mean():+.2f} | median rest-diff {np.median(md):.1f} deg")


def main():
    rows = [json.load(open(f)) for f in sorted(glob.glob(f"{REPO}/gateb_runs/hull_vs_coacd/*.json"))]
    ok = [r for r in rows if "agree_hull_vs_coacd" in r]
    print(f"objects: {len(rows)} total | {len(ok)} evaluated | {len(rows)-len(ok)} skipped")
    print("\n=== HULL-vs-CoACD RESTING ROBUSTNESS ===")
    agg(ok, "ALL")
    agg([r for r in ok if r["object"] not in CONCAVE], "convex-ish (global)")
    agg([r for r in ok if r["object"] in CONCAVE], "strongly concave")

    p = f"{REPO}/gateb_runs/eval_pred_reldisjoint.npz"
    if not os.path.exists(p):
        print("\n[re-stratify Gate B] skipped (run `python -m drop.eval_gateb` first)"); return
    d = np.load(p, allow_pickle=True)
    obj = d["obj"]; point = d["point_nll"]; diff = d["diff_nll"]; oracle = d["diff_oracle_nll"]
    gap = {r["object"]: r["geom_gap"] for r in ok}
    have = sorted(gap); gv = np.array([gap[o] for o in have])
    order = np.argsort(gv); n = len(have); t1, t2 = n // 3, 2 * n // 3
    strata = {"low-gap": np.array(have)[order[:t1]], "mid-gap": np.array(have)[order[t1:t2]], "high-gap": np.array(have)[order[t2:]]}
    print(f"\n=== Gate B core claim (reldisjoint, known objects) re-stratified by boundary geom_gap (n_obj={len(have)}) ===")
    for name, names in strata.items():
        m = np.isin(obj, list(names))
        if not m.any(): continue
        print(f"  {name:10s} n_obj={len(names):2d} n_ep={int(m.sum()):5d} | dist>point {(point[m]-diff[m]).mean():+.3f} "
              f"| oracle>diff(CoM causal) {(diff[m]-oracle[m]).mean():+.3f}")
    per_obj_gain = np.array([(point[obj == o] - diff[obj == o]).mean() if (obj == o).any() else np.nan for o in have])
    ok2 = ~np.isnan(per_obj_gain)
    print(f"  corr(geom_gap, per-object dist>point gain) = {np.corrcoef(gv[ok2], per_obj_gain[ok2])[0,1]:+.2f}")


if __name__ == "__main__":
    main()
