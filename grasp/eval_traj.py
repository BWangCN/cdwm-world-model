"""Phase-2 eval (Codex go/no-go): per-mode trajectory-SUMMARY error of the GT-mode expert, full vs pose-only vs the
per-mode-MEDIAN baseline, object-disjoint + object bootstrap. GO if full beats pose AND the per-mode median clearly.
Summary layout: [0:6]settle_6d [6:9]settle_mm [9]maxDrift° [10]maxDrift_mm [11]slip°/s [12]drop [13]dropT [14]heldFrac.

    python eval_traj.py traj_full traj_pose
"""
import os, sys, csv, numpy as np
from common.utils import sixd_to_R, geodesic_deg
from common.paths import OUTCOMES as OV
MODES = ["RIGID", "TRANS_SLIP", "PERS_SLIP", "DROP", "NEVER_LIFT"]


def errors(pred_summ_gtmode, gt):                             # pred/gt (N,15) RAW -> dict of per-sample errors
    Rp = sixd_to_R(pred_summ_gtmode[:, :6]); Rg = sixd_to_R(gt[:, :6])
    return dict(settle_rot_deg=geodesic_deg(Rp, Rg),
                settle_trans_mm=np.linalg.norm(pred_summ_gtmode[:, 6:9] - gt[:, 6:9], axis=1),
                maxdrift_deg=np.abs(pred_summ_gtmode[:, 9] - gt[:, 9]))


def train_medians():                                         # per-mode median summary from TRAIN split
    z = np.load(f"{OV}/traj_summaries.npz", allow_pickle=True); S = z["summ"]; u2i = {u: i for i, u in enumerate(z["uid"])}
    sp = {r["object_id"]: r["split"] for r in csv.DictReader(open(f"{OV}/outcome_split.csv"))}
    idx = list(csv.DictReader(open(f"{OV}/outcomes_index.csv")))
    from grasp.my_dataset_outcomes import FIVE
    med = np.zeros((5, 15), np.float32)
    for m, k in FIVE.items():
        rows = [r for r in idx if r["v2_outcome"] == m and sp.get(r["object_id"]) == "train"]
        if rows: med[k] = np.median(np.stack([S[u2i[r["uid"]]] for r in rows]), 0)
    return med


def gt_expert(d):                                            # GT-mode expert summary (N,15) -- ORACLE (mode teacher-forced)
    ps, y = d["pred_summ"], d["y5"]; return ps[np.arange(len(y)), y]


def mixture(d):                                             # deployment: sum_m P(m|x) summary_m (predicted mode, no oracle)
    return np.einsum("nk,nks->ns", d["mode_prob"], d["pred_summ"])


def agg(err, y, obj):                                        # overall + per-mode median error; object-bootstrap on overall
    out = {k: float(np.median(v)) for k, v in err.items()}
    out["per_mode"] = {MODES[m]: {k: float(np.median(v[y == m])) for k, v in err.items()} for m in range(5) if (y == m).sum()}
    return out


def main():
    a, b = (sys.argv[1:] + ["traj_full", "traj_pose"])[:2]
    D = {t: dict(np.load(f"traj_runs/{t}/test_pred.npz", allow_pickle=True)) for t in (a, b)}
    med = train_medians()
    y = D[a]["y5"]; obj = D[a]["obj"]; gt = D[a]["summ_raw"]
    E = {a: errors(gt_expert(D[a]), gt), b: errors(gt_expert(D[b]), gt),
         a + "_mix": errors(mixture(D[a]), gt),               # DEPLOYMENT: predicted-mode mixture (no oracle)
         "median": errors(med[y], gt)}                        # median baseline: per-mode median for GT mode
    print(f"{'metric':16s} | {a+'(oracle)':>14s} {a+'_mix':>12s} {b:>10s} {'median':>8s}   (median err; mix=deployment)")
    for k in ["settle_rot_deg", "settle_trans_mm", "maxdrift_deg"]:
        print(f"{k:16s} | {np.median(E[a][k]):14.2f} {np.median(E[a+'_mix'][k]):12.2f} {np.median(E[b][k]):10.2f} {np.median(E['median'][k]):8.2f}")
    print("\nper-mode settle_rot_deg (full):", {MODES[m]: round(float(np.median(E[a]['settle_rot_deg'][y == m])), 1) for m in range(5) if (y == m).sum()})
    # object-bootstrap: full beats pose AND median on settle_rot + settle_trans?
    uo = np.unique(obj); rng = np.random.default_rng(0); ix = {o: np.where(obj == o)[0] for o in uo}
    print("\n=== object-bootstrap 95% CI of Δmedian-error (positive = full better) ===")
    for k in ["settle_rot_deg", "settle_trans_mm", "maxdrift_deg"]:
        for base in (b, "median"):
            dl = []
            for _ in range(600):
                sel = np.concatenate([ix[o] for o in rng.choice(uo, len(uo), replace=True)])
                dl.append(np.median(E[base][k][sel]) - np.median(E[a][k][sel]))
            lo, hi = np.percentile(dl, [2.5, 97.5]); star = " *" if lo > 0 else ""
            print(f"  {k:16s} vs {base:8s}: {np.mean(dl):+.2f} [{lo:+.2f},{hi:+.2f}]{star}")
    print("  (* = full significantly better. GO if full beats BOTH pose and median on settle_rot/trans.)")


if __name__ == "__main__":
    main()
