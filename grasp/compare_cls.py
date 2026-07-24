"""Grasp-outcome metrics + full-vs-pose_only comparison at NATURAL prevalence, with object-level bootstrap CIs (Codex's
success criterion: cloud+grasp beats pose-only on held-out objects in FAILURE-AUPRC and calibrated 2-tier Brier/NLL, CI
excluding 0). Reads cls_runs/<tag>/test_pred.npz. FAILURE=class1(2-tier), DROP=class3(5-way).

    python compare_cls.py full pose
"""
import sys, numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix

FIVE = ["RIGID", "TRANS_SLIP", "PERS_SLIP", "DROP", "NEVER_LIFT"]


def ece(p, y, bins=15):                                        # expected calibration error (top-label)
    conf = p.max(1); pred = p.argmax(1); acc = (pred == y).astype(float); e = 0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (conf >= lo) & (conf < lo + 1 / bins)
        if m.sum(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e


def metrics(d):
    p2, y2, p5, y5 = d["p2"], d["y2"], d["p5"], d["y5"]
    bal2 = np.mean([(p2.argmax(1)[y2 == c] == c).mean() for c in (0, 1)])
    bal5 = np.mean([(p5.argmax(1)[y5 == c] == c).mean() for c in range(5) if (y5 == c).sum()])
    return dict(
        fail_auprc=average_precision_score(y2 == 1, p2[:, 1]), fail_auroc=roc_auc_score(y2, p2[:, 1]),
        brier2=np.mean((p2[:, 1] - (y2 == 1)) ** 2), nll2=-np.mean(np.log(p2[np.arange(len(y2)), y2] + 1e-9)),
        ece2=ece(p2, y2), bal_acc2=bal2, bal_acc5=bal5,
        macroF1_5=f1_score(y5, p5.argmax(1), average="macro"),
        drop_auprc=average_precision_score(y5 == 3, p5[:, 3]), drop_recall=(p5.argmax(1)[y5 == 3] == 3).mean())


def boot(dfull, dpose, fn, n=1000):                            # object-level bootstrap delta CI (full - pose)
    objs = np.array(dfull["obj"]); uo = np.unique(objs); rng = np.random.default_rng(0); dl = []
    idx = {o: np.where(objs == o)[0] for o in uo}
    for _ in range(n):
        pick = rng.choice(uo, len(uo), replace=True); sel = np.concatenate([idx[o] for o in pick])
        dl.append(fn(_sub(dfull, sel)) - fn(_sub(dpose, sel)))
    lo, hi = np.percentile(dl, [2.5, 97.5]); return float(np.mean(dl)), float(lo), float(hi)


def _sub(d, sel): return {k: (d[k][sel] if k in ("p2", "y2", "p5", "y5", "obj") else d[k]) for k in d}


def main():
    tags = sys.argv[1:] or ["full", "pose"]
    D = {t: dict(np.load(f"cls_runs/{t}/test_pred.npz", allow_pickle=True)) for t in tags}
    print(f"{'metric':14s} | " + " | ".join(f"{t:>10s}" for t in tags))
    M = {t: metrics(D[t]) for t in tags}
    for k in ["fail_auprc", "fail_auroc", "brier2", "nll2", "ece2", "bal_acc2", "bal_acc5", "macroF1_5", "drop_auprc", "drop_recall"]:
        print(f"{k:14s} | " + " | ".join(f"{M[t][k]:10.3f}" for t in tags))
    # base rates
    y2 = D[tags[0]]["y2"]; print(f"\nFAILURE base rate {(y2==1).mean():.3f}  (AUPRC baseline);  DROP base rate {(D[tags[0]]['y5']==3).mean():.3f}")
    if len(tags) >= 2:
        f, p = D[tags[0]], D[tags[1]]
        print(f"\n=== {tags[0]} vs {tags[1]}: object-bootstrap 95% CI of Δ (positive = {tags[0]} better) ===")
        for name, fn in [("FAILURE AUPRC", lambda d: average_precision_score(d["y2"] == 1, d["p2"][:, 1])),
                         ("2-tier Brier↓", lambda d: -np.mean((d["p2"][:, 1] - (d["y2"] == 1)) ** 2)),
                         ("2-tier NLL↓", lambda d: -(-np.mean(np.log(d["p2"][np.arange(len(d["y2"])), d["y2"]] + 1e-9)))),
                         ("bal-acc 2tier", lambda d: np.mean([(d["p2"].argmax(1)[d["y2"] == c] == c).mean() for c in (0, 1)]))]:
            m, lo, hi = boot(f, p, fn); star = " *" if (lo > 0 or hi < 0) else ""
            print(f"  Δ {name:16s} {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]{star}")
        print("  (* = CI excludes 0. Success = cloud beats pose on FAILURE-AUPRC and Brier/NLL with * )")


if __name__ == "__main__":
    main()
