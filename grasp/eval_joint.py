"""(colleague's suggestion, validation) Compare the JOINT model (shared encoder -> MDN + binary head) against our
SEPARATE MDN (traj_runs/mdn_full) and SEPARATE classifier (cls_runs/full), same v2 split. Decisive question:
does joint success supervision HELP the distributional WM, or does it merely match the separate decomposition?
    python eval_joint.py --tag joint_full   [--tag joint_pose]
"""
import sys, math, argparse, numpy as np
from common.utils import sixd_to_R, geodesic_deg
L = lambda p: dict(np.load(p, allow_pickle=True))


def mdn_nll(w, mu, sc, y):                                       # physical-space diagonal-Gaussian mixture NLL (per-sample)
    sc = np.maximum(sc, 1e-6)
    logN = (-0.5 * ((y[:, None, :] - mu) / sc) ** 2 - np.log(sc) - 0.5 * math.log(2 * math.pi)).sum(-1)
    m = (np.log(w + 1e-9) + logN).max(1)
    return -(m + np.log(np.sum(np.exp(np.log(w + 1e-9) + logN - m[:, None]), 1)))


def topk_settle(w, mu, gt):
    Rg = sixd_to_R(gt[:, :6]); best = np.full(len(gt), 1e9)
    for k in range(w.shape[1]): best = np.minimum(best, geodesic_deg(sixd_to_R(mu[:, k, :6]), Rg))
    return np.median(best)


def rc_auc(sev, fail):
    o = np.argsort(-sev); return round(float(np.mean([fail[o[int(r * len(o)):]].mean() for r in np.linspace(0, 0.5, 11)])), 3)


def reject30(sev, fail):
    o = np.argsort(-sev); n = int(0.3 * len(o)); return fail[o[n:]].mean(), fail[o[:n]].mean()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="joint_full"); a = ap.parse_args()
    J = L(f"traj_runs/{a.tag}/test_pred.npz"); M = L("traj_runs/mdn_full/test_pred.npz"); C = L("cls_runs/full/test_pred.npz")
    y = J["y5"]; gt = J["summ_raw"]; fail = (y >= 2).astype(float)
    assert np.array_equal(M["y5"], y) and np.array_equal(C["y5"], y), "align mismatch (test order must match)"
    print(f"=== JOINT vs SEPARATE (tag={a.tag}, n={len(y)}, base fail {fail.mean():.3f}) ===\n")
    print("-- distributional WM (MDN) --")
    print(f"  NLL (lower better):    joint {np.mean(mdn_nll(J['weight'],J['means'],J['scales'],gt)):.2f}   separate {np.mean(mdn_nll(M['weight'],M['means'],M['scales'],gt)):.2f}")
    print(f"  top-K settle-rot° med: joint {topk_settle(J['weight'],J['means'],gt):.2f}   separate {topk_settle(M['weight'],M['means'],gt):.2f}")
    print("\n-- binary success/fail ranking --")
    jr, jp = reject30(J["p2"][:, 1], fail); cr, cp = reject30(C["p2"][:, 1], fail)
    print(f"  risk-cov AUC (lower):  joint {rc_auc(J['p2'][:,1],fail)}   separate-cls {rc_auc(C['p2'][:,1],fail)}   random {rc_auc(np.random.default_rng(0).random(len(y)),fail)}")
    print(f"  reject30% KEPT-fail:   joint {jr:.3f}   separate-cls {cr:.3f}   (REJECT-precision joint {jp:.3f} / cls {cp:.3f})")
    # Brier for calibration sanity
    print(f"  Brier P(fail):         joint {np.mean((J['p2'][:,1]-fail)**2):.3f}   separate-cls {np.mean((C['p2'][:,1]-fail)**2):.3f}")
    print("\nRead: joint should MATCH separate on all rows. Beats -> joint training helps; ties -> separate decomposition sufficient.")


if __name__ == "__main__":
    main()
