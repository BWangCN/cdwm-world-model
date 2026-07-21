"""Object-level bootstrap CIs + paired per-object delta CIs on the headline numbers (Codex consolidation, no retrain).
Resample OBJECTS with replacement (B=1000), recompute the statistic over their episodes -> point + 95% CI. Paired deltas
(same episodes) are stronger than aggregate means. Uses saved predictions; rollout parts need roll_runs/*/metrics.npz.
    python bootstrap_ci.py
"""
import sys, numpy as np
sys.path.insert(0, ".")
from wm.common import sixd_to_R, geodesic_deg
from eval_traj import errors, mixture
L = lambda p: dict(np.load(p, allow_pickle=True))
RNG = np.random.default_rng(0); B = 1000


def _objidx(obj):
    uo = np.unique(obj); return uo, {o: np.where(obj == o)[0] for o in uo}


def oboot(vals, obj, reduce=np.median):
    uo, idx = _objidx(obj); pt = reduce(vals)
    bs = [reduce(vals[np.concatenate([idx[o] for o in RNG.choice(uo, len(uo), replace=True)])]) for _ in range(B)]
    lo, hi = np.percentile(bs, [2.5, 97.5]); return pt, lo, hi


def oboot_fn(fn, obj, *arrs):
    uo, idx = _objidx(obj); pt = fn(*arrs)
    bs = []
    for _ in range(B):
        sel = np.concatenate([idx[o] for o in RNG.choice(uo, len(uo), replace=True)])
        bs.append(fn(*[a[sel] for a in arrs]))
    lo, hi = np.percentile(bs, [2.5, 97.5]); return pt, lo, hi


def auroc(p, y):
    o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(len(p)); npos = y.sum(); nneg = len(y) - npos
    return (r[y == 1].sum() - npos * (npos - 1) / 2) / (npos * nneg + 1e-9)


def topk_settle(w, mu, gt):
    Rg = sixd_to_R(gt[:, :6]); best = np.full(len(gt), 1e9)
    for k in range(w.shape[1]): best = np.minimum(best, geodesic_deg(sixd_to_R(mu[:, k, :6]), Rg))
    return best                                                    # per-episode


def fmt(name, t): print(f"  {name:38s} {t[0]:7.3f}  [{t[1]:7.3f}, {t[2]:7.3f}]")


def main():
    mdn = L("traj_runs/mdn_full/test_pred.npz"); tf = L("traj_runs/traj_full/test_pred.npz"); cls = L("cls_runs/full/test_pred.npz")
    y = mdn["y5"]; gt = mdn["summ_raw"]; obj = mdn["obj"]; fail = (y >= 2).astype(float)
    assert np.array_equal(tf["y5"], y) and np.array_equal(cls["y5"], y), "slip test order mismatch"

    print("=== POINT ESTIMATES + object-bootstrap 95% CI (B=1000) ===")
    mdn_tk = topk_settle(mdn["weight"], mdn["means"], gt)
    lab = errors(mixture(tf), gt)["settle_rot_deg"]
    fmt("MDN top-K settle-rot deg (median)", oboot(mdn_tk, obj))
    fmt("labeled-mixture settle-rot deg (median)", oboot(lab, obj))
    fmt("classifier AUROC (fail)", oboot_fn(auroc, obj, cls["p2"][:, 1], fail))
    fmt("classifier accuracy", oboot_fn(lambda p, f: ((p >= .5) == f).mean(), obj, cls["p2"][:, 1], fail))

    print("\n=== PAIRED per-object delta CIs (negative = first better; same episodes) ===")
    fmt("MDN top-K  -  labeled-mixture (settle°)", oboot(mdn_tk - lab, obj, np.mean))

    # rollout metrics (need metrics.npz from eval_rollout)
    try:
        rf = L("roll_runs/roll_full/metrics.npz"); rp = L("roll_runs/roll_pose/metrics.npz")
        rcls = L("cls_runs/full/test_pred.npz")
        yr = rf["y5"]; obr = rf["obj"]; failr = (yr >= 2).astype(float)
        aligned = np.array_equal(rcls["y5"], yr)
        print("\n=== ROLLOUT point CIs ===")
        fmt("rollout energy score ALL (median)", oboot(rf["ES"], obr))
        fmt("no-motion energy score ALL (median)", oboot(rf["NMES"], obr))
        rtp = L("roll_runs/roll_full/test_pred.npz")
        fmt("rollout read-off AUROC", oboot_fn(auroc, obr, rtp["p2"][:, 1], failr))
        print("\n=== ROLLOUT paired per-object delta CIs ===")
        fmt("energy  rollout - no-motion (ALL)", oboot(rf["ES"] - rf["NMES"], obr, np.mean))
        for mi, m in [(0, "RIGID"), (3, "DROP"), (4, "NEVER_LIFT")]:
            g = yr == mi
            if g.sum(): fmt(f"energy rollout-nomo [{m}]", oboot(rf["ES"][g] - rf["NMES"][g], obr[g], np.mean))
        fmt("single-shot k10  rollout - no-motion", oboot(rf["G1"][:, -1] - rf["NOM"][:, -1], obr, np.mean))
        fmt("energy  full - pose (geometry sharpens)", oboot(rf["ES"] - rp["ES"], obr, np.mean))
        if aligned:
            br = lambda p, f: ((p - f) ** 2).mean()
            fmt("Brier  read-off - classifier", oboot(( rtp["p2"][:, 1] - failr) ** 2 - (rcls["p2"][:, 1] - failr) ** 2, obr, np.mean))
    except FileNotFoundError:
        print("\n[rollout metrics.npz not found yet — run eval_rollout first, then rerun]")


if __name__ == "__main__":
    main()
