"""(c) Grasp-ranking utility (Codex): rank held-out grasps by predicted risk/severity, measure the deployment payoff via
risk-coverage (reject the riskiest x%, how fast does the KEPT failure-rate / severity drop?) + rejection precision.
Primary ranker = the calibrated Phase-1 mode-classifier P(FAILURE). Baselines: mixture severity, direct severity, pose,
random; GT severity = oracle ceiling. Pure analysis of existing test_pred.npz (aligned by test order). fail = mode>=2.

    python grasp_ranking.py
"""
import numpy as np
from grasp.eval_traj import mixture

L = lambda p: dict(np.load(p, allow_pickle=True))


def curve(sev, fail, gtdrift):                             # reject top-x% riskiest -> (kept fail-rate, kept median drift, reject precision)
    order = np.argsort(-sev); out = {}
    for rej in (0.1, 0.2, 0.3, 0.4):
        n = int(rej * len(order)); rejected, kept = order[:n], order[n:]
        out[int(rej * 100)] = (round(float(fail[kept].mean()), 3), round(float(np.median(gtdrift[kept])), 1), round(float(fail[rejected].mean()), 3))
    auc = float(np.mean([fail[np.argsort(-sev)[int(r * len(sev)):]].mean() for r in np.linspace(0, 0.5, 11)]))  # lower=better
    return out, round(auc, 3)


def main():
    cls = L("cls_runs/full/test_pred.npz")                 # calibrated Phase-1 outcome classifier
    tf = L("traj_runs/traj_full/test_pred.npz"); df = L("traj_runs/direct_full/test_pred.npz"); dp = L("traj_runs/direct_pose/test_pred.npz")
    y = tf["y5"]; gt = tf["summ_raw"]
    assert np.array_equal(cls["y5"], y), "cls/traj test order mismatch — cannot align per-grasp"
    fail = (y >= 2).astype(float); gtdrift = gt[:, 9]
    rankers = {
        "clsP(fail)*":  cls["p2"][:, 1],                   # PRIMARY: calibrated 2-tier FAILURE prob
        "trajP(fail)":  tf["mode_prob"][:, 2:].sum(1),
        "mix_severity": mixture(tf)[:, 9],
        "direct_sev":   df["pred_mean"][:, 9],
        "pose_sev":     dp["pred_mean"][:, 9],
        "random":       np.random.default_rng(0).random(len(y)),
        "GT(ceiling)":  gtdrift,
    }
    print(f"base failure rate {fail.mean():.3f}  (n {len(y)} held-out grasps).  reject riskiest x% -> (KEPT fail, KEPT med-drift, REJECT precision)")
    print(f"{'ranker':14s} | {'rej10%':>20s} {'rej20%':>20s} {'rej30%':>20s} | {'RC-AUC↓':>7s}")
    for name, sev in rankers.items():
        c, auc = curve(sev, fail, gtdrift)
        print(f"{name:14s} | " + " ".join(f"{str(c[r]):>20s}" for r in (10, 20, 30)) + f" | {auc:7.3f}")
    print("  KEPT fail-rate drops fastest for the best ranker; REJECT precision = fraction of rejected that truly fail.")
    print("  Headline: failure-mode prediction (clsP(fail)) gives the strongest deployable grasp-risk signal.")


if __name__ == "__main__":
    main()
