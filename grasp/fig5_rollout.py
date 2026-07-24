"""Fig 5 — short-window rollout WM (Codex consolidation). Two panels, CPU-only (loads roll_runs/*/metrics.npz):
 (A) horizon curve: median drift° per step for single-shot N1 vs best-of-N vs no-motion (shows N1<no-motion, coverage>);
 (B) paired energy-score delta (rollout - no-motion) per outcome with object-bootstrap 95% CI (negative=model better as a
     DISTRIBUTION; RIGID positive=over-disperses). Okabe-Ito palette, matches figures/fig1-4.
    python fig5_rollout.py
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from grasp.bootstrap_ci import oboot
from grasp.eval_traj import MODES
OI = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})
L = lambda p: dict(np.load(p, allow_pickle=True))


def main():
    rf = L("roll_runs/roll_full/metrics.npz"); y = rf["y5"]; obj = rf["obj"]; K = int(rf["K"])
    ks = np.arange(1, K + 1)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.0))

    # (A) horizon curve
    axA.plot(ks, np.median(rf["Gbest"], 0), "-o", color=OI[2], lw=2, ms=4, label="best-of-N (coverage)")
    axA.plot(ks, np.median(rf["G1"], 0), "-s", color=OI[1], lw=2, ms=4, label="single-shot N=1")
    axA.plot(ks, np.median(rf["NOM"], 0), "--", color=OI[6], lw=2, label="no-motion")
    axA.set_xlabel("horizon step k (dt=0.032s)"); axA.set_ylabel("object drift error (°, median)")
    axA.set_title("Rollout is a DISTRIBUTION, not a point\n(coverage beats no-motion; single-shot does not)")
    axA.legend(frameon=False, fontsize=9)

    # (B) paired energy delta (rollout - no-motion) per outcome, object-bootstrap 95% CI
    rows = [("ALL", np.ones(len(y), bool))] + [(MODES[m], y == m) for m in range(5)]
    labs, pts, los, his = [], [], [], []
    for name, g in rows:
        if g.sum() < 5: continue
        p, lo, hi = oboot(rf["ES"][g] - rf["NMES"][g], obj[g], np.mean)
        labs.append(name); pts.append(p); los.append(p - lo); his.append(hi - p)
    ypos = np.arange(len(labs))[::-1]
    cols = [OI[0] if p < 0 else OI[1] for p in pts]                 # blue=better(neg), orange=worse(pos)
    axB.barh(ypos, pts, xerr=[los, his], color=cols, height=0.6, error_kw=dict(lw=1.2, capsize=3))
    axB.axvline(0, color="gray", lw=1)
    axB.set_yticks(ypos); axB.set_yticklabels(labs)
    axB.set_xlabel("energy score:  rollout − no-motion  (↓ = model better)")
    axB.set_title("Distributional win is significant per outcome\n(negative = better; RIGID over-disperses)")
    for yy, p in zip(ypos, pts): axB.text(p + (0.08 if p >= 0 else -0.08), yy, f"{p:+.2f}", va="center", ha="left" if p >= 0 else "right", fontsize=8)
    fig.tight_layout(); fig.savefig("figures/fig5_rollout.png"); print("wrote figures/fig5_rollout.png")


if __name__ == "__main__":
    main()
