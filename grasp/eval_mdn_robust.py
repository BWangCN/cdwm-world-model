"""MDN robustness controls (Codex): NLL + top-K settle-rot coverage across K-sweep and seeds, + emergent-mode stability
(does the same physical-mode structure recur across seeds?). Pure analysis of saved test_pred."""
import sys, math, numpy as np
from common.utils import sixd_to_R, geodesic_deg
from grasp.eval_traj import MODES
L = lambda t: dict(np.load(f"traj_runs/{t}/test_pred.npz", allow_pickle=True))


def nll(d):
    w, mu, sc, y = d["weight"], d["means"], np.maximum(d["scales"], 1e-6), d["summ_raw"]
    logN = (-0.5 * ((y[:, None, :] - mu) / sc) ** 2 - np.log(sc) - 0.5 * math.log(2 * math.pi)).sum(-1)
    m = (np.log(w + 1e-9) + logN).max(1); return float(np.mean(-(m + np.log(np.exp(np.log(w + 1e-9) + logN - m[:, None]).sum(1)))))


def topk(d):
    Rg = sixd_to_R(d["summ_raw"][:, :6]); best = np.full(len(Rg), 1e9)
    for k in range(d["means"].shape[1]): best = np.minimum(best, geodesic_deg(sixd_to_R(d["means"][:, k, :6]), Rg))
    return float(np.median(best))


def dom_modes(d):                                  # alive components -> dominant physical mode (for stability)
    w = d["weight"]; y = d["y5"]; comp = w.argmax(1)
    out = []
    for k in range(w.shape[1]):
        if (comp == k).sum() > 30: out.append(MODES[int(np.bincount(y[comp == k], minlength=5).argmax())])
    return sorted(out)


def main():
    print(f"{'config':10s} {'K':>3s} {'NLL↓':>7s} {'topK-cov°↓':>10s}")
    for tag, K in [("mdn_full", 8), ("mdn_K4", 4), ("mdn_K6", 6), ("mdn_K12", 12), ("mdn_s1", 8), ("mdn_s2", 8)]:
        try:
            d = L(tag); print(f"{tag:10s} {K:>3d} {nll(d):>7.2f} {topk(d):>10.2f}")
        except Exception as e:
            print(f"{tag:10s} {K:>3d}  (missing: {e})")
    print("\nemergent-mode stability across seeds (dominant mode of each alive component):")
    for tag in ["mdn_full", "mdn_s1", "mdn_s2"]:
        try: print(f"  {tag}: {dom_modes(L(tag))}")
        except Exception: print(f"  {tag}: (missing)")
    print("  (stable = the same physical modes recur across seeds; K-sweep: NLL/coverage not a lucky K.)")


if __name__ == "__main__":
    main()
