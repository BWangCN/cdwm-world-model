"""B eval-hardening: stratify the drop-WM rotation error by difficulty. Uses the per-episode dumps from
eval_drop.py (drop_runs/<tag>/test_pred.npz: geo=model err, geo0=GT rotation magnitude = no-motion err, obj).

Difficulty axis = the GT rotation magnitude (geo0): how far the object actually reoriented from release. Low =
stable (returns to release basin), high = tipped / near-boundary. Shows (a) the model beats no-motion across the
regime and especially captures the TAIL, (b) where the release frame helps vs object-frame.

    python drop_strata.py
"""
import os, numpy as np

ARMS = ["full_release", "full_obj", "pose_only_release"]
BANDS = [(0, 2), (2, 5), (5, 15), (15, 30), (30, 1e9)]
BLAB = ["0-2", "2-5", "5-15", "15-30", "30+"]
RNG = np.random.default_rng(0)


def oboot_med(vals, obj, B=1000):
    uo = np.unique(obj); idx = {o: np.where(obj == o)[0] for o in uo}
    bs = [np.median(vals[np.concatenate([idx[o] for o in RNG.choice(uo, len(uo), True)])]) for _ in range(B)]
    return np.median(vals), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    D = {a: np.load(f"drop_runs/{a}/test_pred.npz", allow_pickle=True) for a in ARMS
         if os.path.exists(f"drop_runs/{a}/test_pred.npz")}
    ref = D["full_release"]; g0 = ref["geo0"]                  # GT rotation magnitude = difficulty
    print(f"stratifying by GT rotation magnitude (=no-motion error). n={len(g0)}\n")
    print(f"{'band(deg)':>9} {'n':>6} {'no-motion':>10} " + " ".join(f'{a:>16}' for a in D))
    for (lo, hi), lab in zip(BANDS, BLAB):
        m = (g0 >= lo) & (g0 < hi); n = int(m.sum())
        if n == 0: continue
        nm = np.median(g0[m])
        cells = []
        for a in D:
            v = D[a]["geo"][m]; cells.append(f"{np.median(v):>7.1f}° (n{n})" if False else f"{np.median(v):>15.1f}°")
        print(f"{lab:>9} {n:>6} {nm:>9.1f}° " + " ".join(cells))
    # tail focus (>=15 deg): does the model capture tips, and does the release frame help there? (object-bootstrap)
    tail = g0 >= 15; ntail = int(tail.sum())
    print(f"\nTAIL (GT rotation >=15deg, n={ntail} = the tips/near-boundary episodes):")
    for a in D:
        pt, lo, hi = oboot_med(D[a]["geo"][tail], D[a]["obj"][tail])
        print(f"  {a:20s} model geo median {pt:5.1f}° [{lo:.1f},{hi:.1f}]")
    pt, lo, hi = oboot_med(ref["geo0"][tail], ref["obj"][tail])
    print(f"  {'no-motion':20s}            {pt:5.1f}° [{lo:.1f},{hi:.1f}]  (= the GT rotation there)")


if __name__ == "__main__":
    main()
