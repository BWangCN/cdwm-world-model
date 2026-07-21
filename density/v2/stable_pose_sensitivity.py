"""V2 (analytic): which objects' RESTING-POSE DISTRIBUTION is sensitive to the center of gravity?

Faster + more rigorous than noisy drop sims for the go/no-go: trimesh.compute_stable_poses takes the CoM as
input, so for each object we compare the stable-pose distribution at the geometric centroid vs a plausible
CoM offset (0.35 * half-extent) along each principal axis, over the WHOLE corpus. CoM matters for an object
iff shifting the CoM materially changes where it prefers to rest.

Metric = TV distance between the two resting-pose distributions (matched by resting-face normal), per axis;
headline = max over axes. Guarded against the DEGENERACY confound: near-spherical objects have many
near-equal stable poses (high N_eff = 1/sum p^2) and rest anywhere, so a high TV there is meaningless -->
flagged, not counted. A CoM-SENSITIVE object = few, well-separated stable poses (low N_eff) AND high TV.

    python stable_pose_sensitivity.py        # ranks all objects; writes stable_pose_sensitivity.csv
"""
import os, csv, glob, numpy as np, trimesh
from trimesh.poses import compute_stable_poses

PC = os.environ.get("CDWM_PCDIR",
    "/misc/kcgscratch1/MengyeGroup/yy5259/datasets/cdwm-grasp-dataset/outcomes_v2/point_clouds")
OFFSET_FRAC = 0.35
THRESH = 0.02                                          # prune stable poses below this probability


def stable(mesh, cm):
    T, P = compute_stable_poses(mesh, center_mass=cm, n_samples=1, threshold=THRESH)
    return T, P / P.sum() if P.sum() > 0 else P


def down_hist(T, P):                                   # resting-face signature -> prob (world-up in body frame)
    h = {}
    for t, p in zip(T, P):
        n = tuple(np.round(t[:3, :3].T @ [0, 0, 1], 1))
        h[n] = h.get(n, 0) + p
    return h


def tv(h0, h1):
    return 0.5 * sum(abs(h0.get(k, 0) - h1.get(k, 0)) for k in set(h0) | set(h1))


def analyze(obj):
    mu = np.load(f"{PC}/{obj}.npz")["mu"].astype(np.float64)
    if len(mu) > 6000: mu = mu[np.random.default_rng(0).choice(len(mu), 6000, replace=False)]
    m = trimesh.Trimesh(vertices=mu).convex_hull
    c = m.center_mass.copy()
    _, _, vt = np.linalg.svd(m.vertices - m.vertices.mean(0))
    half = np.array([(m.vertices @ vt[i]).ptp() / 2 for i in range(3)])
    T0, P0 = stable(m, c)
    if len(P0) == 0: return None
    n_eff = float(1.0 / np.sum(P0 ** 2))              # effective # stable poses (high => round/degenerate)
    h0 = down_hist(T0, P0)
    tvs = []
    for ai in range(3):
        T1, P1 = stable(m, c + OFFSET_FRAC * half[ai] * vt[ai])
        tvs.append(tv(h0, down_hist(T1, P1)))
    return dict(object=obj, n_stable=len(P0), n_eff=round(n_eff, 1), pmax=round(float(P0.max()), 2),
                tv_long=round(tvs[0], 2), tv_mid=round(tvs[1], 2), tv_short=round(tvs[2], 2),
                tv_max=round(max(tvs), 2), long_mm=round(float(half[0]) * 2000, 1))


def main():
    objs = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{PC}/*.npz"))
    rows = []
    for o in objs:
        try:
            r = analyze(o)
            if r: rows.append(r)
        except Exception as e:
            print(f"  {o}: ERR {e}")
    # CoM-sensitive AND meaningful = high TV with FEW well-separated poses (n_eff small); round objects flagged
    for r in rows: r["degenerate"] = r["n_eff"] > 8          # ~round: rests anywhere, TV meaningless
    outp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stable_pose_sensitivity.csv")
    with open(outp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    meaningful = [r for r in rows if not r["degenerate"]]
    print(f"{len(rows)} objects ({sum(r['degenerate'] for r in rows)} flagged near-round/degenerate).\n")
    print("=== CoM-SENSITIVE (non-degenerate, top 15 by tv_max) ===")
    print(f"{'object':40s} {'n_eff':>6} {'pmax':>5} {'tv_max':>7} {'tv(long/mid/short)':>20}")
    for r in sorted(meaningful, key=lambda x: -x["tv_max"])[:15]:
        print(f"  {r['object']:38s} {r['n_eff']:>6} {r['pmax']:>5} {r['tv_max']:>7}   {r['tv_long']}/{r['tv_mid']}/{r['tv_short']}")
    print("\n=== CoM-ROBUST (non-degenerate, bottom 10 by tv_max) ===")
    for r in sorted(meaningful, key=lambda x: x["tv_max"])[:10]:
        print(f"  {r['object']:38s} {r['n_eff']:>6} {r['pmax']:>5} {r['tv_max']:>7}")
    import numpy as _np
    tvm = _np.array([r["tv_max"] for r in meaningful])
    print(f"\nnon-degenerate tv_max: median={_np.median(tvm):.2f}  frac>0.3={( tvm>0.3).mean():.0%}  frac>0.5={(tvm>0.5).mean():.0%}")
    print(f"wrote {outp}")

if __name__ == "__main__":
    main()
