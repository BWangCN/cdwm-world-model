"""Data-derived concavity audit for the convex-hull scope (see docs/00_overview.md "Geometry: the convex-hull scope").

Both the simulator and the world model operate on the object point-cloud convex hull. This audit measures how good
that approximation is, over the Gate B objects that have a source mesh, with two proxies:
  (1) global convexity ratio = mesh_volume / hull_volume         -- coarse, global shape.
  (2) contact-facing: stable-pose count of the real mesh vs its convex hull (trimesh.compute_stable_poses, same CoM
      convention) -- directly measures whether convexification collapses resting modes (the bull 8->3 effect).
The convex/concave split is read off the ratio DISTRIBUTION (largest natural gap), not a hard-coded threshold.

    python -m drop.concavity_audit
"""
import glob, os, warnings
import numpy as np, trimesh
from common.paths import OBJECTS, GATEB_DIR, REPO
warnings.filterwarnings("ignore")
THR = 0.02  # a resting pose counts if it holds >= 2% of the time
UNFAITHFUL = 3  # a mesh-having object is "hull-unfaithful" if the hull loses >= this many resting modes


def n_stable(mesh):
    try:
        _, p = mesh.compute_stable_poses(n_samples=1, threshold=0.0)
        return int((np.asarray(p) >= THR).sum())
    except Exception:
        return -1


def _match_dir(name):                       # npz object names are truncated at <U71; match a mesh dir by prefix
    for d in os.listdir(OBJECTS):
        if d.startswith(name) or name.startswith(d):
            return d
    return None


def stratify_transfer():
    """Robustness check (doc "Geometry" section): is the grounded-CoM transfer gain an artifact of concave hulls?
    Split the per-object transfer gain (no_latent NLL - grounded NLL) by hull fidelity. Needs the eval output
    gateb_runs/transfer_hardening.npz (regenerate with `python -m drop.eval_transfer_hardening`)."""
    p = os.path.join(REPO, "gateb_runs", "transfer_hardening.npz")
    if not os.path.exists(p):
        print("\n[transfer stratification] skipped (run `python -m drop.eval_transfer_hardening` first)")
        return
    d = np.load(p, allow_pickle=True)
    obj = d["grounded_oracle__obj"]; g = d["grounded_oracle__nll"]; b = d["no_latent__nll"]
    strata = {"hull-faithful": [], "hull-unfaithful": [], "point-cloud-only": []}
    for o in sorted(set(obj.tolist())):
        dd = _match_dir(o); mp = f"{OBJECTS}/{dd}/mesh.obj" if dd else None
        if mp and os.path.exists(mp):
            m = trimesh.load(mp, force="mesh")
            loss = n_stable(m) - n_stable(m.convex_hull)
            strata["hull-unfaithful" if loss >= UNFAITHFUL else "hull-faithful"].append(o)
        else:
            strata["point-cloud-only"].append(o)
    print("\n[transfer stratification] grounded-CoM gain (no_latent NLL - grounded NLL) by hull fidelity:")
    for k, objs in strata.items():
        mask = np.isin(obj, objs)
        gain = (b[mask] - g[mask]).mean() if mask.any() else float("nan")
        print(f"  {k:18s} {len(objs):>2} obj: gain {gain:+.3f} (n={int(mask.sum())})")
    print(f"  {'OVERALL':18s} {len(set(obj.tolist())):>2} obj: gain {(b - g).mean():+.3f} (n={len(obj)})")
    print("  (positive in every stratum => the effect is not a convex-hull artifact)")


def main():
    objs = sorted(os.path.basename(f).replace("_gateb_s0.npz", "") for f in glob.glob(f"{GATEB_DIR}/*_gateb_s0.npz"))
    objs = [o for o in objs if o != "hammer"]
    haveM = [o for o in objs if os.path.exists(f"{OBJECTS}/{o}/mesh.obj")]
    print(f"Gate B objects: {len(objs)} | have mesh: {len(haveM)} | point-cloud-only (hull is the only geometry): {len(objs)-len(haveM)}")

    rows = []  # (obj, ratio, n_mesh, n_hull)
    for o in haveM:
        try:
            m = trimesh.load(f"{OBJECTS}/{o}/mesh.obj", force="mesh"); ch = m.convex_hull
            rows.append((o, min(abs(m.volume) / abs(ch.volume), 1.0), n_stable(m), n_stable(ch)))
        except Exception as e:
            print(f"  skip {o}: {e}")

    v = np.array([r for _, r, _, _ in rows]); sv = np.sort(v); gi = int(np.diff(sv).argmax())
    print(f"\n[proxy 1] global convexity ratio over {len(v)} meshes: "
          f"quantiles 10/25/50/75/90 = " + " ".join(f"{q:.2f}" for q in np.percentile(v, [10, 25, 50, 75, 90])))
    print(f"  largest natural gap {sv[gi]:.2f} -> {sv[gi+1]:.2f}  => data-driven convex/concave split ~{(sv[gi]+sv[gi+1])/2:.2f}")

    valid = [(o, r, nm, nh) for o, r, nm, nh in rows if nm > 0 and nh > 0]
    nmv = np.array([nm for _, _, nm, _ in valid]); nhv = np.array([nh for _, _, _, nh in valid])
    loss = nmv - nhv; vr = np.array([r for _, r, _, _ in valid])
    print(f"\n[proxy 2] stable-pose count real mesh vs convex hull (n={len(valid)}): mean mesh {nmv.mean():.1f} hull {nhv.mean():.1f}")
    print(f"  resting modes LOST to the hull: mean {loss.mean():.1f} median {int(np.median(loss))} max {loss.max()}")
    print(f"  corr(convexity ratio, modes lost) = {np.corrcoef(vr, loss)[0,1]:+.2f} (the two proxies disagree: volume alone does not flag the problem objects)")

    rows.sort(key=lambda x: x[2] - x[3])  # by modes lost
    print("\nhull-faithful (few/no modes lost) -> demo-safe:")
    for o, r, nm, nh in rows[:6]:
        print(f"  ratio {r:.2f}  modes mesh={nm:>2} hull={nh:>2}  {o}")
    print("strongly concave (hull collapses resting modes) -> out of scope:")
    for o, r, nm, nh in sorted(rows, key=lambda x: -(x[2] - x[3]))[:6]:
        print(f"  ratio {r:.2f}  modes mesh={nm:>2} hull={nh:>2}  (lost {nm-nh})  {o}")

    stratify_transfer()


if __name__ == "__main__":
    main()
