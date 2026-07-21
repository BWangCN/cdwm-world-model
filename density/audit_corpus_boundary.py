"""Audit the drop corpus for near-boundary / metastable samples: how often does an object actually leave its
release basin (the samples where CoG *could* matter)? Streams train_corpus scalar fields only.

Signals: basin_transition (release_basin != rest_basin = crossed a boundary), net_rot_from_release_deg,
outcome_subclass (TIPPED/ROLLED/...), vs release tilt_deg. Reports overall + by object + by tilt bin.
NOTE basin_transition is a LOWER bound on "released near a boundary": episodes that teetered but fell back
have no transition. Still the cleanest stored proxy.

    python audit_corpus_boundary.py
"""
import io, os, glob, json, tarfile, collections
import numpy as np

ROOT = os.environ.get("CDWM_DROP", "/misc/kcgscratch1/MengyeGroup/yy5259/datasets/cdwm-drop-corpus")
SHAPE = {n: m.get("shape_class", "?") for n, m in json.load(open(f"{ROOT}/manifest.json"))["meshes"].items()}

def main():
    tot = 0
    valid = 0
    validity = collections.Counter()
    subclass = collections.Counter()
    bt = 0                                                   # basin_transition among valid
    netrot = []
    by_obj = collections.defaultdict(lambda: [0, 0])        # obj -> [valid, transitions]
    by_shape = collections.defaultdict(lambda: [0, 0])
    by_tilt = collections.defaultdict(lambda: [0, 0])       # tilt bin -> [valid, transitions]
    tiltbin = lambda t: "0-5" if t < 5 else "5-10" if t < 10 else "10-15" if t < 15 else "15+"

    for tp in sorted(glob.glob(f"{ROOT}/train_corpus/*.tar")):
        with tarfile.open(tp) as tf:
            for m in tf.getmembers():
                if not m.name.endswith(".npz"): continue
                z = np.load(io.BytesIO(tf.extractfile(m).read()), allow_pickle=True)
                tot += 1
                v = str(z["validity"]); validity[v] += 1
                if not bool(z["valid_training"]): continue
                valid += 1
                obj = str(z["object"]); t = float(z["tilt_deg"])
                tr = bool(z["basin_transition"])
                bt += tr; netrot.append(float(z["net_rot_from_release_deg"]))
                subclass[str(z["outcome_subclass"])] += 1
                by_obj[obj][0] += 1; by_obj[obj][1] += tr
                by_shape[SHAPE.get(obj, "?")][0] += 1; by_shape[SHAPE.get(obj, "?")][1] += tr
                b = tiltbin(t); by_tilt[b][0] += 1; by_tilt[b][1] += tr

    nr = np.array(netrot)
    print(f"total episodes: {tot}   valid_training: {valid} ({valid/tot:.1%})")
    print("validity:", dict(validity))
    print(f"\n=== near-boundary signal (among {valid} valid) ===")
    print(f"basin_transition (crossed to a NEW basin): {bt} = {bt/valid:.1%}")
    print(f"net_rot_from_release_deg: p50={np.percentile(nr,50):.1f}  p90={np.percentile(nr,90):.1f}  "
          f"p99={np.percentile(nr,99):.1f}  max={nr.max():.1f}")
    print(f"  frac net_rot>30 (TIPPED-ish): {(nr>30).mean():.1%}   >90: {(nr>90).mean():.1%}")
    print("outcome_subclass:", dict(subclass))
    print("\n=== basin_transition rate by release tilt ===")
    for b in ["0-5", "5-10", "10-15", "15+"]:
        if by_tilt[b][0]: print(f"  tilt {b:>6}: {by_tilt[b][1]}/{by_tilt[b][0]} = {by_tilt[b][1]/by_tilt[b][0]:.1%}")
    print("\n=== basin_transition rate by shape_class ===")
    for s, (n, k) in sorted(by_shape.items(), key=lambda x: -x[1][1] / max(x[1][0], 1)):
        print(f"  {s:>12}: {k}/{n} = {k/max(n,1):.1%}")
    print("\n=== objects MOST often crossing a boundary (top 15, min 100 valid) ===")
    ranked = sorted(((k / n, k, n, o) for o, (n, k) in by_obj.items() if n >= 100), reverse=True)
    for r, k, n, o in ranked[:15]: print(f"  {o:40s} {k:5d}/{n:<5d} = {r:.1%}")
    print(f"\n  ...{sum(1 for r,_,_,_ in ranked if r < 0.01)} of {len(ranked)} objects (>=100 eps) transition <1% of the time.")

if __name__ == "__main__":
    main()
