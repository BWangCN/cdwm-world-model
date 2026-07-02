"""STAGE 1 (env: gaussianobject). Select ~16-18 NORMAL near-success grasps (rigid margin-2 successes; EXCLUDE extreme
cup-rim style grasps via establishment-reorientation >60deg), in the dominant tilt bands (5-15/15-30) + a few concave 30+,
spanning held-out+train and convex+concave+bottle-can. Roll out the WM (seed 0, best-of-8) and save GT + WM per-step SE(3)
+ dir_err/mag_err for the STAGE-2 direction-annotated render. 'Typical' = pick per cell the grasp whose dir_err is closest
to the band's WM median (magnitude right, direction off ~20deg), except concave-30+ where we keep the ~43deg failure."""
import os, sys, csv, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from make_videos import load_model, rollout
from wm.common import sixd_to_R, quat_to_R, geodesic_deg
import wm.metrics as MET

CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "example_data", "chunk_index", "chunks_corrected.csv")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "viz_out")
EST_MAX = 60.0                                        # exclude establishment-reorientation > 60deg (cup-rim outliers)
MAG_OK = 8.0                                          # 'magnitude right'
# target cells: (split, shape, band, target_dir_err) ; concave-30+ keeps the high-dir failure
TARGETS = [("heldout_object", "convex", "5-15", 20), ("heldout_object", "convex", "15-30", 21),
           ("train", "convex", "5-15", 20), ("train", "convex", "15-30", 21),
           ("heldout_object", "concave", "5-15", 20), ("heldout_object", "concave", "15-30", 28),
           ("heldout_object", "concave", "30+", 43), ("train", "concave", "5-15", 20),
           ("train", "concave", "15-30", 22), ("train", "concave", "30+", 40),
           ("heldout_object", "bottle-can", "5-15", 25), ("heldout_object", "bottle-can", "15-30", 22),
           ("train", "bottle-can", "5-15", 25), ("train", "bottle-can", "15-30", 20),
           ("heldout_object", "convex", "15-30", 21), ("train", "concave", "5-15", 20)]   # a few dups -> 2 per hot cell


def establishment(ep):
    z = np.load(ep, allow_pickle=True)
    return float(geodesic_deg(quat_to_R(z["obj_quat"][0].astype(float)), np.eye(3)))


def main():
    rows = list(csv.DictReader(open(CSV)))
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["split"], r["shape"], r["band"])].append(r)
    rng = np.random.default_rng(20260701)
    # gather candidates (unique episodes) per needed cell, filter establishment<60
    need = {(s, sh, b) for s, sh, b, _ in TARGETS}
    cand = {}
    for key in need:
        rs = cells.get(key, [])
        idx = rng.permutation(len(rs))[:12]
        keep = []
        for i in idx:
            try:
                if establishment(rs[i]["episode"]) < EST_MAX:
                    keep.append(rs[i])
            except Exception:
                pass
        cand[key] = keep
    # roll out all candidates once
    allc = {r["episode"]: r for ks in cand.values() for r in ks}
    crows = list(allc.values())
    m, acp, sd, mu, H = load_model(0)
    GT, PR, _ = rollout(m, acp, sd, mu, H, crows)
    Rg = sixd_to_R(GT[:, -1, 3:]); Rp = sixd_to_R(PR[:, -1, 3:])
    geo, mag, der, _ = MET.decompose(Rp, Rg)
    stat = {crows[i]["episode"]: dict(dir=float(der[i]) if np.isfinite(der[i]) else 999,
                                      mag=float(mag[i]), row=crows[i], PR=PR[i]) for i in range(len(crows))}
    # pick per target: mag_ok, dir closest to target; avoid repeats
    picked, used = [], set()
    for s, sh, b, tgt in TARGETS:
        pool = [stat[r["episode"]] for r in cand.get((s, sh, b), []) if r["episode"] in stat and r["episode"] not in used]
        pool = [p for p in pool if p["mag"] <= MAG_OK and p["dir"] < 900]
        if not pool: continue
        p = min(pool, key=lambda x: abs(x["dir"] - tgt)); used.add(p["row"]["episode"]); picked.append(p)
    # save
    eps = [p["row"]["episode"] for p in picked]
    PRg = np.stack([sixd_to_R(p["PR"][:, 3:]) for p in picked])           # [M,H,3,3] WM cumulative gripper-frame rot
    np.savez(f"{OUT}/direction_grasps.npz",
             episodes=np.array(eps), object=np.array([p["row"]["object"] for p in picked]),
             shape=np.array([p["row"]["shape"] for p in picked]), band=np.array([p["row"]["band"] for p in picked]),
             split=np.array([p["row"]["split"] for p in picked]), net_tilt=np.array([float(p["row"]["net_tilt"]) for p in picked], np.float32),
             dir_err=np.array([p["dir"] for p in picked], np.float32), mag_err=np.array([p["mag"] for p in picked], np.float32),
             establishment=np.array([establishment(e) for e in eps], np.float32), PRg=PRg.astype(np.float32), H=H)
    print(f"selected {len(picked)} normal near-success grasps:")
    for p in picked:
        r = p["row"]
        print(f"  {r['split'][:4]:4s} {r['shape'][:9]:9s} {r['band']:6s} nomtilt {float(r['net_tilt']):5.1f}  dir_err {p['dir']:5.1f}  mag_err {p['mag']:4.1f}  {r['object'][:20]}")


if __name__ == "__main__":
    main()
