"""One-time precompute for the baseline drop WM: stream the drop corpus once into a compact per-episode index
+ release/rest pose cache + object-disjoint split. Mirrors the slip pipeline's t0_poses precompute so the
dataloader hot path stays cheap (no tar streaming per __getitem__).

Release = trajectory frame 0 (obj_quat[0] == release_quat, verified); rest = last frame. basin_transition =
(release_basin != rest_basin). Writes to $CDWM_DROP/derived/.

    python precompute_drop.py
"""
import os, io, glob, json, csv, tarfile
import numpy as np

from common.paths import DROP_CORPUS as DROP
from common.paths import PCDIR
DERIVED = os.path.join(DROP, "derived")
os.makedirs(DERIVED, exist_ok=True)
MANI = json.load(open(f"{DROP}/manifest.json"))
HELDOUT = set(MANI["held_out_meshes"])
HAS_CLOUD = {os.path.basename(f)[:-4] for f in glob.glob(f"{PCDIR}/*.npz")}

FIELDS = ["object", "shard", "idx", "validity", "valid_training", "basin_transition",
          "tilt_deg", "drop_h", "yaw_rad", "net_rot_from_release_deg", "outcome_subclass",
          "release_basin", "rest_basin", "n_frames", "has_cloud", "held_out"]


def build_index():
    rows = []
    rq, rp, tq, tp = [], [], [], []                         # release quat/pos, rest quat/pos
    for tp_ in sorted(glob.glob(f"{DROP}/train_corpus/*.tar")):
        shard = os.path.basename(tp_)
        with tarfile.open(tp_) as tf:
            for i, m in enumerate(tf.getmembers()):
                if not m.name.endswith(".npz"): continue
                z = np.load(io.BytesIO(tf.extractfile(m).read()), allow_pickle=True)
                obj = str(z["object"])
                oq, op = z["obj_quat"], z["obj_pos"]
                rows.append(dict(object=obj, shard=shard, idx=i, validity=str(z["validity"]),
                                 valid_training=bool(z["valid_training"]), basin_transition=bool(z["basin_transition"]),
                                 tilt_deg=float(z["tilt_deg"]), drop_h=float(z["drop_h"]), yaw_rad=float(z["yaw_rad"]),
                                 net_rot_from_release_deg=float(z["net_rot_from_release_deg"]),
                                 outcome_subclass=str(z["outcome_subclass"]), release_basin=str(z["release_basin"]),
                                 rest_basin=str(z["rest_basin"]), n_frames=len(op),
                                 has_cloud=obj in HAS_CLOUD, held_out=obj in HELDOUT))
                rq.append(oq[0]); rp.append(op[0]); tq.append(oq[-1]); tp.append(op[-1])
    return rows, (np.array(rq, np.float32), np.array(rp, np.float32), np.array(tq, np.float32), np.array(tp, np.float32))


def make_split(rows, seed=0):
    """object-disjoint: held_out meshes -> test; remaining objects 85/15 train/val (by object)."""
    objs = sorted({r["object"] for r in rows})
    train_val = [o for o in objs if o not in HELDOUT]
    rng = np.random.default_rng(seed); rng.shuffle(train_val)
    n_val = max(1, int(0.15 * len(train_val)))
    val = set(train_val[:n_val]); split = {}
    for o in objs:
        split[o] = "test" if o in HELDOUT else ("val" if o in val else "train")
    return split


def main():
    rows, (rq, rp, tq, tp) = build_index()
    split = make_split(rows)
    with open(f"{DERIVED}/drop_index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS + ["split"]); w.writeheader()
        for r in rows: r["split"] = split[r["object"]]; w.writerow({k: r[k] for k in FIELDS + ["split"]})
    np.savez(f"{DERIVED}/drop_poses.npz", release_quat=rq, release_pos=rp, rest_quat=tq, rest_pos=tp)
    # report
    import collections
    n = len(rows); sp = collections.Counter(r["split"] for r in rows)
    vt = sum(r["valid_training"] for r in rows); bt = sum(r["basin_transition"] for r in rows)
    nc = sum(not r["has_cloud"] for r in rows)
    print(f"episodes: {n}  splits: {dict(sp)}  valid_training: {vt} ({vt/n:.1%})  basin_transition: {bt} ({bt/n:.1%})")
    print(f"no-cloud episodes: {nc}  objects: {len({r['object'] for r in rows})} "
          f"(test/heldout {len(HELDOUT)}, val {sum(v=='val' for v in split.values())}, "
          f"train {sum(v=='train' for v in split.values())})")
    print(f"wrote {DERIVED}/drop_index.csv + drop_poses.npz")

if __name__ == "__main__":
    main()
