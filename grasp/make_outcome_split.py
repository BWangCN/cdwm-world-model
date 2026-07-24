"""Object-disjoint split for the outcomes_v2 grasp-outcome task, with near-duplicate FAMILY merge (Codex: merge cups/
vessel/variant families before splitting so near-identical assets never straddle train/test). Writes outcome_split.csv
(object_id, family, object_sets, n_episodes, split) — the manifest that prevents later leakage ambiguity.

    python make_outcome_split.py
"""
import os, sys, csv, re, collections, json
import numpy as np
from common.paths import OUTCOMES as OV


def family(oid):
    m = re.match(r"^(\d+)-[a-z]_(.+)$", oid)                 # YCB variant set 065-a_cups -> 065_cups
    if m: return f"{m.group(1)}_{m.group(2)}"
    return oid                                              # google-scanned assets are unique


def main():
    rows = list(csv.DictReader(open(f"{OV}/outcomes_index.csv")))
    objs = sorted(set(r["object_id"] for r in rows))
    ep = collections.Counter(r["object_id"] for r in rows)
    osets = collections.defaultdict(set)
    for r in rows: osets[r["object_id"]].add(r["object_set"])
    fam = {o: family(o) for o in objs}
    fams = sorted(set(fam.values()))
    # assign FAMILIES to splits 60/15/25 by a deterministic shuffle (seed fixed)
    rng = np.random.default_rng(0); order = rng.permutation(len(fams))
    n = len(fams); ntr, nva = int(0.60 * n), int(0.15 * n)
    fsplit = {}
    for rank, fi in enumerate(order):
        fsplit[fams[fi]] = "train" if rank < ntr else ("val" if rank < ntr + nva else "test")
    # write manifest
    out = f"{OV}/outcome_split.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["object_id", "family", "object_sets", "n_episodes", "split"])
        for o in objs:
            w.writerow([o, fam[o], "|".join(sorted(osets[o])), ep[o], fsplit[fam[o]]])
    # stats
    sp = {o: fsplit[fam[o]] for o in objs}
    print(f"objects {len(objs)}  families {len(fams)}  -> {out}")
    for s in ["train", "val", "test"]:
        so = [o for o in objs if sp[o] == s]; ne = sum(ep[o] for o in so)
        print(f"  {s:5s}: {len(so):3d} objects  {ne:6d} episodes")
    # outcome balance + multi-set assets (Codex leak check)
    multi = [o for o in objs if len(osets[o]) > 1]
    print(f"assets in >1 object_set: {len(multi)}  (family merge keeps them intact regardless)")
    byspl = collections.defaultdict(collections.Counter)
    for r in rows: byspl[sp[r["object_id"]]][r["v2_outcome"]] += 1
    print("outcome counts per split:")
    for s in ["train", "val", "test"]:
        tot = sum(byspl[s].values())
        print(f"  {s:5s} (n{tot}): " + "  ".join(f"{k}:{100*v/tot:.0f}%" for k, v in sorted(byspl[s].items())))


if __name__ == "__main__":
    main()
