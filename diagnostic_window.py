"""Pick the rollout window empirically: for one shard, drift of object-in-gripper from the seated (hold) reference at K
steps past LIFT ONSET (first phase==3), by outcome. Does divergence show within ~5 steps? how much of FINAL is realized?"""
import os, sys, csv, collections, numpy as np
sys.path.insert(0, ".")
from wm.common import geodesic_deg
from compute_traj_full import rel_traj
OV = os.environ.get("CDWM_OV", "/misc/kcgscratch1/MengyeGroup/yy5259/datasets/cdwm-grasp-dataset/outcomes_v2")
rows = list(csv.DictReader(open(f"{OV}/outcomes_index.csv")))
byshard = collections.defaultdict(list)
for i, r in enumerate(rows): byshard[r["shard"]].append(i)
shard = sorted(byshard)[0]; ii = byshard[shard]
zf = np.load(f"{OV}/{shard}"); A = {k: zf[k] for k in ("obj_pos", "obj_quat", "base_pos", "base_quat", "phase")}
offs = [1, 2, 3, 5, 8, 10]
acc = collections.defaultdict(lambda: collections.defaultdict(list)); finald = collections.defaultdict(list); nlift = collections.defaultdict(int)
for i in ii:
    r = rows[i]; s = int(r["frame_start"]); e = s + int(r["n_frames"]); oc = r["v2_outcome"]
    ph = A["phase"][s:e]
    R_rel, _ = rel_traj(A["obj_pos"][s:e].astype(float), A["obj_quat"][s:e].astype(float), A["base_pos"][s:e].astype(float), A["base_quat"][s:e].astype(float))
    hold = np.where(ph == 2)[0]; lift = np.where(ph == 3)[0]
    if not len(hold) or not len(lift): continue
    ref = R_rel[hold].mean(0); U, _, Vt = np.linalg.svd(ref); ref = U @ Vt
    o0 = lift[0]; drift = geodesic_deg(R_rel, np.broadcast_to(ref, R_rel.shape)); nlift[oc] += 1
    for k in offs:
        if o0 + k < len(drift): acc[oc][k].append(drift[o0 + k])
    finald[oc].append(drift[-1])
print(f"shard={shard}  (frames dt=0.032s; k5 ~ 0.16s)")
print(f"median drift° from seated at K steps past lift-onset, by outcome:\n{'outcome':22s}" + "".join(f"{'k'+str(k):>7s}" for k in offs) + f"{'FINAL':>8s}{'n':>7s}")
for oc in ["RIGID", "TRANSIENT_SLIP", "PERSISTENT_SLIP", "LIFTED_DROPPED", "CLOSED_NEVER_LIFTED"]:
    print(f"{oc:22s}" + "".join(f"{np.median(acc[oc][k]):7.1f}" if acc[oc][k] else f"{'-':>7s}" for k in offs) + f"{np.median(finald[oc]):8.1f}{nlift[oc]:7d}")
