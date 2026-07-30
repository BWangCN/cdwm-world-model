"""Aggregate + QA the dry pi0.5 corpus: per-episode verdicts + cross-episode schema/distribution/loadability
checks (Codex's dry-corpus asks). Reads figures/vla_records/episode_*/{episode.npz,meta.json}."""
import os, glob, json, numpy as np

REC = "figures/vla_records"
eps = sorted(glob.glob(f"{REC}/episode_*"))
print(f"=== dry corpus: {len(eps)} episodes ===\n")

rows = []; schemas = set(); action_dims = set(); posture = []; tilts = []
for e in eps:
    mp = os.path.join(e, "meta.json"); dp = os.path.join(e, "episode.npz")
    m = json.load(open(mp))
    if not os.path.exists(dp):
        print(f"[{os.path.basename(e)}] REJECTED ({m['validator']['hard']} hard) — provenance stub only"); rows.append((m, None)); continue
    d = np.load(dp, allow_pickle=True)
    lens = {k: d[k].shape[0] for k in d.files}
    aligned = len(set(lens.values())) == 1
    schemas.add(tuple(sorted(d.files))); action_dims.add(json.dumps(m["action_dim"], sort_keys=True))
    posture.append(d["observation.state"][:, :7]); tilts.append(m.get("grasp_net_tilt"))
    rows.append((m, {"frames": m["num_frames"], "aligned": aligned, "keys": len(d.files)}))
    v = m["validator"]
    print(f"[{m['object_id']:>22s}] {v['verdict']:6s} (h{v['hard']} w{v['warn']}) | {m['num_frames']}f | "
          f"grasp={m['grasp_source']}#{m['grasp_id']} net_tilt={m.get('grasp_net_tilt')} | aligned={aligned}")
    print(f"    task: \"{m['task']}\" | phases: {[s['phase'] for s in m['phases']]}")

# ---- cross-episode QA ----
print("\n=== cross-episode QA ===")
print(f"schema identical across episodes: {len(schemas) == 1}  ({len(schemas)} distinct key-sets)")
print(f"action_dim identical: {len(action_dims) == 1}  -> {list(action_dims)[0] if action_dims else '-'}")
verds = [m['validator']['verdict'] for m, _ in rows]
print(f"verdicts: PASS={verds.count('PASS')} REJECT={verds.count('REJECT')}")
tv = [t for t in tilts if t is not None]
if tv: print(f"grasp net_tilt across corpus: min {min(tv):.1f} med {np.median(tv):.1f} max {max(tv):.1f} deg")
if posture:
    P = np.concatenate(posture, 0)
    rng = (P.max(0) - P.min(0))
    print(f"arm-posture coverage (per-joint range, rad): {np.round(rng,2)}")
    print(f"  -> mean joint range {rng.mean():.2f} rad (higher = more diverse postures; watch multi-seed-IK bias)")
passed = sum(v == 'PASS' for v in verds)
print(f"\nCORPUS: {passed}/{len(rows)} episodes PASS -> loadable pi0.5 records with aligned schema.")
