# `gateb` — near-boundary hidden-CoM drop split (compact)

A distributional-WM split for the drop corpus: near-stability-boundary releases of CoM-sensitive objects, each with a **hidden center of mass** that makes the resting outcome multimodal. It is distributed in a **compact recipe form** (13 MB) rather than as full trajectories, because the trajectories regenerate deterministically from the recipe.

## Files and schema
One `.npz` per object, `<object>_gateb_s0.npz`, with N episodes:

| key | shape | meaning |
|---|---|---|
| `object` | scalar str | object id (point cloud in `cdwm-grasp-dataset/outcomes_v2/point_clouds/<object>.npz`) |
| `release_quat` | (N, 4) | release orientation, wxyz |
| `drop_h` | (N,) | release height of the lowest vertex above the table (m) |
| `com` | (N, 2) | **hidden CoM latent** = `[principal-axis index in {0,1,2}, signed offset in metres]` (sampling target) |
| `com_sim` | (N, 2) | the CoM the **simulator actually used** = `com` with the offset quantized to 3 mm (use this to regenerate) |
| `rest_quat` | (N, 4) | resting orientation, wxyz |
| `basin` | (N,) | resting-basin id (nearest stable pose, yaw-invariant) |

88 CoM-sensitive objects (76 sensitive + 12 negative controls), ~2000 episodes each.

## Why compact, and the reproducibility guarantee
The recipe stores the **inputs** (release pose, drop height, hidden CoM) and the **outcome** (resting pose, basin). The full free-fall trajectory is a deterministic function of those inputs plus the object hull, so it is regenerated on demand rather than stored:

```bash
# needs the point clouds + MuJoCo 2.3.7 (pinned); re-simulates stored episodes and compares to the recorded outcome
python -m drop.gateb.verify_gateb --objs 011_banana 050_medium_clamp
```

Verified: re-simulating an episode from **`com_sim`** reproduces the recorded **basin 100%** (`verify_gateb` reports `basin_match_com_sim = 1.0`), resting-orientation error 0.16–1.55 degrees. The build model is `drop.density.v2.drop_sweep` (hull + explicit-inertial CoM offset).

**Two CoM fields, on purpose:** `com` is the continuously-sampled hidden-CoM latent (what the oracle conditions on); `com_sim` is that value quantized to 3 mm, which is what the simulator actually used (generation caches models by a 3 mm-rounded offset). **Regenerate from `com_sim`** for a bit-exact recipe. Re-simulating from the unquantized `com` matches the basin ~82–95% (a few near-boundary flips), reported as `basin_match_stored_com`.

## Requirements to regenerate
- `cdwm-grasp-dataset` point clouds (the collision hull the sim uses is the convex hull of the object point cloud, so the sim geometry equals what the world model sees),
- **MuJoCo 2.3.7** (pinned; the deterministic settle is version-specific),
- `trimesh` (stable poses / basin assignment).

## Relation to the main corpus
The main `cdwm-drop-corpus` splits (`train_corpus`, `v0_grid_eval`, `spread_eval_v2`, `tier_b`) hold full world-frame SE(3) trajectories under uniform density. `gateb` is a **distinct split**: near-boundary releases with a randomized hidden CoM, stored as the compact recipe above. Both describe the same simulator; `gateb` targets the regime where a distributional world model is necessary.
