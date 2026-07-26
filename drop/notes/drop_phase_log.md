# Drop-phase log — dataset setup + task analysis (2026-07-21)

Branch: `drop` (in `cdwm-world-model/`, repo `BWangCN/cdwm-world-model.git`).
Dataset: `BWangCN/cdwm-drop-corpus` → downloaded to
`/misc/kcgscratch1/MengyeGroup/yy5259/datasets/cdwm-drop-corpus/` (1.25 GB, 46 HF files).

---

## Status / index (updated 2026-07-21)

This file = the drop-phase index; §1–6 below are the original dataset + task analysis. Progress since:

- **Object density / center-of-gravity investigation — COMPLETE** (Gate A + A.5). Full report:
  [`../density/REPORT.md`](../density/REPORT.md); plan/gates: [`drop_density_todo.md`](drop_density_todo.md); colleague
  brief (中文): [`drop_density_plan_zh.md`](drop_density_plan_zh.md). Verdict: CoG diversity is a *targeted* contribution
  (CoM-sensitive shapes × near-boundary releases → multimodal → distributional WM), NOT worth adding to the corpus
  as-sampled (2.3% near-boundary). Off the baseline critical path → **Gate B** (T5–T7) when prioritized.
- **Baseline drop WM — COMPLETE.** Results: [`drop_baseline_results.md`](drop_baseline_results.md). Point-prediction
  reference (`train_drop.py`/`my_dataset_drop.py`/`wm/drop_net.py`/`eval_drop.py`). Held-out test: full/release **beats
  no-motion** (2.38° vs 7.45°, non-overlapping CIs; unlike the grasp baseline), basin-transition **AUROC 0.973**; the
  **cloud is essential** (pose-only ≈ chance) and the **release frame halves** the rotation error vs object-frame. The
  drop corpus is a largely well-posed deterministic mapping → point prediction is adequate on the natural corpus.
- **Gate B — CoG-aware *distributional* drop WM — DONE** (`gateb/`, `notes/gateb_results.md`). DiT diffusion head vs point
  vs oracle-latent on 10 CoM-sensitive objects, near-boundary + hidden-CoM. **CI-scoped verdict:** the distribution
  *significantly* beats the point predictor **and** a basin-frequency baseline on held-out release neighborhoods (+0.21/+0.25
  NLL, object-bootstrap CIs exclude 0, 9/10 objects) → in this regime a distribution is genuinely better than a point.
  Oracle-latent (the "CoM is causal" claim) + boundary-subset margins are **directional but not significant at 10 objects**;
  cross-object transfer **fails** (object-disjoint). Completes the CDWM drop thesis (natural=deterministic/point · boundary
  hidden-CoM=distribution necessary · cross-object=unsolved).
- **KEY LIMITATION = OBJECT COUNT.** Gate B used **10** objects; grasp used **171**. The underpowered oracle/boundary CIs and
  the cross-object failure are both attributable to too few objects. Pool available (from the 558 point-cloud objects, V2-ranked):
  **104 CoM-sensitive (tv_max≥0.5), 77 strong + few-well-separated-poses (best Gate B candidates), 175 at tv_max≥0.3.**
- **Next (recommended):** #1 SCALE Gate B to ~50–100+ CoM-sensitive objects (powers oracle-causality + boundary CIs + attempts
  cross-object transfer) · #2 physical per-hull density (vs the controlled explicit-inertial offset) · #3 rigor (reliability/ECE).
  Architecture (DiT head) is sound + on-theme (Codex-confirmed) — NOT the bottleneck.
- **Gate B SCALED to 88 objects + CROSS-OBJECT TRANSFER SOLVED — DONE** (`notes/gateb_results.md`, `notes/cross_object_transfer_plan.md`).
  At 88 objects the distribution>point result is significant/large/well-calibrated and hidden-CoM is demonstrably causal
  (model-free MI, oracle model, null on neg-controls). **Cross-object transfer now WORKS** with a geometry-grounded latent
  (per-point vector-to-CoM + distance): grounded > no-latent **+0.184 SIG**, grounded > abstract **+0.191 SIG**, grounded >
  shuffled-CoM **+0.160 SIG** (object-disjoint, 19 held-out objects, eval `1105406`). Supersedes the "cross-object transfer
  fails" note above (that was 10 objects + the abstract encoding). **GPU-consistency** verified: V100-vs-TITAN training effect
  +0.013 NLL, CI includes 0 (ns); conclusion identical across GPUs.
- **DEMO VIDEOS** ([`demo_videos.md`](demo_videos.md), `render_drop_demo.py`, job 1105467): same object + same near-boundary
  release, sweep the hidden CoM → the resting basin flips (banana 1→0→0→1, clamp 1→1→1→2, mouse 1→1→0→1). MuJoCo/EGL from
  the WM's own point-cloud hull. Illustrates why a distributional WM is needed; ready to hyperlink into the paper write-up.

---

## 1. What the drop corpus is

A **rigid-body drop / placement** world-model dataset: object released above a table → free-fall → impact → settle.
Sibling to `cdwm-grasp-dataset`; reuses the same per-object colorless-12-feat 3DGS cloud assets.

| config | episodes | role |
|---|---|---|
| `train_corpus/` | 160,200 | training (ALL validity classes, flags intact) |
| `v0_grid_eval/` | 990 | held-out-configuration eval (structured grid) |
| `tier_b/` | 30 | gripper finger-open release validation |
| `spread_eval_v2/` | 180 | multimodality calibration at near-boundary conditions |

Valid training episodes (SETTLED within 4 s): **153,996 / 160,200**.
Objects: **80 meshes**, 12 held-out for cross-object generalization (`manifest.json` `held_out` flag).

## 2. Per-episode schema (train_corpus)

Per-episode `.npz` (packed in `<config>/<prefix>_NNN.tar`, ~5000 episodes/shard):

| field | shape | meaning |
|---|---|---|
| `obj_pos` | (N, 3) | object COM position, world frame, 31.25 Hz |
| `obj_quat` | (N, 4) wxyz | object orientation, world frame |
| `t` | (N,) | timestamps |
| `release_quat` | (4,) wxyz | orientation at release |
| `drop_h` | scalar | clearance between object lowest point and table at release (mm) |
| `tilt_deg` | scalar | tilt angle from stable pose at release (U[0, 15°]) |
| `yaw_rad` | scalar | yaw about vertical axis (U[0, 2π)) |
| `xy` | (2,) | release XY in table frame (U disk r=5 cm) |
| `validity` | str | `SETTLED` / `NO_REST` / `ARTIFACT` / `SETTLED_OFF_SUPPORT` |
| `valid_training` | bool | True iff SETTLED within 4 s (the trainability gate) |
| `outcome_label` | str | `SUCCESS` / `FAILURE` (descriptive, does NOT gate training) |
| `outcome_subclass` | str | `TIPPED` / `ROLLED` / `NEVER_SETTLED` / … |
| `release_basin` | str | nearest stable pose at release |
| `rest_basin` | str | nearest stable pose at rest |
| `basin_transition` | bool | True iff object tipped to a different stable pose |
| `net_rot_from_release_deg` | scalar | total rotation from release to rest |
| `resting_pen_mm` | scalar | mesh penetration into table at rest (ARTIFACT gate: >2 mm) |
| `settle_t` | scalar | time to convergence (s) |
| `object`, `config`, `config_kind` | str | object/config identifiers |

Trajectory length: **12–40 frames typical** (SETTLED episodes). Physics: MuJoCo 3.9, `condim=6`, rolling μ = 1e-4,
zero-velocity kinematic release (NO gripper), 4 s ceiling.

## 3. Loading

Streams directly from tar — no extraction needed:

```python
import io, glob, tarfile, numpy as np

DROP = "/misc/kcgscratch1/MengyeGroup/yy5259/datasets/cdwm-drop-corpus"

def iter_episodes(root, config="train_corpus"):
    for tp in sorted(glob.glob(f"{root}/{config}/*.tar")):
        with tarfile.open(tp) as tf:
            for m in tf.getmembers():
                if m.name.endswith(".npz"):
                    yield m.name, np.load(io.BytesIO(tf.extractfile(m).read()), allow_pickle=True)
```

`load_example.py` in the dataset root does exactly this. Verified: `python load_example.py <root> train_corpus` →
160,200 episodes. Label integrity: `python verify_dataset.py <root> --n 200` → **200/200 OK**.

**Point clouds**: from `cdwm-grasp-dataset/outcomes_v2/point_clouds/{obj}.npz` — **79/80 objects present**.
Missing: `005_tomato_soup_can` (not in grasp dataset either; skip or handle separately).
Env var convention: `CDWM_DROP=/misc/kcgscratch1/MengyeGroup/yy5259/datasets/cdwm-drop-corpus`.

## 4. How this differs from the grasp WM

| | Grasp WM (rigid) | Grasp WM (slip) | Drop WM |
|---|---|---|---|
| Conditioning | cloud + gripper pose `base_rel` | cloud + gripper pose | cloud + release params (h, tilt, yaw, xy, quat) |
| Reference frame | **gripper frame** (the big win) | gripper frame | **release / object frame** (TBD) |
| Trajectory length | H=32 fixed (closing) | 124–328 frames | 12–40 frames (settling) |
| Multimodality | binary (hold vs slip) | 5-way outcome | **basin transitions** (different stable poses) |
| N train episodes | 15,013 (rigid) | 53,917 | **153,996 valid** |
| Target | (H, 9) SE(3)-delta during closing | 15-d summary / K=10 rollout | final (obj_pos, obj_quat) at rest |

Key structural difference: **no gripper** — the contact-weighted gripper-frame encoder (`build_feats`) does NOT apply.
The natural frame analogue is the **release frame** (rotate cloud into `release_quat`), but this is TBD.

Key challenge: **multimodality from basin transitions** — visually similar releases land in different stable poses
(e.g., a box tipping onto its side). Same cloud, same drop height, different rest basin. The spread_eval_v2 config
(180 near-boundary episodes) is purpose-built to calibrate this.

## 5. Task structure and proposed sequencing

By analogy with the grasp slip phase (outcome prediction → trajectory WM):

**A. Outcome classification (first).**
Given (cloud, release params), predict:
- `valid_training` (SETTLED vs NOT) — equivalent to grasp success classification
- `basin_transition` (tip vs stay) — the key multimodality signal
- `outcome_subclass` (TIPPED / ROLLED / NEVER_SETTLED) — fine-grained

Controls: pose-only (release_quat + drop_h, no cloud); object-prior (cloud, no release).
Splits: 12 held-out meshes from `manifest.json`.

**B. Resting-pose prediction (WM).**
Given (cloud, release params), predict final `(obj_pos, obj_quat)`:
- Base: regress mean resting pose → lower bound
- MDN / DiT: calibrated distribution (needed for basin-transition multimodality)
- Eval: resting position error (mm), resting orientation error (°), **energy score** (cf. slip MDN eval)
- Eval configs: `v0_grid_eval` (990 structured) + `spread_eval_v2` (180 near-boundary)

**Horizon candidates** (from README): `(dt_train=192 ms, H=6)` or `(dt_train=96 ms, H=12)` from 31.25 Hz raw.
Or predict FINAL pose directly (skip trajectory, regress to rest).

**Simplest first experiment**: predict `net_rot_from_release_deg` (pre-computed scalar) + `basin_transition` from
(cloud + release_quat + drop_h + tilt_deg). If this is learnable above an object-mean baseline, the cloud carries
signal. If not, the geometry fully determines outcome deterministically and signal comes from release params only.

## 6. Known boundaries (from dataset README)

- Non-YCB masses are a 400 kg/m³ proxy → elevated resting-pen tail in expansion (1.8% > 2 mm → ARTIFACT).
- 3 configs have NO_REST > 20% (roll-prone); 8 configs have settle-time p95 > 3.2 s (near 4 s ceiling).
- Ball-class excluded (roll-dominant, never settle cleanly).
- `005_tomato_soup_can` cloud missing from grasp dataset; skip in WM training.
- `tier_b` validates that finger-open release velocity is negligible (median 0.15 cm/s ≪ 6 cm/s gate).
