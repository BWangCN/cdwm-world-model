# CDWM — Contact-Dynamics World Models

World models for how a rigid object moves under contact, across three regimes that share **one point-cloud encoder** and differ only in the prediction head. The unifying thesis: geometry determines **one reliable future when the physics is deterministic, and a calibrated distribution when a hidden latent controls the outcome**.

| Sub-project | Task | Baseline | Our model |
|---|---|---|---|
| **`grasp/`** | object tilt as a parallel-jaw gripper closes (SE(3)) | single-shot N1 = 12.3° geodesic error, *loses* to no-motion (9.6°) | contact-frame encoder + geodesic loss: **N1 = 3.46°**, beats no-motion, at the ~2–5° label-jitter floor |
| **`grasp/` (slip)** | grasp-outcome / near-term contact dynamics after lift | single-mean over-disperses; loses to no-motion single-shot | **calibrated distribution** (MDN / short-window rollout): energy score 3.45 vs 5.71, wins on every non-rigid mode; geometry sharpens it |
| **`drop/`** | resting pose / basin after a release | point predictor 2.38° *beats* no-motion (7.45°); corpus is largely deterministic | **distributional WM** for near-boundary releases where a hidden center-of-mass makes the outcome multimodal: distribution >> point (well-calibrated, ECE 0.015 vs 0.227), the hidden CoM is causal, and the latent **transfers to unseen objects when geometry-grounded** |

## Layout
```
common/   shared core: DiTWMLocal point-cloud encoder (dit_local), DiT diffusion head (dit),
          model, utils, metrics, my_config, paths.py (all data roots resolve here)
grasp/    rigid grasp + slip: datasets, DiT-rollout / MDN heads, eval, figures, docs, models/
drop/     resting-pose WM: datasets, drop_net / drop_diffusion heads, Gate B (distributional),
          cross-object transfer, physical-density realism, demos, gateb/ data, docs, models/
```
Every module imports as `common.*`, `grasp.*`, or `drop.*`; run from the repo root (e.g. `python -m drop.train_gateb --arm grounded_oracle`). All external + repo-relative paths are centralized in [`common/paths.py`](common/paths.py) (override via `CDWM_*` env vars).

## Setup
```bash
pip install -r requirements.txt      # torch, numpy, scipy, trimesh, mujoco==2.3.7, coacd, imageio, ...
```
Datasets (HuggingFace): `BWangCN/cdwm-grasp-dataset` (grasp point clouds + meshes, reused by drop) and `BWangCN/cdwm-drop-corpus` (drop trajectories + the compact `gateb/` near-boundary split). Released checkpoints are under each `*/models/` (Git LFS).

## Docs
- Grasp + slip: [`grasp/docs/`](grasp/docs) (`00_overview`, `01_rigid_gripper_contraction`, `02_slip_contact_dynamics_wm`)
- Drop: [`drop/docs/00_overview.md`](drop/docs/00_overview.md)
