# CD-WM — Contact-Dynamics Grasp World Model

From a colorless 3DGS gaussian cloud + a parallel-jaw grasp, predict how the grasped object behaves as the gripper closes
and lifts — its **tilt** (rigid regime) and its **slip/drop outcome distribution** (slip regime).

**Start here → [`docs/00_overview.md`](docs/00_overview.md)** (shared encoder, honest-eval protocol, dataset), then:
- [`docs/01_rigid_gripper_contraction.md`](docs/01_rigid_gripper_contraction.md) — object tilt under gripper closing (DiT diffusion head).
- [`docs/02_slip_contact_dynamics_wm.md`](docs/02_slip_contact_dynamics_wm.md) — slip/drop world model (MDN) + coupling validations.

## Layout
- `wm/` — shared model package (local encoder + DiT / classifier / MDN / rollout heads).
- `train_*.py` · `eval_*.py` · `compute_traj_*.py` — training, evaluation, precompute.
- `make_figures.py` · `fig5_rollout.py` · `render_rollouts.py` — result figures + rollout clips.
- `docs/` — design, experiments & results, terminology, paths.  `figures/` — figures + representative rollout clips.
- `outcome_split.csv` — object-disjoint train/test split (also regenerable via `make_outcome_split.py`, seed-fixed).
- `models/` — released model checkpoints + norm stats (Git LFS); see [`models/README.md`](models/README.md).

## Data & weights
- Dataset: [BWangCN/cdwm-grasp-dataset](https://huggingface.co/datasets/BWangCN/cdwm-grasp-dataset) (Hugging Face).
- Model checkpoints: [`models/`](models/) — in this repo via **Git LFS**, organized `01_rigid_gripper/` → `02_slip_contact/` (+ `norm_stats/`).
- Derived targets (`traj_summaries.npz`, `traj_full.npz`, …) regenerate from the dataset via the precompute/train scripts.

## Environment
Python 3.10 + PyTorch 2.1 (`requirements.txt`).
