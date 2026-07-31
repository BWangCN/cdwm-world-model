---
license: mit
library_name: pytorch
tags: [robotics, grasping, world-model, 3d-gaussian-splatting, contact-dynamics]
---

# CD-WM — Contact-Dynamics Grasp World Model

From a colorless 3D-Gaussian-Splatting cloud + a parallel-jaw grasp, predict how the grasped object behaves as the gripper
closes and lifts: its **tilt** (rigid regime) and its **slip/drop outcome distribution** (slip regime). The checkpoints are
organized to mirror the code docs — **01 rigid gripper** then **02 slip contact** (coupling validations under slip).

- **Code + docs**: https://github.com/BWangCN/Evolving_Environment/tree/main/cdwm-world-model
- **Dataset**: https://huggingface.co/datasets/BWangCN/cdwm-grasp-dataset

## Layout

```
models/
├── 01_rigid_gripper/               object tilt as the gripper closes (DiT diffusion head)
│   ├── local_geo.pt                  best — 3.46° single-shot (local encoder + DiT + geodesic loss)
│   └── base_hf.pt                    object-frame baseline
├── 02_slip_contact/                slip/drop world model (MDN) + coupling validations
│   ├── classifier_full.pt            5-way + 2-tier outcome classifier (P(fail) grasp ranker)
│   ├── mdn_full.pt                   ★ MAIN — mixture-density world model
│   ├── mdn_pose.pt                   pose-only control (geometry-vs-pose)
│   ├── traj_full.pt                  labeled mode-mixture baseline (the MDN beats it)
│   └── validations/
│       ├── joint_full.pt             joint MDN + boolean (clean-negative control)
│       ├── roll_full.pt              short-window rollout world model
│       └── roll_pose.pt              pose-only control
└── norm_stats/                     z-score statistics — REQUIRED for inference
    ├── summ_stats.npz                15-d trajectory-summary stats
    ├── trajfull_stats.npz
    ├── trajfull_stats_short.npz
    └── feat_stats_ov_local.npz       cloud-feature stats
```

## Usage
Every model reuses the shared `local` encoder; load a checkpoint with the matching class from the code repo
(`wm/dit_local.py`, `wm/classifier.py`, `wm/trajnet.py`, `wm/dit_rollout.py`) and the `norm_stats/` z-scores.
Architectures, training configs, and full results are in the code docs (`notes/01`, `notes/02`).
