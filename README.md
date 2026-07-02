# CD-WM — Cross-Object Contact-Dynamics World Model

A diffusion **world model** that predicts how a grasped object **settles and tilts in a parallel-jaw gripper** over a
short horizon, from a **colorless 3D-Gaussian-Splatting point cloud** of the object plus the grasp pose. Trained
cross-object (generalizes to unseen objects) on rigid lift-success grasps simulated in MuJoCo.

The hard, useful signal is the **tilt**: closing off an object's center of mass rotates it in the jaws. The model is
graded on the *direction and magnitude* of that rotation.

```
python infer_example.py        # loads a checkpoint + bundled example data → predicts tilt vs ground truth
```

---

## What's here

```
my_config.py            paths (data dir via env CDWM_DATA)          wm/            model + rotation/metric utils
my_dataset.py           colorless dataset (the WM input pipeline)      common.py    quats(wxyz)/6D/geodesic/axis
my_dataset_frame.py     +FRAME variant: cloud in the gripper frame     dit.py       DiTWM (PointNet enc + DiT diffusion head)
train.py                trainer (DDPM ε-pred; optional geodesic aux)   metrics.py   EVAL_STANDARD: stratified dir/mag err
infer_example.py        standalone inference demo (GPU or CPU)         model.py     (legacy bowl-WM head)
eval/                   base-vs-arm per-band eval + matched floor    checkpoints/   base700_s{0,1,2}.pt + target_norm_stats
viz/                    direction-annotated video / analysis         example_data/  mini dataset (2 objects, 6 grasps)
```

## Install
```
pip install -r requirements.txt      # torch + numpy are all inference needs
```

## The model input — colorless 12-feature gaussians
Each object is a set of 3D gaussians; each gaussian is encoded by **12 geometry features** (no color):

`mu(3)` center · `rot_quat(4, wxyz)` orientation · `scale(3)` semi-axes · `opacity(1)` · `is_completed(1)` (1 = added by
shape-completion). Color/spherical-harmonics are intentionally dropped — the WM is geometry-driven. Features are
per-channel standardized (`feat_stats.npz`) and 4096 gaussians are drawn **uniformly at random** per forward pass. A
PointNet (max+mean) pools them into an object embedding; the grasp pose (`base_rel`), commanded closing, and table plane
are concatenated as context; a DiT diffusion head predicts the per-step gripper-frame SE(3) delta (the "target").

**`+FRAME` variant** (`my_dataset_frame.py`): re-expresses `mu`/`rot_quat` in the gripper frame at t=0 (an experiment in
contact-frame conditioning); everything else identical.

## Data
The full dataset (171 objects, 15,013 rigid grasps — meshes, 3DGS reconstructions, colorless point clouds, and grasp
targets) is on Hugging Face: **`BWangCN/cdwm-grasp-dataset`** (access-controlled ahead of publication). The `objects/<id>/point_cloud.npz`
there are exactly the colorless clouds this code consumes (place them as `clusters/gaussians_<id>_clean.npz`). The
training scripts additionally read the raw settle episodes (for the commanded-closing signal); `example_data/` shows the
expected layout: `clusters/`, `episodes_aug/`, `chunk_index/chunks_corrected.csv`.

Point a run at data with `CDWM_DATA=/path/to/dataset`.

## Train
```
python train.py --seed 0 --tag base --epochs 700 --ckpt_every 40 --dataset_mod my_dataset --outroot wm_runs
#   --dataset_mod my_dataset_frame   # the +FRAME (gripper-frame cloud) variant
#   --lam_aux 0.05                   # optional direction-targeted geodesic auxiliary loss
```
Select checkpoints by **held-out direction error**, not denoising loss (train ≥700 ep). Checkpoints carry
`H,T,dim,depth,n_feat` for reconstruction.

## Evaluate
`eval/eval_arm.py` selects each seed's checkpoint by held-out dir_err then scores per-band + per-shape direction/magnitude
error (`wm/metrics.py`, stratified by tilt band); `eval/compare_arms.py` scores an arm vs the baseline against the
**matched per-chunk floor** (`eval/floor_summary.json`) and the pre-registered criteria. Paths at the top of these
scripts point to training-output dirs — set them to your `wm_runs/`. See `eval/README.md`.

## Visualize
`viz/analyze_direction.py` plots direction-error distributions (standalone). `viz/render_direction_video.py` /
`select_and_rollout.py` render 3DGS+MuJoCo rollouts with GT-vs-WM tilt arrows and **depend on our simulation harness**
(not included) — provided as reference. See `viz/README.md`.

## Checkpoints
`checkpoints/base700_s{0,1,2}.pt` — the validated 700-epoch cross-object colorless WM, 3 seeds (DiT `dim=256, depth=4`,
`H=32`, `T=1000`, DDPM ε-prediction; ~4.9 M params, ~19 MB each). `target_norm_stats.npz` un-z-scores the model output to
physical units. Report metrics as mean ± std over the 3 seeds.

## Conventions
Meters, radians. Quaternions **(w,x,y,z)**. 6D rotation = first two **columns** of R (Zhou et al.), inverted by
Gram-Schmidt (`wm/common.py`). Target = cumulative gripper-frame SE(3) delta from the first settle frame; see the dataset
card on Hugging Face for the exhaustive coordinate spec.

---
*Research code, shared ahead of publication. Some eval/viz scripts reference our internal run layout — adapt paths as noted.*
