# CD-WM — Release Manifest (GitHub + HuggingFace staging plan)

What ships where. Import closure resolved (the clean set won't break). **Prep only** — the actual commit is gated on
collaborator access (HF write for `xavhl` + GitHub push) from BWangCN. Path swap (local→URL) is the last step (see bottom).

## GitHub — clean code (37 files: 30 documented + 6 closure-only + `wm/__init__.py`)
**Keep:**
- models: `wm/{dit,dit_local,dit_rollout,trajnet,classifier,common,model,metrics}.py` + `wm/__init__.py`
- train: `train.py`, `train_{cls,mdn,traj,joint,rollout}.py`
- data/precompute: `my_config.py`, `my_dataset_{hf,hf_local,outcomes,traj,trajfull}.py`, `compute_traj_{summaries,full}.py`, `make_outcome_split.py`
- eval/figures/viz: `eval_{hf,traj,mdn,mdn_risk,mdn_robust,joint,rollout}.py`, `compare_cls.py`, `grasp_ranking.py`, `bootstrap_ci.py`, `make_figures.py`, `fig5_rollout.py`, `render_rollouts.py`, `diagnostic_window.py`
- `figures/` — fig1–5 (.png) **+ rollout_{rigid,slip,drop}.mp4** (representative rollout clips for the PR viz)
- **⚠ closure-only** (imported but NOT named in docs — dropping any breaks the code): `eval_traj.py`, `my_config.py`, `my_dataset_hf.py`, `wm/common.py`, `wm/metrics.py`, `wm/model.py`, `wm/__init__.py`
- **add**: `README.md` (→ points at `docs/`), `requirements.txt` (exists), `docs/`, `slurm/` launchers (optional)

**Drop or archive (13 — exploratory / ablation-only):**
- exploratory/legacy: `analyze_physics.py`, `analyze_selfobtain.py`, `infer_example.py`, `validate_hf.py`, `my_dataset.py`
- a′ negative experiment: `train_direct.py`, `eval_direct.py`
- rigid-ladder ablation loaders: `my_dataset_{frame,hf_frame,hf_grip,hf_lr,hf_objtgt,hf_pose}.py`
- → if you want the **fig4 rigid ladder** or the **a′-fails** reproducible from scratch, move these to `ablations/` instead of deleting.
- Do NOT rewrite the kept scripts — they're already terse; clean = remove clutter + add README, not re-author.

## HuggingFace — MODEL release ONLY (dataset `outcomes_v2` already on HF)
HF = dataset/model release; ship only what fits that (rest is regenerable via the GitHub scripts).
**Weights → HF *model* repo — ~9 headline `best.pt` (~135 MB)** (back every reported number):
- rigid: `wm_runs/colorless_s0/local_geo` (best 3.46°), `.../base_hf` (honest baseline)
- slip: `cls_runs/full`, `traj_runs/mdn_full` (MAIN), `traj_runs/mdn_pose` + `traj_runs/traj_full` (baselines)
- validations: `traj_runs/joint_full`, `roll_runs/roll_full`, `roll_runs/roll_pose`
- **+ tiny norm stats** bundled (needed to run inference): `summ_stats`, `trajfull_stats(_short)`, `feat_stats_ov_local`.
- *(exclude the 176 periodic epoch snapshots = 4.1 GB; full 36-model set = 539 MB only if every ablation must reproduce)*

**NOT on HF — derived/eval artifacts (regenerable via scripts):**
- `traj_full.npz`, `traj_summaries.npz`, `t0_poses.npz` — derived training targets → `compute_traj_{full,summaries}.py`, precompute in the loaders.
- prediction dumps (`test_pred.npz`, `roll_runs/*/metrics.npz`) — eval outputs → rerun eval on the checkpoints (final numbers already in `docs/03`).

**To GitHub instead:**
- `outcome_split.csv` (51 KB) — split definition (config, not data).
- **representative rollout MP4s** rendered from `traj_full.npz` (one each: rigid / transient-slip / drop) as PR visualization.

## Deferred — path swap (do LAST, after repos exist)
Docs currently use **local machine paths**. One find/replace pass over `docs/*.md` on release:
- code `wm/dit.py` → `<github-url>/blob/main/wm/dit.py`
- weights `traj_runs/mdn_full/best.pt` → `<hf-url>/blob/main/...`
- dataset `/misc/.../cdwm-grasp-dataset/` → `BWangCN/cdwm-grasp-dataset` (HF)
