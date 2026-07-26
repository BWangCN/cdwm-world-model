# drop/exploratory — hidden-CoM regime (Gate B)

**Exploratory** work beyond the corpus task: our own **88-object dataset** with **near-boundary releases + a per-episode hidden center-of-mass**, to study when a *distributional* world model is necessary and whether the hidden CoM can be *inferred from interaction*. Not representative of the colleague's corpus physics. The primary (corpus) deliverable is [`../docs/00_overview.md`](../docs/00_overview.md).

## Docs
- **`hidden_com.md`** — full write-up: Gate B distributional WM, cross-object transfer, CoM-from-observation (Bayesian inference), physical/material-density realism, convex-hull scope, CoACD end-to-end, Gate-B rollout. Results, figures.
- **`hidden_com_zh.md`** — 中文说明: design diagrams (K=8 anchors, diffusion), what changed vs the v1 dataset, results, the 88-object table.
- **`figures/`** — Gate B figures (CoM-sweep counterfactuals, basins, scope boundary, rollout grid).

## Code + data (live in the main `drop/` tree, shared with the corpus pipeline)
- **Data:** `drop/gateb/*_gateb_s0.npz` (hidden-CoM episodes, 88 objects) · `drop/gateb/*_coacd_s0.npz` + `drop/gateb_coacd_clouds/` (CoACD end-to-end) · `drop/objlist_mesh.txt`.
- **Models:** `drop/models/gateb/*.pt` (point / no_latent / grounded_oracle / abstract / shuffled / coacd_* / diff_oracle).
- **Code:** `train_gateb` · `eval_gateb` · `eval_transfer` · `com_infer` · `hull_vs_coacd` · `aggregate_hvc` · `concavity_audit` · `multi_physical_density` · `gateb/generate_coacd` · `build_mesh_cloud` · `eval_coacd` · `train_roll` · `render_grid` · `render_drop_demo` · `render_basins` · `my_dataset_gateb` (CDWM_GATEB_SRC=coacd path). These stay in `drop/` for shared imports; they are exploratory, not part of the corpus deliverable.
