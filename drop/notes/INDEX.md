# Notes index — CDWM Drop world model

Grouped map of the working notes. English notes = method/experiment record (bullet-point, kept as-is). Chinese notes = colleague-facing. Archived = stale/superseded/grasp-era (moved to `archive/`).

## Drop — method & experiments (English, the record)
- **`drop_phase_log.md`** — dataset setup + task analysis (the corpus formulation, splits, baselines).
- **`drop_baseline_results.md`** — baseline endpoint WM results (H=1 resting-pose prediction).
- **`gateb_results.md`** — Gate B distributional WM: distribution ≫ point, hidden CoM causal, calibration.
- **`gateb_related_work.md`** — related work / positioning (literature survey).
- **`cross_object_transfer_plan.md`** — geometry-grounded CoM latent transfers to unseen objects (plan + results).
- **`drop_rollout.md`** — rollout WM: trajectory supervision (traj_k8) + corpus diffusion rollout (roll_corpus), demos.
- **`coacd_and_com_plan.md`** — the two extensions: CoACD end-to-end geometry + CoM-from-observation (plan + all results + robustness + Codex sign-off + the amortized-encoder null).
- **`drop_density_todo.md`** — object-density / CoG-diversity phase record (Gate A / A.5).
- **`demo_videos.md`** — demo/animation inventory.

## Drop — colleague docs (Chinese)
- **`colleague_summary_zh.md`** — ★ top-level concise summary: what changed vs the HF v1 dataset (88 objects, near-boundary + hidden CoM, diffusion WM + K=8, CoACD), results, the null. START HERE.
- **`drop_wm_explained_zh.md`** — design explainer with ASCII diagrams (trajectory / K=8 anchors / diffusion denoising / mesh-vs-cloud).
- **`gate_b_objects_zh.md`** — the 88-object table + the two-object-universe comparison + Gate A/A.5/B milestones.
- **`colleague_update_extensions.md`** — the two extensions in full (EN + ZH), with all numbers.

## Reference / cross-project
- **`RESULTS.md`** — consolidated results (project-level).
- **`REPORT_OUTLINE.md`** — paper/report scaffold.
- **`RELEASE_MANIFEST.md`** — GitHub + HuggingFace staging plan.
- **`slip_phase_plan.md`** — Slip task plan (sibling task, not drop).

## Archived (`archive/`, stale/superseded/grasp-era)
Fulfilled data requests, superseded Chinese progress updates, and WM#1 (grasp) history — kept for provenance, safe to delete.
