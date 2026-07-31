# Notes index — CDWM Drop world model (working repo)

Two coherent overviews carry the story; the rest are detailed working records. Shipped to GitHub: `drop/docs/00_overview.md` (EN) + `drop/docs/01_overview_zh.md` (ZH) — the same content as here.

## Theme notes (clean handoff set — formerly `docs/`, merged in 2026-07-31)
- **`00_overview.md`** — CD-WM overview & shared foundations (index, shared encoder, honest-eval protocol, paths).
- **`01_rigid_gripper_contraction.md`** — object tilt under gripper closing (DiT diffusion head, H=32).
- **`02_slip_contact_dynamics_wm.md`** — slip/drop contact-dynamics WM (DiT rollout, featured) + coupling validations.

  (The `restructure-common-grasp-drop` ship branch carries the canonical `grasp/docs/` copy of these three.)

## Overviews (start here)
- **`overview_zh.md`** — ★ single Chinese overview: changes vs the HF v1 dataset (88 objects, near-boundary + hidden CoM, diffusion WM + K=8, CoACD), design diagrams, results, 88-object table. (= the pushed `01_overview_zh.md`.) The English overview is the pushed `drop/docs/00_overview.md`.

## Drop — detailed experiment records (English)
- **`drop_phase_log.md`** — dataset setup + task analysis.
- **`drop_baseline_results.md`** — baseline endpoint WM.
- **`gateb_results.md`** — Gate B distributional WM (distribution ≫ point, hidden CoM causal, calibration).
- **`cross_object_transfer_plan.md`** — geometry-grounded CoM latent transfers to unseen objects.
- **`drop_rollout.md`** — trajectory supervision (traj_k8) + corpus diffusion rollout (roll_corpus) + demos.
- **`coacd_and_com_plan.md`** — the two extensions (CoACD e2e + CoM-from-observation): full results, robustness, Codex sign-off, the amortized-encoder null.
- **`drop_density_todo.md`** — object-density / CoG-diversity phase record (Gate A / A.5).
- **`gateb_related_work.md`** — related work / positioning.
- **`demo_videos.md`** — demo/animation inventory.

## Reference / cross-project
- **`RESULTS.md`** · **`REPORT_OUTLINE.md`** · **`RELEASE_MANIFEST.md`** — project-level scaffold.
- **`slip_phase_plan.md`** — Slip task plan (sibling task).

## `archive/`
Stale/superseded (fulfilled data requests, merged Chinese sources, WM#1 grasp-era history). Safe to delete.
