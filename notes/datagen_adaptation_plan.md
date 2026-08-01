# Plan — adapt `gs-native-datagen` with the grasp settle WM as the only engine

**Decided (2026-08-01):** VLA is archived (see `notes/vla_pipeline.md`). The live datagen
path is the colleague's `gs-native-datagen` (3DGS-native → LeRobot → Diffusion Policy).
Of our three WM components, **only the grasp in-gripper settle WM moves forward**:

- **grasp settle WM** — USED. `gs-native-datagen` loads the `colorless_s{0,1,2}/frame/best.pt`
  3-seed `DiTWM` ensemble (K DDIM draws) for the one spot kinematics can't cover: the grasped
  object settling under the jaws.
- **boolean slip gate** (`GraspMTL.head2`) — NOT used. Datagen filters with
  `wm_bridge/wm_infer.py::reject_reasons()` (physical plausibility + OOD), not the gate.
- **drop WM** (`roll_corpus`) — NOT used. Placement is scripted mplib FK.

## Quality status (why we trust the engine)

Confirmed at **both** levels:
- **Downstream (the real proof):** DP already trains successfully on data this WM generated
  (colleague's rationale for dropping VLA).
- **Offline** (`grasp_eval_suite.py`, job 1107717, on `grasp_mtl`): rot FDE 3.16° / 1.85°
  best-of-8, trans FDE ~2 mm, ADD-S ~6 mm, coverage 0.90, ood 0.02, residual-motion 0.014 cm.
- **Caveat:** offline numbers are on `grasp_mtl`, not the exact `colorless` ensemble the
  datagen loads → Phase 0 closes that gap. Downstream DP already covers the ensemble.

Not overfitting: splits are **object-disjoint** (`my_dataset_outcomes.py:90-91` asserts
train/val/test object sets disjoint), so strong TEST metrics are on unseen objects. Dataset =
**562 objects**, **53,917 episodes**, of which the trajectory WM supervises on **12,584 RIGID**
only (~4.9M-param model — too small to memorize that scale). Training loss = eps-prediction
MSE + small geodesic-aux (`train.py:6`); eval metrics are on the **sampled** trajectory, a
different and stricter quantity than the training loss.

## Phased plan

**Phase 0 — stand up & confirm the engine** (mostly ours; cheap)
- Create envs `gsdg-gen` (render) + `gsdg-build` (pack lerobot==0.1.0).
- Download ~1.2 GB `BWangCN/gs-native-datagen-assets` → **scratch** (confirm exact location first).
- Eval the ACTUAL `colorless` ensemble with `grasp_eval_suite.py` → attach numbers to the
  in-loop model (closes the `grasp_mtl` gap). Gate: settle metrics match/beat `grasp_mtl`.
- Smoke-gen a few L0/L1 episodes end-to-end; verify LeRobot records + `ms_view.mp4` renders.

**Phase 1 — harden the in-loop filter** (ours; pure upside, no retrain)
- Port group-D plausibility signals from `grasp_eval_suite.py` into `reject_reasons()`
  (convergence/residual-motion, contact-hold, 6D orthonormality, symmetry-aware).
- Measure slip prevalence on the datagen's AnyGrasp grasps × objects → decide if the gate is
  worth wiring in. If material: calibrate + threshold-sweep (NOT retrain). If rare: skip.

**Phase 2 — scale objects & scenes** (colleague's repo; coordinated)
- Expand past the 2 example objects. Reuse our 3DGS clouds where clusters exist; else run the
  upstream recon+segmentation for new scenes.
- Apply the **flat-object exclusion** (plates; see `notes/object_selection.md`) — grasp
  contacts sit too close to the tabletop for the jaws.
- Ensure a grasp bank per object.

**Phase 3 — scale volume + QA loop**
- Generate the target corpus (resumable, `--budget-h`).
- Run the plausibility gate on outputs; **label ~50 stratified once** to validate the
  evaluator; then scale on metrics + per-scene spot-checks.

**Phase 4 — downstream validation** (the gold standard)
- Train DP on the generated corpus; measure policy success — repeats the colleague's
  validation at our scale/objects and closes the loop.

**Phase 5 — conditional extensions** (future)
- Richer placement → wire in the **drop WM** (replace scripted FK). Multi-embodiment (UR) via
  the swappable arm layer. Slip gate iff Phase-1 measurement showed it material.

## Division of labor
Ours: WM engine, `grasp_eval_suite.py`, `reject_reasons()` hardening (Phases 0–1 unblock all).
Colleague's: pipeline repo, object list, scene recon, plate handling (Phase 2+).

## Reference — drop corpus size (answered 2026-08-01)
`BWangCN/cdwm-drop-corpus` v1.0: ~1.4 GB on disk; ~161k episodes (train_corpus 160,200;
153,996 valid SETTLED) over ~80 meshes / 108 configs. NOT consumed by datagen (FYI only).
