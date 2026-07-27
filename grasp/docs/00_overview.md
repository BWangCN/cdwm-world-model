# CD-WM — Overview & Shared Foundations

Index + shared pieces for the theme notes. Paths are repo-relative (`cdwm-world-model/`).
Code → GitHub; model weights / `*.npz` artifacts → HuggingFace; dataset already on HF.

> **Note (paths)**: file paths in these notes are current **local** placeholders — they will be swapped for the
> GitHub / HuggingFace URLs once the repos are populated (last release step). See `notes/RELEASE_MANIFEST.md` for the
> exact ship-list and the swap rules.

## Theme notes
- [01 — Object interactions under gripper contraction (rigid tilt)](01_rigid_gripper_contraction.md)
- [02 — Contact-dynamics world model (slip/drop/hold): DiT rollout (featured) + coupling validations](02_slip_contact_dynamics_wm.md)

> **Canonical head = DiT diffusion.** Across every regime the featured world model is the **DiT diffusion rollout** (the
> colleague's design; grasp `local_geo` **3.46°**, slip/drop rollout wins as a distribution). The MDN summary-space head in
> [02](02_slip_contact_dynamics_wm.md) is an **exploratory alternative** (predicts a 15-d motion *summary*, not the raw
> per-step trajectory the diffusion WM denoises); it is retained for the record but **de-emphasized** and not part of the
> shipped workflow. See `notes/alignment_diffusion.md`.

Each theme note is self-contained: **design → experiments & results (with CIs) → figures → model & data**.

> **`docs/` vs `notes/`**: `docs/` is the clean handoff set (these theme notes). `notes/` holds the internal dev logs
> (`progress_summary_zh`, `slip_phase_plan`, `RELEASE_MANIFEST`, working `RESULTS.md`).

> **Unifying thread**: both regimes use the *same* DiT diffusion rollout to imagine the object's future motion from the
> grasp — [01](01_rigid_gripper_contraction.md) rolls out the longer **H=32** gripper-*closing* tilt (one reliable future),
> while [02](02_slip_contact_dynamics_wm.md)'s slip model rolls out a short **K=10** lift-*onset* window as a **distribution**
> of possible outcomes (hold / slip / drop).

## Task
- From a **colorless 3DGS gaussian cloud** (12-feat, at grasp t=0) + a **parallel-jaw grasp pose**, predict **contact
  dynamics** — how the object moves/tilts/slips/drops as the gripper closes and lifts. (Not grasp *synthesis* — grasp is given.)

## Shared `local` encoder (reused by every model via `.encode()`)
- File: `wm/dit_local.py` (`DiTWMLocal.encode`). Output = 256-d conditioning embedding.
- Pieces: **gripper-frame cloud** (point features expressed in the contact frame) + per-gaussian **covariance**
  (Σ = R·diag(s²)·Rᵀ) + **contact-weighted pool** (soft-weight by distance to the pinch line).
- Design principle: same encoder across regimes; only the **output head** changes by task. **The canonical head is the
  DiT diffusion rollout** (grasp + slip/drop); the classifier is an auxiliary risk head and the MDN is an exploratory,
  de-emphasized summary-space alternative.

## Honest-evaluation protocol (applied throughout)
- **Object-disjoint** split (family-merged, near-dup-guarded): `make_outcome_split.py`.
- **Single-shot N=1** (deployable), not oracle best-of-N.
- **Calibration** (Brier / ECE / NLL) + **object-level bootstrap 95% CIs** + **paired per-object deltas** (`bootstrap_ci.py`).
- **Baselines**: no-motion, pose-only, per-mode median, random.

## Data
- Root: `/misc/kcgscratch1/MengyeGroup/yy5259/datasets/cdwm-grasp-dataset/` (HF: `BWangCN/cdwm-grasp-dataset`).
  - Rigid v1: `grasps.npz` — 171 objects / 15,013 successful grasps (reconstructed losslessly from HF).
  - Slip v2: `outcomes_v2/` — 562 objects / 53,917 episodes, full close→lift→outcome SE(3) trajectories + outcome labels.

## Environment / compute
- conda env `graspgen` (python 3.10, torch 2.1). GPU: V100 via SLURM, `--qos=batch`.
- Launchers in `slurm/`; all training/eval run as sbatch jobs.
