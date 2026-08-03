# Gripper-state head on the grasp diffusion WM — 2026-08-03

**Session goal:** Extend the existing grasp diffusion world model with a predictor for the gripper's
*achieved* state, using the colleague's regenerated `gripper_v3` dataset, while keeping every other part
of the WM identical. Confirm the addition does not degrade the trajectory or slip-gate behaviour.

---

## 1. Summary of what worked

- **`gripper_v3` validated and adopted as the single canonical dataset.** Tier A (`tier_a_outcomes_v2_aligned`,
  53,917 episodes, 100 % label-invariance vs the published `v2_index`) is self-contained: it carries the
  object/base pose channels **and** the real gripper channels — `driver_rad` (right_driver_joint, 0–0.8 rad =
  the *achieved* closure). The old `ctrl` channel was a dead, identical-in-every-episode command ramp; the
  achieved closure is the signal (final-frame std 0.245 across episodes vs 0.0 for `ctrl`).
- **The settle target was rebuilt purely from v3** (never joining v3 gripper channels onto v2 poses). At the
  same `ks` frame indices already used for the H=32 pose target, `driver_rad` and `phase` are sliced into
  `grip_target (n,H)` and `grip_phase (n,H)`. Round-trip residual median 0.0000.
- **One private gripper head added, mirroring the boolean branch**, and one third Kendall task. Everything
  else — shared `DiTWMLocal` trunk, the trajectory DiT diffusion (rigid-masked, eps + geodesic aux), the
  boolean slip branch, from-scratch training, bs/epochs/lr/scheduler — is unchanged.
- **Result: a clean additive change.** Trajectory point-accuracy and the slip gate held within noise; the new
  head learned real object-specific closure. Full training + the rigorous eval suite + a rollout animation are
  done.

---

## 2. What changed (minimal, additive)

| File | Change |
|---|---|
| `compute_settle_target.py` | rebuild `settle_target.npz` from v3 tier A; add `grip_target (n,H)` + `grip_phase (n,H)` sliced at the same `ks` as the pose target |
| `my_dataset_grasp_mtl.py` | load `grip_target`, normalize `/0.8` → `[0,1]`, return `grip_tgt (H,)` + `grip_phase (H,)`; `closing` stays zeros (the driver angle is a **target, never an input** → no leakage) |
| `wm/grasp_mtl.py` | add private `grip_branch = mlp([D,D,D/2])` + `grip_head = Linear(D/2,H)`; extend `log_s` 2→3 Kendall tasks; `predict_grip(cond)→(B,H)` |
| `train_grasp_mtl.py` | grip loss = `smooth_l1(predict_grip(cond), grip_tgt)` over H as the 3rd Kendall term on **all** episodes; eval reports MAE + phase-sliced metrics |

**Drop modules were left in place.** The grasp WM already imports zero drop code, so the "remove drop
diffusion" requirement is met without deleting the separate, completed drop-phase / Gate-B subsystem
(≈20 files depend on it, its own frozen paper). Nothing to remove.

### Which episodes each head trains on
Per the v3 README, Tier A aligns with `outcomes_v2` = **all** outcomes (rigid + slip + clearance), not a
rigid-only slice.

| Head | Trained on | Why |
|---|---|---|
| Trajectory (SE(3) DiT diffusion) | **rigid only** (masked) | slip episodes have no meaningful settle trajectory |
| Boolean slip classifier | **all** | its job is to separate rigid vs slip |
| **Grip head (new)** | **all** | achieved closure is a real signal in every episode; masking it would discard valid supervision |

Training log corroborates: `train 35152 (rigid 0.26)` — 26 % rigid, ~74 % slip, all loaded.

---

## 3. Results

Full training (job 1108260, from scratch, 200 epochs, ~1h37m on a v100), best checkpoint **ep140**.

### 3.1 Base WM preserved (rigorous eval suite, S=8, test split)
New grip-head/v3 run vs the preserved 2-head/v2 baseline (`eval_suite_baseline_v2_jul31.json`):

| Metric | Baseline (2-head, v2) | New (grip head, v3) | Δ |
|---|---|---|---|
| rot ADE n1 / bestK (°) | 2.606 / 1.739 | **2.428 / 1.648** | better |
| rot FDE n1 / bestK (°) | 3.158 / 1.851 | **3.075 / 1.772** | better |
| trans ADE n1 (cm) | 0.195 | **0.175** | better |
| ADD n1 / bestK (cm) | 0.805 / 0.363 | 0.835 / 0.363 | ~tied |
| miss_rate_rot@10° | 0.0829 | **0.0753** | better |
| coverage_ADD | 0.901 | 0.903 | tied |
| calib_spread_err_corr | 0.613 | 0.519 | slightly worse |
| **gate AUROC** | 0.8923 | **0.8879** | ~tied (−0.004) |
| gate slip_recall | 0.918 | 0.911 | ~tied |
| ood_rate vs p99 | 0.0203 | 0.0058 | better |

**Read:** adding the grip head did **not** degrade the base WM. Trajectory point-accuracy is slightly better
and the slip gate is within noise. Only calibration-spread correlation dipped (0.61→0.52), mild.
**Honest caveat:** not perfectly identical ground truth — v3 relabels marginally (test n_rigid 1887 vs 1967),
so sub-0.2° trajectory deltas are within data-shift + run noise. The defensible claim is **"no degradation,"
not "improvement."**

### 3.2 The new grip head (full val, 6,623 episodes)
Lead with MAE, not exact-1%-bin accuracy (Codex: 1 % bins are punishing on a near-constant plateau).

| | MAE (rad) | ≈deg | within 0.05 rad | within 0.10 rad |
|---|---|---|---|---|
| **Overall** | **0.065** (8.1 % of 0–0.8 range) | 3.7° | 62.5 % | 78.4 % |
| Close ramp (phase 1) | 0.044 | 2.5° | 73.0 % | — |
| **Hold plateau (phase 2)** | 0.094 | 5.4° | 47.2 % | — |

**Notable finding:** the settled-**hold** value is *harder* to predict than the close ramp (2× the error),
the opposite of the naive guess. The close ramp is partly deterministic (follows the command before contact),
whereas the hold plateau is the **object-specific achieved closure** — exactly the gripper-measured object
width that `gripper_v3` was generated to expose. So the model nails the easy ramp and carries a real ~5.4°
residual on the genuine object-width signal.

---

## 4. Rollout animation

**MuJoCo replay** (real Robotiq 2F-85 + real textured object mesh), rendered from **this** checkpoint via
`render_grasp_mtl_roll.py`. The gripper's 4-bar linkage is solved by physics and set kinematically at the
commanded `driver_rad`, so open/close is faithful; the object is the actual mesh posed by the settle
trajectory. **Left = ground truth** (object at the GT settle pose, gripper at the GT achieved `driver_rad`);
**right = model** (object at the *predicted* pose, gripper at the *predicted* closure). The script auto-selects
a well-predicted rigid held-out episode so the panels are directly comparable.

Three well-predicted held-out examples (auto-selected, distinct objects). Within each clip **left = ground
truth**, **right = model**; the two panels are near-indistinguishable because the model reproduces both the
settle pose and the achieved closure.

| object | endpoint pose err | closure GT → pred |
|---|---|---|
| ![Inositol bottle grasp GT vs model](../figures/rollout_grasp_mtl.gif) | 0.67° | 0.273 → 0.274 rad |
| ![Crosley alarm-clock grasp GT vs model](../figures/rollout_grasp_mtl_2.gif) | 1.40° | 0.378 → 0.447 rad |
| ![Argan-oil box grasp GT vs model](../figures/rollout_grasp_mtl_3.gif) | 1.01° | 0.456 → 0.443 rad |

*Rendered by `render_grasp_mtl_roll.py` → `figures/rollout_grasp_mtl{,_2,_3}.{mp4,gif}`. Full-res MP4s:
[1](../figures/rollout_grasp_mtl.mp4) · [2](../figures/rollout_grasp_mtl_2.mp4) · [3](../figures/rollout_grasp_mtl_3.mp4).*

---

## 5. Codex review (checkpoint confirms)

Two rounds, `gpt-5.5`, read-only.

- **Code review (pre-train):** faithfully additive; no gripper-target leakage into the encoder input; settle
  frame alignment correct (pose target and grip slice share `ks`); grip-on-all-episodes correct (rigid-masking
  it would discard valid supervision). Flagged: report phase-sliced **MAE** (done), lead with MAE over
  exact-bin accuracy (done). A raised "CPU-mask indexes CUDA cond" concern was a non-issue — the live GPU run
  executed that path and printed valid `traj_geo`.
- **Result confirm (post-eval):** agreed with **"no degradation, useful grip signal learned"**; would not
  claim trajectory *improvement* given the v2→v3 GT shift. The calibration dip is "mild, not a blocker"
  (bootstrap-CI it only if sample-ranking/abstention matters downstream). Confirmed `acc100` is a poor lead
  metric; use tolerance accuracy (within 0.05 / 0.1 rad) — adopted above.

---

## 6. Artifacts

| Output | Path |
|---|---|
| Trained checkpoint (grip head) | `gated_runs/grasp_mtl/best.pt` (ep140) |
| Rebuilt v3 settle target (+ grip) | `.../gripper_v3/tier_a_outcomes_v2_aligned/settle_target.npz` |
| Eval suite (new) | `gated_runs/grasp_mtl/eval_suite.json` |
| Eval suite baseline (preserved) | `gated_runs/grasp_mtl/eval_suite_baseline_v2_jul31.json` |
| Grip metrics | `gated_runs/grasp_mtl/grip_eval.json` |
| Rollout animation | `figures/rollout_grasp_mtl.{mp4,gif}` |

---

## 7. Next steps / TODOs

### Short-term
- [x] Validate v3, rebuild settle target, add grip head, train, eval, render, dev-log.
- [ ] (Optional) bootstrap-CI the calibration-spread dip if sample-ranking / abstention becomes relevant.

### Medium-term
- [ ] (Optional) per-object breakdown of hold-phase grip MAE vs object width — does the residual correlate
      with grasp width / geometry?
