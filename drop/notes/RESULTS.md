# CD-WM — Consolidated Results

Two regimes studied on the same objects + the same reusable encoder, with one honest-evaluation discipline throughout.

**Shared methods.** Input = colorless 12-feature 3DGS gaussian cloud + parallel-jaw grasp pose (MuJoCo sim data).
Encoder = **`local`**: cloud expressed in the **gripper frame** + per-gaussian **covariance** + a **contact-weighted pool**.
Evaluation = **object-disjoint** test (family-merged, near-duplicate-guarded), **single-shot / calibrated**, **object-level
bootstrap CIs**, and explicit **baselines** (no-motion, pose-only, per-mode median, random). All compute via SLURM.

---

## Part 1 — Rigid regime: predict the tilt of a successful grasp (v1)

- Reconstructed the full 15,013-grasp training set from HuggingFace **losslessly** (recomputed targets match to 1e-16).
- **Honest eval exposed the real baseline:** at single-shot (N=1) the shipped model's geodesic error (12.3°) is **worse than
  a no-motion predictor (9.6°)** — its reported "~18–21°" skill was **oracle best-of-8** + test-set checkpoint selection.
- **The fix is gripper-frame conditioning: 12.3° → 4.0° single-shot (~3×)**, attributed cleanly:

| config | N1 geo° | what it adds |
|---|---|---|
| base (object frame, global pool) | 12.26 | — (loses to no-motion 9.57) |
| **+frame (gripper frame)** | **4.49** | **≈94% of the gain** (= colleague's `+frame` idea) |
| +covariance | 4.13 | −0.36 |
| +contact-pool (= `local`) | 4.02 | −0.11 |
| **+geodesic loss (`local_geo`, best)** | **3.46** | colleague's aux loss |

- Mechanism test: object-frame *target* stays at ~12° → it's the **contact frame specifically**, not generic alignment.
  Gripper-as-pointcloud & L/R finger pooling: marginal/redundant. **We are at the ~2–5° label-jitter floor. FROZEN.**
- **Contribution = evaluation methodology + clean attribution** (not a new architecture; the winning idea is the colleague's).
- Physics note: sim is uniform-density-from-hull (deterministic, no hidden θ); a naive "torque-about-CoM" mechanism does NOT
  explain the tilt (contact-equilibrium, not global mass-torque).

---

## Part 2 — Slip/outcome regime: the contact-dynamics world model (v2 `outcomes_v2`)

Data: **53,917 episodes / 562 objects**, full `close→lift→outcome` SE(3) trajectories + outcome labels
{RIGID, TRANSIENT_SLIP, PERSISTENT_SLIP, LIFTED_DROPPED, CLOSED_NEVER_LIFTED}.

**Identifiability (data-only):** outcome is predictable from t=0 (nearest-grasp agreement 0.81) but with a **~13% aleatoric
boundary** (near-identical grasps flip outcome) — the colleague's card confirms marginal grasps flip class.

**Phase 1 — outcome prediction.** Geometry predicts the **failure MODE** (5-way macro-F1 **+0.20**, DROP-AUPRC **+0.12** vs
pose-only, held-out) but **not coarse failure** (settle-vs-fail is grasp-pose-driven). *Geometry → mode; pose → coarse.*

**Phase 2 — mode-conditioned world model.** Target = object-in-**moving-gripper** motion summaries.

| settle-rot° (median) | mode-conditioned WM | pose | per-mode median |
|---|---|---|---|
| oracle mode | **14.2** | 15.1 | 124.5 |

Crushes the median (14° vs 124°) → learns real mode-conditioned dynamics; geometry helps rotation/drift; **degrades
gracefully to deployment** (predicted mode: 15.3°, still crushes median). *Deployment bottleneck = mode prediction (aleatoric).*

**(a') mode-agnostic single-mean regression fails** — a single mean over a multimodal target (fair rerun: 12.97° point
estimate is fine, but no coverage of the multimodality).

**(c) Grasp ranking = deployment payoff.** Rank held-out grasps by the classifier's `P(fail)`:

| ranker | reject-30% KEPT-fail | REJECT precision | RC-AUC↓ |
|---|---|---|---|
| **classifier P(fail)** | **0.46** (from 0.61) | **0.95** | 0.488 |
| continuous-severity / pose | ~0.57 | ~0.70 | ~0.58 |
| random | 0.61 | 0.60 | 0.61 |

**MDN unification (the main slip result).** One **mixture-density world model**, mode-agnostic, **no threshold labels**:

| MDN capability | result |
|---|---|
| calibrated distribution (NLL, physical) | −3.07 (cloud) vs −1.17 (pose) |
| **top-K oracle coverage (settle-rot°)** | **8.35** — beats the *label-supervised* mode-conditioned mixture (15.26) |
| unsupervised mode discovery | emergent components align with physics (comp = never-lift 0.95 … transient-slip 0.30 fail-rate); self-selects ~5 modes |
| grasp-risk ranking (MDN-derived P(fail)) | RC-AUC **0.496 ≈ classifier 0.488** (matches a dedicated classifier) |

### Colleague-suggestion validations (2026-07-12)

Two designs the colleague raised, tested as pre-registered controls; both land as expected.

**(i) Joint MDN + boolean head (one model, Kendall-weighted).** Does *joint* success supervision beat our *separate* MDN +
separate classifier? **No.** The boolean only *ties* the dedicated classifier (risk-AUC 0.491 vs 0.488; reject-30% precision
0.940 vs 0.947), and the MDN *degrades* (test NLL −1.40 vs −3.07) — Kendall up-weighted the MDN (≈276×) yet it overfit
(train −25 → test −1.40). **The separate calibrated classifier + separate MDN decomposition is strictly better.**

**(ii) Short-window rollout WM ("visualize the future").** A DiT denoises the **K=10-step object-in-gripper trajectory from
lift-onset** (native time, target = motion relative to onset) over **all outcomes**, + a read-off boolean. Window chosen from
data (drift separates outcomes by ~k5; the transient/persistent split is the irreducible late aleatoric boundary).

| metric (held-out) | result |
|---|---|
| single-shot (N=1) per-frame drift | **worse than no-motion** (k10 7.98° vs 6.20°) — not a better point predictor |
| **energy score** (proper scoring rule, ↓) | **model 3.45 vs no-motion 5.71** — wins as a *distribution*, on every non-rigid mode |
| geometry effect | full-cloud ES 3.45 **< pose-only 4.75** (every mode) → geometry sharpens the distribution |
| boolean read-off (Brier/ECE/AUROC) | 0.193 / 0.147 / 0.831 — **does not beat** the classifier (0.162/0.078/0.846) |
| sample dispersion vs risk | corr +0.26 (positively associated) |

*Same thesis as the summary MDN: geometry → a calibrated **distribution** over near-term contact dynamics, not a point*
(best-of-N/energy-score win; single-shot does not). It is a **dynamics + visualization WM, not a superior risk head**;
RIGID over-dispersion is a known limit. **Robustness:** MDN main result stable across K∈{8,12} and seeds; emergent modes recur.

### Uncertainty (object-level bootstrap 95% CI, B=1000) — `bootstrap_ci.py`

Headline numbers with object-resampled CIs, and **paired per-object deltas** (same episodes — stronger than aggregate means).
Every claim, including the three honest negatives, is significant (CI does not cross the null).

| quantity | point | 95% CI |
|---|---|---|
| MDN top-K settle-rot° | 8.35 | [7.19, 9.84] |
| labeled-mixture settle-rot° | 15.26 | [13.24, 17.92] |
| classifier AUROC (fail) | 0.846 | [0.797, 0.887] |
| classifier accuracy | 0.768 | [0.733, 0.805] |
| rollout energy score (ALL) | 3.46 | [2.84, 4.08] |
| no-motion energy score | 5.71 | [4.28, 6.37] |
| rollout read-off AUROC | 0.831 | [0.786, 0.875] |

| paired per-object delta (negative = first better) | mean | 95% CI | reading |
|---|---|---|---|
| MDN top-K − labeled-mixture (settle°) | **−12.23** | [−14.11, −10.50] | MDN significantly beats label-supervised |
| rollout − no-motion energy (ALL) | **−1.91** | [−2.53, −1.31] | rollout wins as a distribution |
| ⤷ DROP / NEVER_LIFT | −2.80 / −3.63 | [−3.71,−2.00] / [−4.05,−3.13] | significant on failure modes |
| ⤷ RIGID | **+0.79** | [+0.57, +1.15] | *worse* on rigid (over-disperses) — honest |
| single-shot k10 − no-motion | **+2.91** | [+2.19, +3.69] | single-shot *worse* than no-motion — honest |
| full − pose energy | **−1.08** | [−1.26, −0.90] | geometry significantly sharpens |
| read-off − classifier Brier | **+0.031** | [+0.014, +0.049] | read-off *worse* risk head — honest |

Tight CIs → 3-seed reruns unnecessary (no conclusion is threatened). See **fig 5** (`figures/fig5_rollout.png`):
horizon curve (coverage beats no-motion; single-shot does not) + per-outcome energy delta with CIs.

---

## Headline contributions

1. **Deployment-aligned evaluation methodology** that overturned an inflated baseline (rigid) and set honest ceilings (both).
2. **Rigid:** gripper-frame *contact-frame* conditioning is the 3× lever; the regime is at its ~2–5° aleatoric floor.
3. **Slip (the world model):** *geometry determines a calibrated **distribution** over contact outcomes* — one unsupervised
   MDN covers the true multimodal outcome better than a label-supervised model, **discovers the physical failure modes** as
   emergent components, and yields a **deployable grasp-risk signal matching a dedicated classifier**.

## Honest ceilings / open
- Rigid: ~2–5° grasp-jitter floor. Slip: ~13% aleatoric mode boundary; coarse failure is pose-driven.
- Optional next: full frame-by-frame SE(3) rollout (visualization), grasp-perturbation study (measure the aleatoric floor
  directly), the pending pre-lift-failure data (NEVER_HELD/TOPPLED).
