# CD-WM — Report Outline (documentation scaffold)

Report-ready skeleton mapping each claim → its evidence (table/figure/CI). Prose deferred until colleague review;
this is the structure to write into. Evidence lives in `RESULTS.md` (numbers), `figures/fig1-5`, `bootstrap_ci.py` (CIs).

**One-line thesis:** *From a colorless 3DGS cloud + a parallel-jaw grasp, geometry determines a calibrated **distribution**
over contact outcomes — not a single future — under a deployment-honest evaluation.*

---

## Abstract (1 para)
Honest single-shot evaluation overturns an inflated grasp-tilt baseline; a shared gripper-frame **contact encoder** gives a
3× rigid fix; and in the slip regime a mixture-density world model predicts a **calibrated distribution** over contact
outcomes, discovers physical failure modes unsupervised, and yields a deployable risk signal — while two coupling variants
(joint boolean; full-rollout risk head) are shown, with CIs, to be redundant / distributional-only.

## 1. Introduction & problem
- Task: predict contact dynamics (tilt / slip / drop) of a grasp, not *where* to grasp.
- Gap: grasp datasets/benchmarks (GraspNet, **GraspFactory** 109M static poses) keep only binary feasibility — they discard
  the trajectory/mode signal we model (→ §7). Deployment-honest eval is missing.
- Contributions bullet (mirror §9).

## 2. Setup
- Data: MuJoCo 3DGS, rigid v1 (171 obj / 15,013) + slip `outcomes_v2` (562 obj / 53,917 episodes, close→lift→outcome SE(3)).
- Shared **`local` encoder**: gripper-frame cloud + per-gaussian covariance + contact-weighted pool (`.encode()` reused).
- **Honest eval protocol**: object-disjoint split, single-shot N=1, calibration, **object-level bootstrap CIs**, baselines
  (no-motion, pose-only, per-mode median, random).

## 3. Rigid regime — the contact-frame lever  [Table Part-1 · fig4]
- Honest eval exposes baseline single-shot 12.3° < no-motion 9.6° ("18–21°" was oracle best-of-8 + test-set selection).
- Ladder → **local_geo 3.46°**; **+frame ≈ 94% of the gain** (colleague's idea); mechanism test = contact frame specifically.
- At ~2–5° jitter floor → FROZEN. Contribution = eval + attribution, not architecture.

## 4. Slip identifiability & Phase-1  [fig1]
- Outcome predictable from t=0 (nn-agreement 0.81) with ~13% aleatoric boundary.
- **Geometry → failure MODE** (5-way macro-F1 +0.20, DROP-AUPRC +0.12), NOT coarse fail (pose-driven).
- Classifier `P(fail)` = grasp ranker: reject-30% → 95% precision.

## 5. Slip Phase-2 — the mixture-density world model (MAIN)  [fig1,2,3 · CIs]
- Mode-conditioned WM crushes settle-median (14° vs 124°); single-mean (a') fails on the multimodal target.
- **MDN**: calibrated NLL (−3.07 cloud vs −1.17 pose); **top-K coverage 8.35 [7.19,9.84] beats label-supervised 15.26**
  (paired Δ −12.23 [−14.1,−10.5]); unsupervised modes align w/ physics; MDN-`P(fail)` ≈ classifier.
- Robust across K∈{8,12} and seeds; modes recur.

## 6. Validations of two coupling designs  [fig5 · CIs]
- **(i) Joint MDN+boolean** → clean-negative: boolean ties classifier, MDN degrades (NLL −1.40 vs −3.07). Separate is better.
- **(ii) Short-window rollout DiT + read-off boolean** (K=10 from lift-onset): wins as a **distribution** (energy −1.91
  [−2.5,−1.3]; geometry sharpens −1.08 [−1.3,−0.9]) but single-shot < no-motion (+2.91) and read-off ≯ classifier (Brier
  +0.031) → dynamics + visualization WM, not a superior risk head. Honest negatives all significant.
- Design notes (for methods/appendix): boolean is a **read-off** (grad co-trains encoder; P(fail) NOT fed into DiT);
  feed-P(fail)→DiT rejected (no-op); outcome-conditioned DiT = deferred optional.

## 7. Related work / positioning
- **GraspFactory & large-scale grasp-pose line** = grasp *synthesis* (109M static poses, binary labels, SE(3)-DiffusionFields).
  Different task; complementary/upstream (candidate-grasp source); their validation discards exactly the mode/trajectory
  signal we model. No overlap, no undermining.

## 8. Limitations / external validity
- ~13% aleatoric mode boundary; coarse fail is pose-driven; RIGID rollout over-disperses.
- Object scale (562 slip objects) vs GraspFactory's 14k–33k → external-validity caveat; path = generate more dynamics
  rollouts on more objects (P3 lever), not adopt static-pose benchmarks.
- Pending colleague pre-lift-failure data (NEVER_HELD/TOPPLED).

## 9. Contributions
1. Deployment-honest evaluation that overturned an inflated baseline (rigid) and set honest ceilings (both regimes).
2. Rigid: gripper-frame contact conditioning = 3× lever; at aleatoric floor.
3. Slip: geometry → a calibrated **distribution** over contact outcomes; unsupervised failure-mode discovery; deployable
   risk signal; two coupling designs characterized (negative / distributional-only) with CIs.

---

## Figure & table inventory
| id | file | shows | § |
|---|---|---|---|
| fig1 | fig1_emergent_modes.png | MDN components ↔ physical modes | 4,5 |
| fig2 | fig2_risk_coverage.png | risk-coverage: MDN ≈ classifier | 5 |
| fig3 | fig3_coverage.png | distributional coverage (top-K) | 5 |
| fig4 | fig4_rigid_ladder.png | rigid ladder (frame = 3×) | 3 |
| fig5 | fig5_rollout.png | rollout horizon curve + energy Δ CIs | 6 |
| T1 | RESULTS Part-1 | rigid attribution ladder | 3 |
| T2 | RESULTS Part-2 | MDN capabilities | 5 |
| T3 | RESULTS validations | joint + rollout | 6 |
| T4 | RESULTS Uncertainty | bootstrap CIs + paired deltas | 5,6 |

## Status
- **Done/frozen:** all experiments, figures, CIs. Report-ready.
- **Deferred:** formal prose (awaits colleague review); pre-lift-failure data; optional outcome-conditioned DiT.
- **Next:** colleague review → then write §-by-§ into this skeleton.
