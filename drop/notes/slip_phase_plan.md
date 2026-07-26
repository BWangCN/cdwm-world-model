# Slip-phase plan (outcomes_v2) — converged design (Claude + Codex, 2026-07)

Data: `datasets/cdwm-grasp-dataset/outcomes_v2/` — 53,917 episodes / 562 objects (100% clouds). 5-way outcome
{RIGID 12584, TRANSIENT_SLIP 12884, PERSISTENT_SLIP 10665, CLOSED_NEVER_LIFTED 15092, LIFTED_DROPPED 2669};
2-tier SUCCESS 25468 / FAILURE 28449. Full close→lift→outcome world-frame SE(3) trajectories (124–328 frames;
phase 1=close 2=hold 3=lift 4=settle; per-frame clear_mm/held). Continuous fields: converged_drift_deg,
post_lift_min_clear_mm, lift_h_mm, held_end. Sim deterministic (uniform-density-from-hull).

## Sequencing: A → joint → B
- **A. Outcome prediction from t=0 (FIRST).** Reuse the `local` encoder (gripper-frame + covariance + contact-pool),
  swap the SE(3) head for an outcome head. = grasp-success task + identifiability probe. Do NOT skip to B.
- **Joint (2nd).** encoder → calibrated outcome/mode distribution → outcome-conditioned trajectory decoder.
- **B alone (last).** Trajectory WM first hides failure behind multimodal loss; needs A's predictability ceiling first.

## First experiment — outcome prediction
Heads: 2-tier (SUCCESS/FAILURE) + 5-way (esp. DROP / persistent-slip recall), **calibrated** (temperature/Dirichlet).
Controls: **pose-only baseline** (grasp pose + lift params, no cloud); cloud-only/no-grasp (object-prior leakage);
train-only object-ID memorization sanity.
Metrics: balanced-acc, macro-F1, per-class recall, **AUROC/AUPRC for FAILURE & DROP, Brier/NLL/ECE, reliability curves**,
confusion matrix. Rare DROP: class-balanced sampling/loss but **evaluate on natural prevalence**.
Splits: object-disjoint (mandatory) + near-duplicate guard (cups/vessels) + family/source-disjoint stress split (hold out
whole object_sets focused_187/573/B_vessels) + vessel-heldout.
Aux (after discrete works): continuous heads converged_drift_deg / post_lift_min_clear_mm / held_end as AUXILIARY
supervision (not "regress-then-threshold"; learn calibrated monotone maps if boundaries needed).

Decision rule (object-disjoint bootstrap CIs): if cloud+grasp does NOT clearly beat pose-only and accuracy saturates low →
observation under-informative → task = **calibrated risk P(outcome)**, not hard prediction. Early red flags: train high /
test poor = leakage; both low but calibrated = missing state; pose-only ≈ cloud+grasp = geometry adds little; DROP AUPRC ≈
base rate = rare drop unlearnable.

## Identifiability (deterministic sim ≠ deterministic labels-from-observation)
Our input omits contact microstate; v1 ±2.5mm→2–5° sensitivity → outcome may be partly aleatoric-from-t=0.
Cheap DATA-ONLY probes (run before training): (1) nearest-neighbor outcome entropy in t=0 feature space;
(2) near-identical (object, grasp-pose) clusters → label entropy + continuous-target variance; (3) continuous-boundary
view: calibrate P(failure) vs converged_drift_deg / post_lift_min_clear_mm / held_end / lift_h_mm.

## Trajectory WM (later) — NOT unconditional whole-rollout diffusion
Structure: encoder → calibrated outcome/mode distribution → **mode-conditioned** trajectory model. Predict outcome probs +
final/settle SE(3) + drift/slip summary + drop-time/held timeline (+ optional full rollout). Variable length: normalize by
**PHASE not raw frame**; phase-conditioned dynamics; length masks. Contact modes: let them **emerge** (use phase/held/clearance
as observed signals/aux, don't hand-label with tuned thresholds). Diffusion justified ONLY after A shows one-to-many from t=0.

## Codex's top missing piece (later study)
Grasp-perturbation study: re-simulate tiny grasp-center neighborhoods (deterministic sim) → measures the irreducible
boundary thickness. Needs MuJoCo re-sim; not the first move. Also: top-k "avoid bad grasps" risk utility may matter more
than exact 5-way accuracy.

## PHASE-1 RESULT (outcome prediction, held-out 137 objects / 12,133 episodes)
| metric | full | pose-only | Δ bootstrap CI |
|---|---|---|---|
| FAILURE AUPRC (primary) | 0.882 | 0.870 | +0.012 [-0.05,+0.06] ✗ |
| 2-tier Brier↓ | 0.162 | 0.174 | +0.012 [-0.002,+0.023] borderline |
| 2-tier bal-acc | 0.765 | 0.726 | +0.040 [+0.016,+0.066] ✓ |
| 5-way bal-acc | 0.557 | 0.455 | large |
| 5-way macro-F1 | 0.541 | 0.412 | large |
| DROP AUPRC | 0.260 | 0.144 | ~2× (base 0.06) |

**Verdict (honest):** learnable; cloud helps the FINE outcome (5-way, DROP, +bal-acc — object-disjoint so real shape
generalization) but NOT the coarse FAILURE-AUPRC/Brier (CI includes 0 -> Codex primary criterion NOT clearly met). Coarse
"will it fail" is grasp-pose-driven (dominant CLOSED_NEVER_LIFTED is pose-predictable); object geometry matters for
distinguishing failure MODES (slip/drop/rigid). Coarse near a pose+aleatoric ceiling (~0.77 held-out vs ~0.87 within-object).
Clean story: "geometry matters for failure MODE, not coarse failure detection." Supports Phase-2 (mode-conditioned dynamics
= where geometry pays off). Temp scaling needed (T2~3.9 full = overconfident from balanced sampler; ECE 0.076 after).

## SYNC w/ colleague (repos): GitHub unchanged (no slip code/plan -> ours to design). outcomes_v2 README ALIGNS: they
recommend the moving-gripper-frame outcome signal (= our conditioning), reuse v1 colorless-12 clouds, and acknowledge the
marginal-grasp class-flip = our ~13% aleatoric boundary. Pending future data: 16,954 pre-lift failures (NEVER_HELD/TOPPLED).

## GIVEN-IT-LIFTED mode test (Codex de-risk; test=8182 held-out lifted episodes; classes RIGID/TRANS/PERS/DROP)
| metric | lifted_full | lifted_pose | Δ CI |
|---|---|---|---|
| FAILURE AUPRC (binary) | 0.631 | 0.632 | +0.004 [-0.06,+0.07] ✗ |
| 2-tier Brier↓ | 0.222 | 0.242 | +0.020 [+0.002,+0.034] ✓ |
| 2-tier bal-acc | 0.662 | 0.610 | +0.052 [+0.029,+0.074] ✓ |
| 5-way bal-acc | 0.530 | 0.418 | large |
| 5-way macro-F1 | 0.517 | 0.319 | +0.20 |
| DROP AUPRC | 0.382 | 0.263 | +0.12 (base 0.09) |

**Verdict: conditional-go CONFIRMED.** Robust pattern (full & lifted): cloud does NOT win coarse binary (pose-driven) but
CLEARLY predicts failure MODE (5-way macroF1 +0.20, DROP-AUPRC +0.12, held-out). Geometry -> mode, pose -> coarse. Moderate
absolute ceiling (aleatoric + cross-object). PROCEED to Phase-2 = mode-conditioned trajectory WM (report per-mode+calibrated).

## PHASE-2 RESULT (trajectory summary WM; held-out objects, GT-mode expert = oracle; median error, lower=better)
Target = object-in-MOVING-gripper motion summaries; precompute validated (rigid≈const, drop diverges 117°/212mm/held0.38).
| metric | traj_full | traj_pose | per-mode median |
|---|---|---|---|
| settle rotation° | 14.2 | 15.1 | 124.5 |
| settle trans mm | 28.5 | 28.1 | 103.7 |
| max-drift° | 9.4 | 12.2 | 9.6 |
Bootstrap Δ (full better if +): settle-rot vs median +110* / vs pose +0.9*; max-drift vs pose +2.7*; settle-trans vs median +76* / vs pose -0.4(ns).
Per-mode settle-rot (full): RIGID 6.4, TRANS 17, PERS 15.8, DROP 76.6 (chaotic/aleatoric), NEVER_LIFT 14.9.

**Verdict: GO.** (1) Model CRUSHES the per-mode-median baseline (14° vs 124°) -> it predicts the SPECIFIC object-in-gripper
motion, not a mode-average = Codex "world model works" criterion met. (2) Geometry (full vs pose) significantly helps
ROTATION + drift, not translation (= geometry->rotation/tilt, pose->coarse/translation, consistent w/ whole project).
MIXTURE (deployment, predicted mode) DONE: settle-rot 15.3 (oracle 14.2), trans 31.1, maxdrift 10.7 -> degrades only ~8-14%
and STILL crushes median (15.3 vs 124.5). BOTH Codex GO conditions met (oracle beats baselines + mixture degrades gracefully).
KEY DEPLOYMENT INSIGHT: at predicted-mode, geometry-vs-pose edge on rotation WASHES OUT (mix 15.3 ~ pose 15.1) because MODE
PREDICTION is the bottleneck (capped by Phase-1 5-way ~0.53 + aleatoric). max-drift still favors geometry (10.7 vs 12.2).
=> deployment lever = better MODE prediction / mode-uncertainty, NOT a fancier decoder.
NEXT: (a) improve mode head (or joint mode+trajectory training) ; (b) full phase-normalized SE(3) rollout (mode-conditioned
per-mode experts) if we want frame-by-frame ; (c) grasp-ranking-by-P(fail) utility metric. Rigid phase frozen (local_geo 3.46).

## NEXT-STEP PLAN (Codex-confirmed, by impact) — 2026-07
1. **(a') MODE-AGNOSTIC DIRECT continuous summary prediction + uncertainty** [BUILDING] — remove the deployment bottleneck:
   predict the 15-d summary DIRECTLY (heteroscedastic mean+scale, NLL), derive mode/risk from it via the label rules.
   "Don't fight an aleatoric threshold-defined label as the main interface." Compare direct vs Phase-2 predicted-mode
   mixture / pose / median on held-out summary error + NLL/calibration. Decisive: direct >= mixture on summary error AND
   better calibrated risk.
2. **(c) Grasp-ranking utility** — rank by predicted expected SEVERITY (continuous), risk-coverage curve + top-k bad-grasp
   capture; baselines random/pose/mode-P(fail)/mixture. Headline = "predicted dynamics avoid high-severity grasps" (severity
   from continuous GT, not brittle class label).
3. (a) better mode prediction / joint training — now AUX/ablation (lower priority; deploy shouldn't route through 5-way).
4. (b) full frame-by-frame SE(3) rollout — completeness/viz only.

## (a') RESULT = NEGATIVE (confounded) + REPLAN (Codex) — 2026-07
Direct mode-agnostic regressor LOST to the Phase-2 predicted-mode mixture: settle-rot direct_full 19.2 / direct_pose 16.4 /
mixture 15.3; maxdrift direct 14.4 / mixture 10.7. Gaussian NLL direct_full=473(!) + cloud<pose -> the run was BROKEN
(heteroscedastic NLL + drop-over-weighting sampler -> single mean-head chasing multimodal/extreme targets diverged). Codex:
single-mean regression is FUNDAMENTALLY weak for a multimodal contact-outcome target -> mode-conditioning is CORRECT, stop
fighting it. Honest headline now = **mode-conditioned world model + mode classifier for grasp ranking**.
CLEAN robust finding: mode-classifier P(fail) is the BEST grasp-ranker (risk-coverage 0.61->0.47 at 30% reject; beats direct/pose/random).
REPLAN by impact: (c) calibrated mode-classifier GRASP RANKER [building] > improve/calibrate mode prediction > fair (a') rerun
(plain MSE/uniform/grad-clip, sanity only) > better continuous experts. (a') teaches: single-mean regression fails on
multimodal outcomes (report carefully, not as clean ablation).

## (c) GRASP-RANKING RESULT = STRONG POSITIVE (deployment payoff). n=12133 held-out, base fail 0.61.
Reject riskiest x% -> (KEPT fail-rate, REJECT precision), RC-AUC lower=better:
| ranker | @30% KEPT-fail | REJECT-prec | RC-AUC |
|---|---|---|---|
| **classifier P(fail)** | **0.46** | **0.95** | **0.488** |
| traj mode P(fail) | 0.47 | 0.94 | 0.488 |
| mixture severity | 0.57 | 0.71 | 0.572 |
| direct severity | 0.57 | 0.70 | 0.580 |
| GT max-drift (ceiling) | 0.54 | 0.77 | 0.554 |
| random | 0.61 | 0.60 | 0.610 |
**Headline: failure-mode classifier = strong deployable grasp-risk signal.** Reject 30% riskiest -> 95% truly fail,
kept-fail 61%->46%. Beats GT-max-drift ceiling (drift != failure). Continuous-severity rankers ~random (confirms (a'):
classifier not regressor is the risk signal). => Full slip arc: geometry->mode (P1) + mode-conditioned motion (P2, crushes
median) + mode-classifier ranks grasps (c). Mode-agnostic direct fails (a', multimodal).
NEXT (impact): improve/calibrate MODE prediction (drives P2 mixture AND the ranker) -> the one remaining lever. Then report.

## MDN PUSH RESULT = STRONG (the paper result). held-out objects.
NLL (physical, lower better): MDN_full -3.07 vs MDN_pose -1.17 (cloud sharpens the DISTRIBUTION).
settle-rot° median: MDN top-K-oracle 8.35 (BEST) | MDN mix-mean 13.18 | labeled-mixture 15.26 | single-mean(plain,fair) 12.97 | median 124.5.
=> ONE MDN component covers the true outcome to 8.35deg, beating the LABELED mode-conditioned mixture, WITH NO MODE LABELS.
EMERGENT components align with physical modes (comp6 NEVER_LIFT 81%, comp7 DROP 44%, slip clusters); ~5 of 8 alive = self-
selected ~5 modes. Grasp-ranking: classifier P(fail) still best (RC-AUC 0.488 vs MDN E[maxdrift] 0.575 -- crude severity).
(a') confound RESOLVED: fair plain single-mean = 12.97 (not broken 19.2); the heteroscedastic+drop-overweight run had diverged.
PAPER STORY: "geometry -> a calibrated DISTRIBUTION over contact outcomes; components EMERGE as physical failure modes
unsupervised" + deployable classifier ranker (95% reject-precision). NEXT PUSH: derive MDN risk properly (P(failure-region)
from the mixture, not E[maxdrift]) -> can the distributional WM match/beat the classifier ranker & unify the story?

## MDN RISK UNIFICATION = SUCCESS (the push landed). held-out grasp ranking:
MDN-derived P(fail) [sum_k w_k(test) * P(fail|comp_k), comp->fail from TRAIN labels only] MATCHES the dedicated classifier:
RC-AUC MDN 0.496 vs classifier 0.488 (tied, both >> random 0.61); reject30% MDN 92.5% / classifier 94.7% precision.
Per-comp TRAIN P(fail) 0.30..0.95 (comp6 never-lift 0.95, comp1 transient-slip 0.30) = components carry real failure info.
=> UNIFIED: ONE mode-agnostic MDN world model does (1) calibrated distribution over outcomes (NLL, top-K cover 8.35 beats
label-supervised 15.26), (2) unsupervised physical-mode discovery (emergent components), (3) grasp-risk ranking matching a
dedicated classifier -- WITHOUT ever training on discrete mode labels. PAPER-COMPLETE slip arc.

## FIGURES + ROBUSTNESS (Codex order) — 2026-07
Figures (figures/*.png, Okabe-Ito colorblind palette, eyeballed): fig1 emergent-mode alignment heatmap (components->physical
modes w/ P(fail) gradient 0.30..0.95 — the reviewer-proof structure figure), fig2 risk-coverage (MDN-P(fail) ON TOP of
classifier, both crush pose/random), fig3 distributional coverage (MDN top-K 8.35 beats label-supervised 15.26), fig4 rigid ladder.
Robustness batch (sbatch): MDN K-sweep {4,6,12} + seeds {1,2} -> eval_mdn_robust (NLL + top-K coverage per config + emergent-
mode stability across seeds). Point-vs-distribution control already have (single-mean 12.97 vs MDN top-K 8.35, same backbone).
TODO next (Codex reviewer controls): shuffled-geometry control (cloud shuffled across objects -> should degrade to pose);
then #3 perturbation study (aleatoric floor). Writeup deferred (colleague approval). CONSOLIDATED: notes/RESULTS.md.

---

## Joint success-head experiment (colleague's suggestion — validation control)

**Colleague's idea:** add a boolean success/fail prediction head *on top of* the original (DiT) head. Verified the
original head IS DiT (diffusion). But DiT is the *rigid* trajectory head, and v1 rigid data is success-only
(no negatives → a boolean head is degenerate there). So the faithful, cheap validation (per Codex) is on **v2**:

`local encoder → MDN summary head` **+** `local encoder → binary success/fail head`, **jointly** trained
(one model, Kendall-uncertainty-weighted loss so there are no hand-tuned coefficients). = `JointMDNOutcomeNet`.

**Decisive question:** does *joint* training beat our *separate* MDN + separate classifier? We already showed coarse
success/fail is pose-driven and MDN-derived P(fail) ≈ the dedicated classifier — so the expected result is
**clean-negative** ("the separate decomposition is sufficient"). Pose-only control (`joint_pose`) guards against the
boolean looking geometry-aware when it's just the pose-driven coarse signal.

- Model: `wm/trajnet.py::JointMDNOutcomeNet` + `joint_loss` (Kendall). Train `train_joint.py`; eval `eval_joint.py`
  (joint vs mdn_full vs cls_runs/full, same test order): NLL, top-K settle-rot, risk-cov AUC, reject-30%, Brier.
- Jobs (this session): **1104146** joint_full, **1104147** joint_pose, **1104148** eval (dep afterok both). V100/qos=batch.
- Smoke-tested forward+loss+backward (shapes ok, Kendall weights get grads).

---

## Full-rollout WM (colleague's "visualize the future" + boolean) — DiT rollout, all outcomes

**Colleague's picture:** the DiT head should *generate the future trajectory* (visualize) alongside the boolean success
flag. This is the deferred full frame-by-frame rollout, now JOINT with success. Verified & designed with Codex:

- **Train the DiT on FAILED trajectories too** (natural v2 distribution, NO balancing) — mandatory: a success-only DiT
  can only draw a stable hold, never a slip/drop. Diffusion handles the multimodality that sank the single-mean (a').
- **Target:** full `H=32 × 9` object-in-gripper `T_go(t)` (`compute_traj_full.py`), event-aligned to the
  close→lift→outcome window (phase≥2), normalized-time nearest-resample (keeps valid SO(3)). Per-dim z-scored (v1 style).
- **Loss:** DDPM ε-MSE over the full trajectory **+** boolean CE, Kendall-weighted. **Boolean is a READ-OFF** of the
  shared `cond` (encoded once), NOT fed back into the DiT → samples stay `p(traj | scene, action)` (clean generative story).
- **Primary = unconditional-on-outcome** (sample → hold/slip/drop diversity); mode-conditioned = optional later.
- **Model:** `wm/dit_rollout.py::DiTRollout` = `DiTWMLocal(depth=4)` (local encoder + the DiT denoiser now actually USED)
  + read-off head2. Reuses v1 DDPM/DDIM infra.
- **Eval (`eval_rollout.py`), Codex rigor bar:** (1) per-frame best-of-N reconstruction coverage by outcome vs no-motion;
  (2) boolean calibration (Brier/ECE/AUROC) vs separate classifier + joint MDN; (3) sample-diversity vs P(fail). Generative
  outcome-FREQUENCY calibration (NN-label sampled trajs) DEFERRED (labeled). Baselines: no-motion, pose-only DiT (`roll_pose`).
- **Honest framing (agreed):** if boolean isn't better-calibrated than the classifier, this is a full-rollout
  *visualization* WM, not a superior risk model; the scientific win requires temporal structure the 15-dim summary can't recover.
- Jobs (this session): **1104155** precompute → **1104156** roll_full / **1104157** roll_pose → **1104158** eval. V100/qos=batch.
- Smoke-tested forward + DDPM loss + backward + DDIM sample.

### REVISION (short window > full trajectory) — user instinct + data + Codex

Instead of the full close→lift→outcome trajectory, use a **SHORT native-time window: K=10 steps from LIFT-ONSET (~0.32s),
target = motion RELATIVE to the onset frame** (starts at identity; model predicts the DIVERGENCE, not the seated pose).
Rationale: geometry determines the ONSET; the long tail (already-slipping → chaotic final pose) is largely aleatoric and
would make the WM eat irreducible noise. Cleaner claim: "predict near-term contact dynamics from grasp geometry."

Data justification (`diagnostic_window.py`, one shard, median drift° from seated at K steps past lift-onset):

| outcome | k1 | k3 | k5 | k8 | k10 | FINAL |
|---|---|---|---|---|---|---|
| RIGID | 0.6 | 0.7 | 0.8 | 0.9 | 0.9 | 2.4 |
| TRANSIENT_SLIP | 1.7 | 2.7 | 3.6 | 4.7 | 5.4 | 18.0 |
| PERSISTENT_SLIP | 1.4 | 2.6 | 3.6 | 5.0 | 5.8 | 27.8 |
| LIFTED_DROPPED | 1.7 | 5.0 | 8.4 | 11.7 | 14.6 | 93.2 |

→ By **k5 the outcomes are ordered & gapped** (rigid flat 0.8 · slips 3.6 · **drop 8.4** — drop diverges earliest/clearest).
The transient-vs-persistent split is genuinely LATE (18 vs 28 only at FINAL) = the ~13% aleatoric boundary, NOT fixable by
the full trajectory. So short window keeps the geometry-determined signal, drops the aleatoric tail.

Codex convergence: short native window PRIMARY; anchor **lift-onset**; don't hardcode K by taste — predict K=10 and REPORT
the horizon curve (eval prints error/coverage at steps 3/5/8/10, operating point ~5-8 from data); keep full-32 absolute as
viz-secondary (`traj` key). `compute_traj_full.py` emits BOTH (`short` primary + `traj` viz). Boolean stays read-off; eval
adds the horizon curve + per-outcome model-vs-no-motion(stays-seated). Resubmitted: **1104160** precompute → **1104161**
roll_full / **1104162** roll_pose → **1104163** eval. (Superseded full-window jobs 1104155-58 cancelled.)

### RESULTS (2026-07-12) — joint, short-rollout, robustness

**Exp 1 — Joint MDN+boolean (jobs 1104146-48): CLEAN-NEGATIVE (as predicted).**
- MDN degrades under joint: NLL joint −1.40 vs separate −3.07; top-K 8.88 vs 8.35 (CE competes with distribution modeling).
- Boolean only MATCHES the dedicated classifier: risk-AUC 0.491 vs 0.488; reject-30% KEPT-fail 0.467 vs 0.464, precision 0.940 vs 0.947.
- Pose-only joint boolean worse (AUC 0.514, prec 0.872) → cloud adds a little to the boolean, still ≤ dedicated classifier.
- **Verdict: joint success supervision does NOT help; separate calibrated classifier + separate MDN is strictly better.** Validates the decomposition. (Colleague's boolean-head idea = tested as a control.)

**Exp 2 — Short-window rollout WM (K=10 from lift-onset, jobs 1104160-63): POSITIVE on dynamics, HONEST-NEGATIVE on risk.**
- Reconstruction best-of-N vs no-motion(stays-seated), median drift°: k3 0.82/1.40 · k5 1.60/3.03 · k8 2.44/5.08 · k10 2.82/6.20 → **beats no-motion ~2× at every horizon**, across all failure modes (RIGID trivially matched; slips/drop/never-lift all beaten).
- **Geometry helps the dynamics**: full-cloud < pose-only at every horizon (k10 2.82 vs 3.56; TRANS_SLIP 2.7 vs 3.2; NEVER_LIFT 4.8 vs 6.0; DROP marginal 7.2 vs 7.4) → geometry → near-term contact dynamics (per-frame, which the summary MDN couldn't show).
- **Aleatoric uncertainty captured**: corr(P(fail), end-z sample spread) +0.278 (low-risk 3.1mm vs high-risk 5.7mm).
- **Honest ceiling**: boolean read-off does NOT beat the classifier (Brier 0.193/ECE 0.147/AUROC 0.831 vs classifier 0.162/0.078/0.846; joint diffusion training hurt boolean calibration). → **near-term dynamics + visualization WM, NOT a superior risk head** (pre-registered verdict holds).

**Robustness (K-sweep + seeds, jobs 1104140-45): MDN main result is robust.** K=8 NLL −3.07 ≈ K=12 −3.15 (K4/K6 underfit NLL but coverage ok); seeds cov 7.96/8.28/8.35; emergent modes recur across seeds → not a lucky K/seed.

### EXP2 RIGOR CORRECTION (N=1 + energy score) — the honest, distributional claim

Codex flagged: best-of-N=8 = coverage (oracle), not accuracy. Added single-shot(N1) + energy score (proper scoring rule).
- **Single-shot N1 is WORSE than no-motion** at every outcome/horizon (overall k10 7.98° vs 6.20°; over-disperses, esp. RIGID
  0.7/1.3 vs 0.2/0.4). Best-of-N (0.81/1.59/2.42/2.84) was ORACLE coverage, not deployable. So NOT a better point predictor.
- **Energy score (z-space, lower=better): model BEATS no-motion as a DISTRIBUTION** — ALL 3.45 vs 5.71; wins on every
  non-rigid mode (TRANS 3.18/3.82, PERS 3.19/4.29, DROP 6.40/8.26, NEVER 4.91/7.90); only RIGID favors no-motion (0.53/0.24,
  model over-disperses the genuinely-constant case).
- **Geometry sharpens the distribution**: full ES 3.45 < pose ES 4.75, on every mode.
- Boolean read-off still worse than classifier (0.193/0.147/0.831 vs 0.162/0.078/0.846); spread~risk corr +0.26.
- **Honest claim = geometry → a calibrated DISTRIBUTION over near-term contact dynamics, not a point** (best-of-N/energy win;
  single-shot does not). Thematically identical to the summary MDN. NOT a better risk head; RIGID over-dispersion is a known limit.
- Kendall check (EXP1): MDN was UP-weighted (exp(5.62)≈276 vs boolean 2.6) yet test NLL −1.40 (train −25) → joint OVERFIT the
  MDN, worse generalization than separate −3.07. Statement: "joint training failed to preserve MDN test quality despite up-weighting it."

### Decision: feed boolean output into DiT? (回灌) — NO (design justification)

Considered training a variant that feeds the boolean head's PREDICTED P(fail) into the DiT denoiser (the coupling we
avoided). Decided AGAINST (me + Codex converged):
- `P(fail)=f(cond)` is a **deterministic readout of the same conditioning the DiT already has** → **zero independent info**;
  best case no change, worst case entangles the rollout with a noisy 1-D classifier signal and weakens the clean
  `p(traj|scene,action)` interpretation. P(fail) is also coarse/pose-driven → can't disambiguate hold/slip/drop.
- Only value is rhetorical (negative ablation to justify the read-off design). Not a scientific priority.
- **The meaningful "feed outcome to DiT" variant = TRUE-outcome(mode)-conditioned DiT** (not predicted P(fail)): adds info
  geometry can't determine (aleatoric branch); decisive metric `error(traj|true mode)` vs `error(traj|scene,action)` per
  mode → isolates mode-prediction bottleneck vs dynamics quality; enables controllable counterfactual rendering
  (drop-future vs hold-future). Framed as an ORACLE renderer, not a deployed predictor (no leakage). Cheap: 1 V100 job,
  same K=10, add outcome embedding to cond.
- **Ranking (Codex+me): (1) consolidate → colleague review; (2) if one run, the true-outcome-conditioned DiT; (3) skip P(fail)回灌.** Both variants DEFERRED unless colleague review blocks on them.

### CONSOLIDATION DONE (2026-07-17) — report-ready

Codex resume-consult green-lit documentation (no more training; CIs are tight). Done:
- **Object-level bootstrap 95% CIs + paired per-object deltas** (`bootstrap_ci.py`): all claims significant incl. the 3
  honest negatives (single-shot<no-motion +2.91; read-off worse Brier +0.031; RIGID over-disperses +0.79). MDN−labeled −12.23
  [−14.1,−10.5]; rollout−no-motion energy −1.91 [−2.5,−1.3]; full−pose −1.08 [−1.3,−0.9]. Tight → no seed reruns needed.
- **Fig 5** (`fig5_rollout.py` → `figures/fig5_rollout.png`): horizon curve + per-outcome energy delta with CIs.
- CIs folded into `notes/RESULTS.md` (Uncertainty subsection) + `notes/progress_summary_zh.md` (fig5 + CI bullets).
- Report outline: `notes/REPORT_OUTLINE.md`. Formal prose still deferred until colleague review.
