# Baseline drop WM — results (2026-07-21)

The deterministic point-prediction reference on the existing (uniform-density) corpus, against which the CoG /
distributional contribution is measured. Reuses the slip `local` encoder; predicts **basin_transition** (tip vs stay)
and **resting orientation** (rest6d, rel. release) from (release-frame cloud + release params). Trained on the
object-disjoint split (58 train / 10 val / 12 held-out-mesh test), array job `1105057`/`1105100` (V100, `qos=batch`).
Eval vs a **no-motion** baseline (predict identity rotation) — the honest bar. Code: `train_drop.py`, `my_dataset_drop.py`,
`wm/drop_net.py`, `eval_drop.py`.

## Three-arm comparison (held-out test, n=19,881, 12 objects, 44 basin-transition positives)

| arm | basin-transition AUROC | resting-orientation geo median | reads |
|---|---|---|---|
| **full / release** (cloud + release params) | **0.973** | **2.38°** | the baseline |
| pose_only / release (release params, cloud zeroed) | 0.548 | 14.56° | control: is the cloud needed? |
| full / obj-frame (object-canonical cloud) | 0.971 | 4.59° | ablation: does the release frame help? |
| — no-motion (predict identity rotation) | — | 7.45° | the honest bar |

## Object-level bootstrap 95% CIs — all arms (n=19,881, 12 objects, 44 bt positives)

| arm | basin-transition AUROC [95% CI] | resting-orientation geo median [95% CI] |
|---|---|---|
| **full / release** | 0.973 [0.898, 0.977] | **2.38° [1.78, 3.82]** |
| full / obj-frame | 0.971 [0.926, 0.988] | 4.59° [3.31, 6.71] |
| pose_only / release | 0.548 [0.306, 0.776] | 14.56° [9.09, 41.56] |
| no-motion | — | 7.45° [7.18, 7.63] |

## Findings (object-level bootstrap 95% CIs)

1. **Beats no-motion (unlike the grasp baseline).** geo median **2.38° [1.78, 3.82]** vs no-motion **7.45° [7.18, 7.63]**
   — non-overlapping CIs. The drop task is a largely well-posed deterministic mapping (release orientation strongly sets
   the resting pose), so a point predictor is *adequate* on the natural corpus — exactly what the CoG stress tests
   predicted (the interesting failure is the near-boundary / hidden-CoM regime, not this one).
2. **The cloud is essential — no release-parameter shortcut.** Zeroing the cloud (pose_only) collapses basin-transition
   AUROC to **0.548 ≈ chance** and makes resting-orientation error (14.56°) *worse than no-motion*. Object geometry
   carries essentially all the signal (same clean result as the slip pose-only control).
3. **The release frame helps** the rotation prediction: **2.38° [1.78,3.82] (release) vs 4.59° [3.31,6.71] (obj-frame)**
   — lower error, but the median CIs **slightly overlap** (3.82 vs 3.31), so this is *suggestive, not clean-separated*
   (unlike the grasp gripper-frame win). Both clearly beat no-motion (upper bounds < 7.18). It barely changes
   basin-transition AUROC (0.973 vs 0.971) — the cloud handles tip/stay in either frame; the frame matters for the
   finer rotation regression. A stratified/tail view (below) should confirm where the frame helps.
4. **Basin-transition is predictable** despite rarity: AUROC **0.973 [0.898, 0.977]** (object-bootstrap; 634/1000
   resamples had both classes — the CI is honest about the 44-positive sparsity, but the lower bound 0.90 ≫ chance).

## Next steps — order decided (self + Codex, session `019f8664`): **B (eval hardening) → A (Gate B)**

The rotation result is already a solid baseline (n=19,881, CI-separated from no-motion, cloud ablation clean); the
basin-transition claim (44 positives) is too thin to be a narrative pillar. So a **short, tightly-scoped** hardening
pass first (NOT a polishing campaign), then the real contribution.

**B — eval hardening (do first, keep short):**
- [x] object-bootstrap CIs on **all** arms (table above).
- [ ] **transition-enriched eval** for the 44-positive sparsity (report bt on a transition-stratified view / add
      transition-prone held-out objects); frame bt classification honestly as **tail behavior**, resting-pose as the main win.
- [ ] **stratified error** by *data-derived* difficulty — distance-to-basin-boundary or release-instability quantiles
      (Codex: prefer these over arbitrary fixed bands), with the fixed `wm/metrics.py` degree bands (0–2/…/30+) as
      readable secondary reporting. Confirms *where* the release frame helps (finding 3).

**A — Gate B (the L3 payoff): CoG-aware distributional drop WM.** Success criteria to fix UPFRONT (Codex):
1. paired same-visible-state mass interventions must **change basin identity** (not just NLL);
2. an **oracle-latent** model should outperform a no-latent distributional model (the latent is real, learnable signal);
3. the distributional model must beat point prediction on **calibration/likelihood AND basin uncertainty** in the
   boundary subset.
Risks to avoid: a hand-designed sampler manufacturing the desired failures; gains only in NLL not basin prediction;
implausible CoM maps; the model learning object-identity/sampler artifacts instead of belief over latent inertia.

**Narrative (the thesis drop slots into):** natural drop = the **control/deterministic** regime (visible geometry +
release pose set the outcome → point prediction adequate; release-frame canonicalization works, as in grasp); Gate B =
the **frontier** regime (identical visible observations, hidden CoM → different basins → distribution *necessary*). Drop
is the **bridge**: it confirms the deterministic limit, then provides the cleanest path to hidden-latent multimodality —
the same thesis as grasp (canonicalization exposes deterministic contact structure) and slip (geometry → a calibrated
distribution over near-term dynamics).
