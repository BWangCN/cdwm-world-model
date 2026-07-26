# Cross-object latent transfer — study plan (2026-07-22)

The drop-phase Gate B open problem, now being explored (user-chosen; the one high-impact remaining item).

## The problem (grounded)
The resting **basin** (which stable pose the object settles into) depends on the **visible** state (geometry + release
pose) and a **hidden** inertial latent (the CoM / mass distribution). Established results (`notes/gateb_results.md`):
- For **known** objects, the hidden CoM is *usable*: oracle-latent > no-latent is significant (+0.017; also model-free
  I(basin;CoM)=0.165 on CoM-sensitive).
- For **unseen** objects, the true CoM is *useless*: oracle > no-latent is **ns cross-object** (+0.009, eval 1105263).

Why: the mapping **CoM → basin is object-geometry-specific.** "Mass shifted +2 cm along axis 1" only determines the basin
if you know how that shift meets *this* object's stable-pose structure (faces / tipping edges / saddles). A head-heavy
hammer tips toward its head; a head-heavy pair of scissors does something else. The model learns per-object CoM→basin
combinations and can't apply them to a novel shape.

**Goal:** get the latent to help on *unseen* objects — learn a general `f(geometry, hidden-CoM) → basin distribution` that
transfers. Metric = **oracle-latent > no-latent on the object-disjoint split** (currently +0.009 ns → target: significant),
object-bootstrap CI; keep the known-object (reldisjoint) result as the control.

## Root cause: how the latent is fed
Currently the CoM latent is an **abstract offset vector** `[axis_idx, delta]` added to the conditioning; the model must
*learn* to bind it to geometry (works per-object, doesn't transfer). We already feed geometry (the cloud); the fix is to
feed the latent **geometry-grounded** so the encoder perceives *where the mass sits within the shape* (object-agnostic).

## Approaches (cheap → deep)
1. **Data lever (cheapest sanity):** does *more object diversity* alone close the oracle-transfer gap? (retrain with more
   generated objects.) Tells us if it's a coverage problem vs a representation problem.
2. **★ Geometry-grounded latent encoding (first real attempt):** feed the CoM as a **spatial location co-registered with
   the cloud** — e.g. a per-cloud-point feature = the point's position/distance relative to the CoM. The PointNet encoder
   then sees "the heavy region is *here* in the shape," which transfers. (oracle = true CoM location; no-latent = feature
   zeroed/absent. A pure representation change, no new architecture.)
3. **Stable-pose-structured conditioning (deeper):** encode the CoM relative to the object's stable-pose graph (distance to
   tipping edges / the saddle it sits nearest) — a physics-structured inductive bias.

**First study:** approach 2 (geometry-grounded CoM feature) — highest leverage, cheap (edit `build_feats_drop` to add a
CoM-relative per-point feature; retrain object-disjoint; re-eval oracle-transfer). Fall back to 1 (more data) and 3
(stable-pose structure) if 2 is insufficient.

### Confirmed 4-arm design (self + Codex, session `019f8664`)
Same object-disjoint split + same training budget; the 4 arms isolate whether *grounding the latent in geometry* is what
transfers:
- **no-latent** — cloud + release, no CoM (defines the hidden-latent marginal; need NOT improve).
- **abstract-oracle** — the current oracle: CoM as `[axis, delta]` in the context vector.
- **grounded-oracle ★** — CoM as **per-point features**: the *vector from each cloud point to the CoM* (3) + distance (1),
  in the release frame (Codex: scalar distance alone loses direction — use the vector). CoM in `pts`, not the context.
- **shuffled-oracle (control)** — grounded features but with a **wrong/shuffled CoM** → must NOT help; guards against the
  feature becoming an object-identity cue rather than a transferable physical relation.

**Validated iff** grounded-oracle > abstract-oracle **and** grounded-oracle > shuffled-oracle on object-disjoint.
Report: oracle-vs-no-latent on object-disjoint (target: significant), the grounded-vs-abstract-vs-shuffled comparison,
the known-object reldisjoint result (no-regression control), per-object bootstrap CIs, NLL/Brier (not just top-1).

**Risks (Codex):** the true latent may be full inertia/density not just CoM; CoM→basin transfer depends on stable-pose
topology PointNet may not infer from sparse object counts; object-disjoint CIs may stay wide; the grounded feature could
leak object identity (→ the shuffled control); the effect may be real-but-small (a power problem).

## Paper framing (which paper does this belong to)
- **Folds into the drop paper** if a modest fix (approach 1 or 2) closes the gap → the current *limitation* ("latent
  causal for known objects, doesn't transfer") becomes a *result* ("...and generalizes to novel objects"). Same thesis,
  strictly stronger.
- **Could anchor a separate methods paper** if a substantial geometry-structured inductive bias (approach 3 / an
  identification framework) is required — because such a method is **cross-cutting** (applies to grasp + slip + drop, the
  whole CDWM family), so drop stays the regime-characterization paper and the transfer method stands alone.
- We won't know which until we try; the cheap data-lever + representation tests disambiguate "data/representation" (fold
  in) vs "needs a real new inductive bias" (potential new paper) early.

## RESULT (2026-07-23) — geometry-grounded latent DOES transfer cross-object ✓
Object-disjoint split, 88 CoM-sensitive objects (52 train / 17 val / **19 test**, held-out geometry), N=40 basin samples,
paired object-bootstrap CI. Confound-free version = all four arms trained on the same GPU (TITAN RTX), eval `1105406`; a
mixed-GPU version (`1105374`) agrees within noise (GPU note below).

| Arm | latent encoding | NLL | Brier |
|-----|-----------------|-----|-------|
| no_latent | none | 0.679 | 0.318 |
| abstract_oracle | `[axis, delta]` offset (old) | 0.686 | 0.323 |
| **grounded_oracle ★** | per-point vector-to-CoM (3) + distance (1) | **0.495** | **0.243** |
| shuffled_oracle | grounded features, WRONG CoM (control) | 0.655 | 0.308 |

Transfer gain over no_latent (>0 and CI excludes 0 = latent helps on unseen objects):
- abstract `-0.007 [-0.019,+0.007] ns` — the old encoding does NOT transfer (reproduces the prior null).
- grounded `+0.184 [+0.084,+0.312] SIG`
- shuffled `+0.024 [+0.002,+0.051] SIG` — small; the extra channels carry a little geometry even with a wrong CoM.

Key controls (both the plan's validation conditions hold):
- grounded < abstract `+0.191 [+0.091,+0.325] SIG` → grounding the CoM in geometry is what unlocks transfer.
- grounded < shuffled `+0.160 [+0.073,+0.281] SIG` → it is the CORRECT CoM that drives the gain, not the 4 extra channels.

**Hardening pass (Codex parity, eval `1105454`, seeded; `eval_transfer_hardening.py`):**
- Per-object breakdown (rules out few-object dominance): grounded beats no_latent on **18/19** held-out objects (median
  +0.091, only one object -0.06), and beats abstract on **18/19** (median +0.122). Broad, not carried by the one +0.98 winner.
- Calibration/ECE (same method as the Gate B ECE 0.015): grounded **ECE 0.019** (conf 0.818 ≈ acc 0.825) is the
  best-calibrated AND most-accurate arm; no_latent 0.023, abstract 0.024, shuffled 0.019 (all < 0.025, well-calibrated).

**CODEX FINAL SIGN-OFF (2026-07-23, session `019f8f01`): write-up-ready, no must-fix.** Frame narrowly and strongly:
oracle CoM supplied at test, object-disjoint transfer, geometry-grounded latent beats no-latent/abstract/shuffled, broad
across 18/19 held-out objects, calibrated on the transfer split. (Estimating CoM from observations = future work, not a gap.)

**Conclusion.** The CoM→basin latent transfers to unseen objects when fed geometry-grounded (per-point vector-to-CoM +
distance) rather than as an abstract offset. This closes the plan target (oracle > no-latent on object-disjoint, was
+0.009 ns, now +0.184 SIG). Per the framing above this makes the drop-paper limitation into a strictly stronger result:
"the hidden inertial latent is causal for known objects AND generalizes to novel geometry when grounded." Grounded per-point
CoM features live in `my_dataset_gateb.py` (`latent_mode` grounded/shuffled); arms in `train_gateb.py`, eval `eval_transfer.py`.

### GPU-consistency check (does the GPU change results?)
Prompted by the reschedule mixing V100 and TITAN RTX.
- **Eval / inference:** made deterministic by seeding the DDIM initial noise (added optional `x_T` to `ddim_sample`,
  CPU generator → device-independent). GPU then changes only FP rounding. [Direct inference A/B (`1105438`) NOT run: V100 held
  by non-session jobs for 3.5h+, cancelled. Expected ~0 by construction; corroborated in that the seeded diff V100-ckpt eval
  (0.6891) matches the earlier UNSEEDED no_latent V100 eval (0.689) to 3 decimals across both seeding and config.]
- **Training:** retrained the `diff` arm (an untouched genuine V100 checkpoint, job `1105250_1`) on TITAN with identical
  config (epochs 120, bs 256, seed 0, object split; distinct tag `diff_ttcheck`, verified no overwrite of `diff`). Both
  eval'd on TITAN with identical seeded noise: V100-trained **0.6891** vs TITAN-trained **0.6765** →
  **training-GPU effect +0.0126 NLL [-0.0016,+0.0307], CI includes 0 (not significant)**. ~4-6% of the transfer effect
  sizes, consistent with single-seed run-to-run noise, not a systematic GPU effect.
- **Science is robust:** the transfer conclusion is identical on mixed-GPU (`1105374`) and all-TITAN (`1105406`) evals
  (grounded < abstract +0.193 vs +0.191; grounded < shuffled +0.158 vs +0.160). The decisive comparisons are same-GPU by
  construction, so immune regardless. Harness: `gpu_ab.py`, `gpu_ab_compare.py`.

**Lesson (my error, fixed):** first attempt lost the V100 no_latent checkpoint because my "preserve" `cp` ran AFTER the
retrain had already overwritten `gateb_runs/no_latent/best.pt`. Redone with an untouched V100 arm (`diff`), a distinct
output tag, an md5 backup, and an in-monitor assertion that the source ckpt stays unchanged.
