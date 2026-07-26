# Gate B — CoG-aware distributional drop WM (results, 2026-07-22)

The drop-phase frontier contribution: at near-boundary releases with a **hidden center of gravity**, the resting pose is
multimodal for a fixed visible observation, so a point predictor must fail and a **distribution** is necessary. Tests the
DiT-diffusion distributional WM (the CDWM signature head) against a point predictor and a basin-frequency baseline, with an
oracle-latent ceiling. Self-decided → Codex-driven diagnostic ladder (session `019f8664`).

## Setup
- **Data:** `gateb/generate_obj.py` (icing1) — 10 CoM-sensitive objects (V2-ranked), point-cloud-hull collision (= the WM's
  cloud), trimesh stable-pose near-boundary sampler, per-episode **hidden CoM** (explicit-inertial offset, recorded latent).
  ~12k episodes, all multimodal.
- **Model:** `wm/drop_diffusion.py` — DiT diffusion head (H=1, resting SE(3)), reuses `wm/dit.py` DDPM+DDIM + `DiTWMLocal`
  encoder. Arms: **point** (regressor), **diff** (no-latent), **diff_oracle** (given the true CoM). `train_gateb.py` (V100),
  `eval_gateb.py` (sample N=60 → cluster to basins → P(basin); NLL + Brier + boundary-stratified + basin-frequency baseline).
- **Criteria (Codex, pre-fixed):** C3 dist>point · C3b dist>freq-baseline · C2 oracle>no-latent.

## The diagnostic ladder (why the split matters)
| split | test | C3 dist>point | C2 oracle>no-latent | read |
|---|---|---|---|---|
| **object-disjoint** (2 held-out objects) | 2962 | ✗ (1.20 v 1.00) | ✗ (oracle worse) | cross-object calibration doesn't transfer at 10 objs |
| **within-object** (random episodes) | 1780 | ✓ (0.58 v 0.84) | ✓ | passes, but release-neighborhood **leakage** |
| **release-disjoint** (held-out pose cells) | 1830 | ✓ (0.58 v 0.79) | ✓ | **no leakage — the honest test** |

## Final result — release-neighborhood-disjoint (the no-leakage test), test n=1830
| arm | ALL nll / brier | BOUNDARY nll / brier (n=462) |
|---|---|---|
| point | 0.786 / 0.431 | 1.108 / 0.648 |
| **diff** | **0.579 / 0.309** | **1.096 / 0.602** |
| diff_oracle | 0.557 / 0.301 | 1.049 / 0.581 |
| freq-baseline | 0.828 / 0.517 | 1.105 / 0.663 |

**All six criteria pass** (both subsets, both metrics): C3 ✓, **C3b ✓** (beats the per-object marginal → learns
*release-specific* P(basin)), C2 ✓. Point top-1 crashes to 0.50 at boundaries; the diffusion covers the true basin **98%**.

## Object-bootstrap 95% CIs (per-episode NLL gain; the point estimates above needed CIs — Codex/rigor)
| comparison | ALL subset | BOUNDARY subset |
|---|---|---|
| **diff < point** | **+0.210 [+0.091, +0.310] SIG** | +0.014 [−0.135, +0.270] ns |
| **diff < freq-baseline** | **+0.252 [+0.090, +0.373] SIG** | +0.011 [−0.159, +0.447] ns |
| oracle < diff (latent causal) | +0.017 [−0.009, +0.050] ns | +0.057 [−0.027, +0.106] ns |

Per-object: **diff beats point on 9/10 objects** (only banana regresses) → not one-object dominance.

## Verdict (CI-scoped — honest)
- **★ Claim 1 (core) — CONFIRMED & SIGNIFICANT.** On held-out release neighborhoods (known objects), the DiT-diffusion
  distributional WM **significantly** beats the point predictor **and** the basin-frequency baseline (object-bootstrap CIs
  exclude 0), broadly across **9/10 objects**. In the near-boundary hidden-CoM regime a distribution is genuinely better
  than a point — the point predictor cannot represent one-observation/multiple-basin uncertainty.
- **Directional but NOT yet significant at 10 objects** (underpowered, → future work = more objects): the **oracle-latent
  advantage** (the "hidden CoM is *causal*" claim — oracle helps by +0.017/+0.057 but CIs include 0) and the
  **boundary-subset-specific** margins. So "the distribution captures the *hidden CoM's* effect" is *suggested*, not
  established; the established claim is the weaker-but-solid "distribution > point/frequency in this regime."
- **Claim 2 — cross-object transfer FAILS at 10 objects** (object-disjoint). Stated limitation.

## Where it sits (CDWM thesis, all three regimes, shared DiT head)
- **Grasp (rigid):** gripper-frame canonicalization exposes deterministic contact — point suffices.
- **Slip:** geometry → a calibrated distribution over near-term dynamics.
- **Drop:** point suffices on the natural (deterministic) corpus (beats no-motion); at near-boundary releases with a
  **hidden latent** (CoG), a distribution becomes **necessary** — the point predictor fails, the DiT diffusion wins.

## ★★★ SCALE-UP to 88 objects (77 CoM-sensitive + 12 negative controls) — FINALIZED, supersedes the 10-object result

134,576 episodes (icing1 job 1105160); retrain reldisjoint job 1105249; eval 1105260. Test = 20,410 eps / 88 held-out
release neighborhoods; near-boundary subset n=5,107. **Object-bootstrap 95% CIs (the narrow 10-object margins are now
decisive):**

| comparison | ALL | BOUNDARY | CoM-sensitive | neg-control |
|---|---|---|---|---|
| **diff < point** | +0.530 [+0.462,+0.604] **SIG** | +0.406 [+0.330,+0.481] **SIG** | +0.574 **SIG** | +0.300 **SIG** |
| **oracle < diff** (CoM causal) | +0.015 [+0.005,+0.024] **SIG** | +0.023 [+0.000,+0.047] **SIG** | +0.017 [+0.005,+0.028] **SIG** | +0.002 [−0.001,+0.008] **ns** |

diff also beats the frequency baseline (ALL 0.525 v 0.918, BOUNDARY 1.172 v 1.424). **Upgraded verdict — two clean,
CI-hardened findings** (better than the earlier point-estimate "all six pass"):
1. **The diffusion distribution significantly beats point prediction** (and frequency), everywhere, with large margins.
   But it is *also* significant on the CoM-**insensitive** negative controls (+0.300) → this is a **general
   distributional-modeling advantage** (diffusion > point regression), **NOT** CoM-specific. Do not claim otherwise.
2. **The hidden CoM is DEMONSTRABLY CAUSAL — and specifically so.** oracle-latent significantly beats no-latent on
   CoM-sensitive objects (+0.017 [+0.005,+0.028]) and is **null on the negative controls** (+0.002, ns). The scale-up
   (10→88 objects) turned this from underpowered to significant, and the negative controls confirm the effect is
   specific to where CoG matters. This is the crisp mechanistic result: *conditioning on the hidden inertial latent
   helps only where that latent controls basin selection.*

### Cross-object transfer (object-disjoint, 19 held-out objects; eval 1105263)
| comparison | ALL | BOUNDARY |
|---|---|---|
| diff < point | **+0.314 [+0.133, +0.489] SIG** | +0.134 [−0.344, +0.442] ns |
| oracle < diff | +0.009 [−0.016, +0.038] ns | +0.028 [−0.014, +0.077] ns |

**Cross-object diff > point now transfers** (SIG, vs the *failure* at 2 held-out objects: diff 1.20 > point 1.00) → the
distributional-modeling advantage generalizes to unseen geometries. **But oracle > no-latent does NOT transfer** (ns) →
the *latent-causality* is object-geometry-specific (knowing the CoM helps only once you've seen how it maps to that
object's basins). Boundary-subset cross-object margins are ns (wide CIs, 19 objects).

## ★★★ Complete drop-phase verdict (88 objects, both splits, object-bootstrap CIs)
1. **Distribution > point — significant, large, and transfers to unseen objects.** A general distributional-modeling win
   (also significant on the CoM-insensitive negative controls, so *not* CoM-specific).
2. **The hidden CoM is demonstrably causal for known objects** — oracle > no-latent is significant *and specific*
   (significant on CoM-sensitive objects, null on negative controls). The scale-up (10→88 objects) turned this from
   underpowered to significant.
3. **Latent-causality does not yet transfer cross-object** (oracle ns on held-out objects) — an **open problem**: the
   CoM→basin mapping is object-geometry-specific; transferring it needs more objects or geometry-structured modeling.

**Contribution (honest, CDWM-scoped):** we characterize *when* resting-pose prediction requires a distributional (DiT
diffusion) world model — a point predictor suffices on natural/deterministic drop, but at near-boundary releases a
**hidden CoM** makes the outcome multimodal, and (a) a distribution is necessary and generalizes, (b) the hidden inertial
latent is causally useful where CoG matters. Framing: *hidden-latent distributional prediction* (not "belief-state").
### Strengthening additions (Codex-suggested)
- **Model-independent causality** (`scripts inline`): because the hidden CoM was sampled *independently of the release*,
  the unconditional **I(basin; CoM) is the causal effect** (randomized). It is **0.165 bits on CoM-sensitive** objects vs
  **0.046 on negative controls**, difference **+0.119 [+0.065, +0.172]** (object-bootstrap, excludes 0). So CoM causality
  holds *without* relying on the oracle model — a data-level randomized-intervention result.
- **Per-object win table:** diff < point on **88/88 objects** (median gain +0.48, min +0.05) — the distributional advantage
  is broad, not driven by a few objects.
- **Calibration (ECE, reliability of P(basin)):** the **diffusion is well-calibrated — ECE 0.015** (confidence 0.803 ≈
  accuracy 0.814); the **point predictor is badly miscalibrated — ECE 0.227** (confidence 0.498 vs accuracy 0.726). So the
  distributional prediction's uncertainty is *trustworthy*, not just lower-NLL. (oracle ECE 0.016.)
- **Bookkeeping:** 89 listed → **88 generated** (`048_hammer` excluded: no valid point-cloud stable-pose set); the set is
  **76 CoM-sensitive + 12 negative controls**.

- **Physical-density realism check** (`density/v3v4/physical_density_check.py`, hammer 5-hull CoACD, icing1 job 1105266):
  replacing the controlled offset with a **physical per-hull material density** (head hollow→steel, 1000–7800 kg/m³,
  randomized; handle 500) reproduces the effect — **multimodal** rest basins and **I(basin; physical head-density) = 0.161
  bits**, essentially identical to the controlled-offset CoM causality (0.165). So the controlled explicit-inertial offset
  is a **faithful proxy for physical mass-distribution variation, not a synthetic artifact.**
- **Physical-density realism, extended to 6 real-mesh objects** (`multi_physical_density.py`, CoACD, icing1 job 1105472):
  CoACD-decompose each object's mesh, make one end's hulls heavy and randomize that density (a physical hidden CoM), and do
  PAIRED near-boundary drops (identical release, low vs high density, so density is the only varied factor). Across six
  CoACD-decomposed real-mesh objects, randomized end-heavy per-hull density **causally affected the resting basin in every
  object (6/6), with multimodal basin structure in all cases.** Effect size was heterogeneous (paired basin-disagreement:
  can-opener 0.53, Yoshi 0.53, mouse 0.14, hair-straightener 0.12, bull 0.055, clamp 0.028; mean 0.23, mean I(basin;density)
  0.093 bits). Because releases used a GENERIC balanced-long-axis heuristic rather than an object-specific saddle search,
  these rates are a **lower-bound / heuristic-sampled realism check, not each object's maximal density sensitivity**
  (Codex-agreed framing, session `019f8f6f`; results `density/physical_multi.json`). Together with the hammer (I=0.161),
  this confirms the controlled explicit-inertial offset is a faithful proxy for physical mass-distribution variation across
  objects, not a synthetic artifact. The full 88-object physical version (CoACD + regen + retrain) remains a non-blocking
  infra lift; a per-object saddle search or a controlled-vs-physical match on identical geometry would sharpen it if a
  reviewer asks.

**Rigor status:** all high-value items complete — significance CIs, model-free causality, per-object breadth (88/88),
calibration (ECE), negative controls, and the physical-density realism check. The 10-object sections above are the
diagnostic ladder that motivated the scale-up.

## Cross-object latent transfer — SOLVED (2026-07-23; full writeup [`cross_object_transfer_plan.md`](cross_object_transfer_plan.md))
The item previously listed here as future work is now a result. The CoM→basin latent did NOT transfer cross-object with the
abstract `[axis, delta]` encoding (ns), but DOES when the CoM is fed **geometry-grounded** (per-point vector-to-CoM +
distance). Object-disjoint, 19 held-out objects, paired object-bootstrap CI: grounded > no-latent **+0.184 SIG**, grounded >
abstract **+0.191 SIG**, grounded > shuffled-CoM **+0.160 SIG**; broad across **18/19** objects; grounded **ECE 0.019**
(well-calibrated). Uses the oracle CoM supplied at test (estimating CoM from observations = the remaining future work).
Codex signed off write-up-ready (session `019f8f01`). GPU-consistency verified (V100-vs-TITAN training effect +0.013 NLL,
CI includes 0). Only full-scale per-hull physical density (needs CoACD) remains non-blocking.

## Demo videos (see [`demo_videos.md`](demo_videos.md))
Same object + same near-boundary release, sweep the hidden CoM → the resting basin flips (banana / medium_clamp / mouse;
MuJoCo + EGL from the WM's own point-cloud hull; `render_drop_demo.py`, job 1105467).
