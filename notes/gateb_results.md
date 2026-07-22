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

## Remaining rigor (flagged, not run autonomously)
- Object/release-neighborhood **bootstrap CIs** on the diff−point delta (the boundary-subset NLL margin 1.096 v 1.108 is
  narrow, though Brier is clear and the direction is consistent across all three splits and both metrics).
- Reliability/ECE curves; per-object breakdown (confirm the win isn't one or two objects).
- Per-hull *physical* density (vs the controlled explicit-inertial offset) for the final dataset.
