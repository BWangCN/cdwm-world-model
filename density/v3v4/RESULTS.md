# V3 + V4 — boundary sampler + distributional necessity (the L3 thesis, on the hammer)

`boundary_multimodal.py` (dynamical, MuJoCo; reuses the pilot hammer hulls + settle machinery). CoM set via
explicit `<inertial>` along the object's long axis relative to the **true volumetric centroid**; mass + isotropic
inertia held fixed, only CoM moves.

## V3 — CoM-sensitivity peaks at the saddle

Release orientation swept from lying-flat (θ=0, deep in a basin) to balanced-on-end (θ≈90–95, the saddle between
two stable poses). Disagreement = a centroid-CoM vs head-heavy-CoM drop settling in different (yaw-invariant) basins.

| θ (long axis up from horizontal) | 0–75° | 85° | 90° | **95° (saddle)** | 105° |
|---|---|---|---|---|---|
| basin disagreement | **0%** | 8% | 20% | **92%** | 0% |
| median rest-pose diff | ≤1.8° | 1.2° | 1.7° | **179.6°** | 1.0° |

**Sensitivity is ~0 everywhere except a sharp peak at the saddle**, where a plausible CoM shift flips the outcome to
nearly the opposite pose. Validates the pivot: a **boundary sampler that targets the stable-pose saddle** (derived from
geometry) finds the CoM-decisive poses; the corpus's stable-pose + U[0,15°] sampling (θ≈0) sits in the flat 0% region.

## V4 — at the boundary, hidden CoM makes the resting pose multimodal

At the saddle (θ=95°), fix the visible observation (geometry + release pose) and treat the CoM as a **hidden latent**
(unknown internal mass, e.g. how head-heavy the tool is), sampled from a plausible prior:

- **3 distinct rest basins, fractions [0.76, 0.23, 0.01] → genuinely multimodal** (2 modes > 15%).
- **NLL: point-predictor 1.66 vs distributional 0.60 → distributional wins by 1.06 nats.**

For a fixed observation with hidden CoM the outcome is irreducibly uncertain and multimodal, so a point predictor is
structurally inadequate and a **distributional / belief-state predictor is necessary** — the drop analogue of the
slip-MDN result (geometry + release → a calibrated distribution, whose spread is CoM uncertainty).

## Caveats (honest scope)
- **Hammer only** (dynamical depth); breadth = V2's 558-object analytic sweep (CoM-sensitivity is a shape property,
  not hammer-specific).
- **Isotropic inertia** held fixed (clean CoM ablation, Codex-endorsed as ablation-only); the eventual dataset uses
  physically-consistent per-hull inertia.
- The V4 NLL is a **schematic** point-vs-distribution contrast on the induced basin distribution, not a trained-model
  comparison — that (train an actual distributional vs point drop WM on a boundary+heterogeneous-density dataset) is
  the eventual T7 contribution this de-risks.

## Bottom line (Gate A.5 complete)
CoG diversity is worth building **as a targeted contribution**: it matters for CoM-sensitive shapes (V2) near stability
boundaries (V3), where hidden CoM yields multimodal outcomes that need a distributional WM (V4). It is negligible for
flat/round objects and for the corpus as-sampled (audit 2.3%) — so the contribution = **co-design a boundary release
sampler + heterogeneous densities**, layered after the baseline drop WM, not on its critical path.
