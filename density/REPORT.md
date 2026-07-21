# Object density / center-of-gravity in the drop corpus — investigation report

*Branch `drop`. 2026-07-21. Self-decided → Codex-reviewed (session `019f8664`) → converged at every step.
Detailed per-step docs linked inline; this is the consolidated reference.*

---

## Summary

The drop corpus was simulated in MuJoCo with **uniform object density**, so there is no center-of-gravity (CoG)
diversity to study. We asked whether adding realistic / diverse mass distributions is worth it. Answer: **yes, but as a
targeted contribution, not by editing the existing corpus.**

- **Uniform density gives zero CoG diversity** — under MuJoCo `inertiafromgeom`, the CoM sits at the geometric centroid
  *regardless of the density value*. The real lever is **non-uniform density across the ≤24 CoACD hulls each object
  already has.** (proven: `test_density.py`)
- **CoG barely matters on the corpus as-sampled.** Only **2.3%** of 153,996 episodes tip into a new basin; a plausible
  83 mm CoM shift changes **0%** of outcomes in the stable-pose + small-tilt regime. The corpus is shape-dominated.
- **CoG matters in a specific regime:** for **CoM-sensitive shapes** (elongated/asymmetric — spoons, clamps, tools,
  figurines; *not* flat/round objects), released **near a stability boundary**, where a **hidden** CoM makes the resting
  pose **multimodal** and a point predictor structurally fails.
- **So the contribution is:** co-design a **near-boundary release sampler** + **per-hull heterogeneous densities**, then a
  **distributional / belief-state drop WM** — the drop analogue of the slip-phase MDN result. Layered **after** the
  baseline drop WM, not on its critical path.

| finding | number | doc |
|---|---|---|
| YCB proxy mass error | off >15% for **18/24** objects (hammer 2×) | [T1](object_inventory.csv) |
| corpus episodes crossing a basin | **2.3%** (net_rot p50 = 7.7° ≈ release tilt) | [audit](audit_results.md) |
| CoM-sensitive objects (analytic, 558 objs) | median TV **0.28**, **28% > 0.5** | [V2](v2/RESULTS.md) |
| sensitivity at the stable-pose saddle | **92%** basin disagreement (vs 0% in-basin) | [V3](v3v4/RESULTS.md) |
| hidden-CoM resting distribution at the saddle | **multimodal** [0.76, 0.23]; distributional beats point by **1.06 nats** | [V4](v3v4/RESULTS.md) |

---

## 1. The question and the load-bearing physics

The colleague noted the corpus assumes uniform density (400 kg/m³ × convex-hull-volume proxy for 71/80 objects; measured
mass for only 9 YCB), so there is no CoG diversity. The first instinct — "give each object category its real (uniform)
density" — **does not work**:

> MuJoCo `compiler inertiafromgeom` + a single uniform density ⇒ body CoM = the composite-hull volumetric centroid,
> **independent of the density value.** Scaling every hull's density by the same factor scales mass and inertia magnitude
> but does **not move the CoM.**

Confirmed empirically (`test_density.py`, mujoco 2.3.7): a body at uniform density 1000 vs 5000 kg/m³ has CoM at exactly
the same point (mass ×5, CoM unchanged); only **heterogeneous per-hull density** moves the CoM (to the analytic
mass-weighted centroid). The dataset README and `admission_matrix.md` agree empirically — mass only moved the resting-
penetration tail, it changed no admit/reject verdict. **So "update per-category uniform density" buys mass realism only,
zero CoG diversity. The lever is per-CoACD-hull density.**

The density ladder: **L1** uniform-real (no CoM change) · **L2** fixed non-uniform (CoM shifts, but deterministic per
object → learnable, no hidden latent) · **L3** randomized non-uniform (per-episode hidden CoM → aleatoric multimodality →
belief-state WM). Target = **L3, grounded in real latent states** (fill level, head/handle material, ballast), not
arbitrary randomization.

## 2. Real masses (T1)

All 24 YCB objects matched to published masses from [`elpis-lab/ycb_dataset`](https://github.com/elpis-lab/ycb_dataset)
(`ycb_mass_elpis.json`; that repo also ships ready MuJoCo XMLs + 5-hull CoACD per object). The 400 kg/m³ proxy is **off
>15% for 18/24** (hammer 0.33 vs real 0.665 kg; cups ~5× over). 56 GSO objects left at proxy — precise masses are a
Gate-B item; the analysis below uses geometry, not mass. Full table: `object_inventory.csv`.

## 3. Does CoG change outcomes? — pilot + audit

**Pilot (T4, `pilot/`):** the real YCB hammer, per-hull densities giving an 83 mm head-heavy CoM shift vs uniform,
dropped from identical paired poses. In the **corpus's stable-pose + small-tilt regime: 0% basin change** at every tilt
(the flat hammer rocks back to the same face; the CoM shift is in-plane). **Near tipping (balanced on end): 100%** basin
disagreement. → CoG matters only where the dynamics are near a stability boundary.

**Audit (`audit_corpus_boundary.py`):** across all 153,996 valid episodes, only **2.3%** cross into a new basin;
`net_rot_from_release` p50 = 7.7° ≈ the release tilt (objects just return to their release basin). The tail concentrates
in ~5 tall/asymmetric objects (cracker_box 40%, Android figure 45%, Razer mouse 44%); **72/80 objects tip <1%**;
bottle-can and handled classes 0%. → the corpus is shape-dominated as-sampled; CoG is screened off for ~98% of episodes.

## 4. Which objects are CoM-sensitive? (V1 + V2)

`v2/stable_pose_sensitivity.py` (analytic, `trimesh.compute_stable_poses` takes the CoM as input) compares each object's
resting-pose distribution at the centroid vs a plausible CoM offset along **all three** principal axes (the V1 out-of-
plane test folded in), over **558 objects**. **CoG-sensitivity is a shape property:**

- **CoM-ROBUST** (TV ≈ 0.02): flat objects — plates, game/DVD cases, tablets. Two stable poses; a plate lands flat
  regardless of CoM.
- **CoM-SENSITIVE** (TV ≈ 0.9–1.0): elongated / asymmetric — spoon, scissors, spatula, clamp, mouse, hair straightener,
  figurines. CoM flips which end/face rests down.
- Median TV 0.28, **28% > 0.5** → the effect is **not hammer-only**, across many shapes.

**Key correction to intuition (Codex's warning, confirmed):** `003_cracker_box` tips in **40%** of corpus episodes yet is
**CoM-robust** (TV ≈ 0.1) — a box always rests on one of its 6 faces. **Corpus transition rate ≠ CoG-sensitivity.** Target
objects by the CoG-sensitivity ranking, not by how often they tip.

## 5. Where does CoG decide the outcome? (V3)

`v3v4/boundary_multimodal.py` (dynamical, hammer): sweep the release from lying-flat to balanced-on-end. CoM-sensitivity
(centroid-CoM vs head-heavy-CoM settling in different basins) is **~0 deep in the basin and peaks sharply at the
stable-pose saddle** (θ ≈ 95°: **92%** disagreement, 179.6° median — nearly opposite outcomes). → A **boundary sampler
that targets the saddle** (derived from the object's stable-pose graph) finds the CoM-decisive poses; the corpus's
θ ≈ 0 sampling never goes there.

## 6. Why a distribution, not a point? (V4)

At the saddle, treat the CoM as a **hidden latent** (unknown internal mass) sampled from a plausible prior. For a **fixed
visible observation** (same geometry + same release), the resting pose is **genuinely multimodal** — 3 basins
[0.76, 0.23, 0.01] — and a **distributional predictor beats a point predictor by 1.06 nats NLL.** A point prediction is
structurally inadequate here. This is the drop analogue of the slip-phase MDN result: geometry + release → a *calibrated
distribution* over the resting pose, whose irreducible spread is CoM uncertainty.

## 7. Verdict and what's next

**GO for a CoG contribution, as a targeted, separate piece** — CoM-sensitive shapes × near-boundary releases → multimodal
outcomes needing a distributional/belief-state WM. **NO-GO for editing the existing corpus** (shape-dominated, 2.3%). It
belongs **after** the baseline drop WM (outcome + resting-pose point prediction on the current corpus), which is the
deterministic reference the CoG story is measured against.

**Gate B (evidence-backed, when prioritized):** T5 co-design a saddle-based boundary release sampler + per-hull
heterogeneous densities grounded in real latent states; T6 re-simulate (colleague harness or scaled mini-harness);
T7 CoG-aware distributional WM, reported with a proper scoring rule (energy/CRPS), never point-only. Report **both** the
natural and boundary release distributions; paired counterfactuals; negative controls.

**Open refinements (V3/V4):** extend the dynamical V3/V4 to 2–3 more objects (not hammer-only); fix the round-object
degeneracy flag in the analytic V2 tool (sphere collapse: n_eff ≈ 1 with high TV is meaningless).

## Artifacts
```
density/
  REPORT.md                     <- this file
  object_inventory.csv          T1: 80 objects, 24 YCB real masses
  ycb_mass_elpis.json           source masses (elpis-lab/ycb_dataset)
  test_density.py               T3: uniform->centroid, per-hull->shift (PASSES)
  audit_corpus_boundary.py      corpus near-boundary audit (2.3%)
  audit_results.md
  pilot/    drop_ab.py + RESULTS.md          T4: hammer A/B (0% flat, 100% near-tipping)
  v2/       stable_pose_sensitivity.py + RESULTS.md + .csv   V1+V2: shape property, 558 objs
            drop_sweep.py                     (dynamical; superseded by the analytic tool)
  v3v4/     boundary_multimodal.py + RESULTS.md   V3 (saddle peak) + V4 (multimodal, distributional wins)
```
Plan of record: [`../notes/drop_density_todo.md`](../notes/drop_density_todo.md) · colleague brief (中文):
[`../notes/drop_density_plan_zh.md`](../notes/drop_density_plan_zh.md).
