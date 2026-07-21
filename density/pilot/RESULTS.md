# T4 A/B pilot — does center of gravity change where an object settles?

**Setup.** Real YCB hammer (elpis-lab CoACD, 5 hulls). Two mass distributions dropped from **identical** release
poses (paired), MuJoCo, physics matched to the corpus README (condim=6, μ_roll=1e-4, zero-vel release, 4 s, at-rest
<6 mm/s ∧ <6°/s). Harness: `drop_ab.py` (first-pass mini-harness, not the colleague's corpus-v1 harness).

- **UNIFORM** — one density for all hulls → CoM at the geometric centroid (**= the current drop corpus**).
- **HEAD_HEAVY** — handle hull ~500 kg/m³ + head hulls ~4180 kg/m³ → reproduces the real 665 g with a head-weighted CoM.
- Realized **CoM shift = 83 mm** (~25% of the 335 mm object length). Mechanism confirmed.

## Result

| regime | tilt | basin disagreement | median \|Δ rest-orientation\| (U vs H) |
|---|---|---|---|
| **flat** (corpus sampling: stable pose + small tilt) | 10–60° | **0%** everywhere | ≤ 4° |
| **upright** (balanced on one end, near-tipping) | 0° | **100%** | **180°** |
| upright | 5° | 50% | 84° |
| upright | 10–30° | 12–33% | 5–11° |

## Verdict — conditional GO

- **On the corpus's stable-pose + U[0,15°] sampling, CoG diversity is a NO-GO:** an 83 mm CoM shift produces **0%** basin
  change and ≤4° rest-angle change. The flat object rocks back to the same face regardless of CoM; the shift is *in-plane*
  (within the support face). This confirms Codex's shape-dominated risk **and** the corpus README ("mass nearly irrelevant
  to rigid settling") — for that sampling.
- **Near stability boundaries, CoG is decisive:** balanced on end, uniform vs head-heavy settle to **opposite** poses
  (100% disagreement, 180° apart); a 5° perturbation still flips half the episodes. CoM chooses the fall direction there.

**Implication for the plan.** Adding densities to the *existing* corpus and re-simulating changes almost nothing → not
worth it alone. CoG becomes a first-class hidden latent **only if the release sampling also visits near-tipping /
metastable poses**. That is the evidence-based design correction: **CoG diversity ⇒ must co-design a near-boundary release
distribution.** This is exactly the L3 frontier (hidden CoM → multimodal resting pose), now with a concrete requirement.

Note the corpus deliberately lives in the shape-dominated stable regime (its admission gate *excludes* roll-prone /
near-boundary configs). Making CoG matter means deliberately sampling where they chose not to.

## Caveats / follow-ups
- **Hammer only.** The qualitative answer (decisive near tipping, negligible in stable regime) is likely general, but
  confirm on a can (roll axis), a clamp, and an asymmetric concave object before building anything.
- "Upright/balanced-on-end" is my stand-in for a metastable pose; a principled near-boundary sampler (perturb around the
  saddle between two stable basins) is the real T5 design.
- Head/handle labeling from CoACD geometry; reversing it only flips the fall direction, not the magnitude conclusion.
- **In-plane shift only.** The 83 mm shift is along the long axis, so by construction it can't change which broad face is
  down in the flat regime — it only re-torques rocking. A dense top/bottom (out-of-plane) shift *could* bias face
  selection even flat; test it as a **sensitivity bound**, though for a hammer along-axis (head-heavy) is the physical one.

## Codex reflection — converged (session `019f8664`)

Codex endorses the conclusion ("CoG is not globally useful; it's useful near basin boundaries — exactly the right
diagnostic; a hidden mass latent only matters where dynamics are sensitive to it") and adds guardrails:

- **Honest framing of the pivot:** the claim is **not** "real drops are always multimodal" — it is "belief-state contact
  modeling matters in the *subset* of contacts where hidden inertial properties control basin selection." Report **both**
  the natural/stable release distribution *and* the boundary distribution; do not hide that CoG effects are sparse in the
  original sampling.
- **Keep the boundary sampler physical:** derive near-boundary poses from the object's own **stable-pose graph / support
  geometry** (edges, ends, narrow faces, saddle transitions), **not hand-picked angles**. Paired counterfactuals + **negative
  controls** (objects where plausible CoM shifts should *not* matter). Physically-motivated density maps only (fill level,
  head/handle material, base-heavy containers, ballast).
- **Sequencing unchanged and strengthened:** baseline drop WM first (it gives the deterministic reference: on the current
  stable-regime corpus, shape/contact dominates and a point predictor is adequate); CoG/belief-state is a **targeted later
  extension**, not a detour.
- **Go/no-go metric = paired basin-transition probability near objectively-derived basin boundaries**, NOT average
  rest-angle change.

**Four tests before committing engineering to the dataset:** (1) out-of-plane sensitivity on 2–3 objects (stress test);
(2) object diversity (hammer / bottle-can / box / asymmetric toy-tool / flat — effect must not be hammer-only); (3)
boundary-sampler validation (it finds high-sensitivity poses under tiny physical perturbations); (4) distributional
necessity (identical visible observation → genuinely multimodal rest under hidden mass maps, and a distributional
predictor beats a point predictor on **likelihood/calibration**, not just mean error).
