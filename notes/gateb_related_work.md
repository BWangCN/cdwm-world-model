# Gate B — related work + positioning (literature survey, 2026-07-22)

Survey to confirm the near-boundary + hidden-CoM regime is significant AND underexplored, and to position against prior
CoM-aware work. Self-survey (WebSearch) → Codex pressure-test (session `019f8664`).

## Landscape — what exists (so we do NOT claim "nobody studies CoM + resting poses")
1. **Classical part-pose statistics / stable-pose probability** (Goldberg/Mirtich part-feeding; `trimesh.compute_stable_poses`,
   which *we use*): estimate stable-pose probabilities from **known geometry + known/sampled CoM**. **The closest prior — must
   acknowledge directly.**
2. **CoM-aware grasping**: extensive — CoM-based grasp planning/regrasping, CoG estimation (incl. via diffusion,
   [2507.19242](https://arxiv.org/html/2507.19242)), tactile CoM sensing. Focus = grasp *stability*, not resting-pose prediction.
3. **Resting-pose / stable-placement prediction**: predict stable placement poses; drop-in-sim for stable poses;
   probabilistic drop belief (von Mises angle + Gaussian position). But **uniform density / geometric** stability; multimodality
   is from **shape/symmetry**, mass known.
4. **Diffusion multimodal pose**: [AnyPlace](https://arxiv.org/pdf/2502.04531), [DiffusionNOCS](https://arxiv.org/pdf/2402.12647)
   — multimodality from **symmetry/task/geometry**, not a hidden physical latent.
5. **Tipping/toppling**: classic physics; robot tip-over *avoidance*; pivoting under uncertain CoM via **robust control/friction**;
   learning tipping dynamics. Focus = avoid/control, not distributional outcome prediction.
6. **Inertial-parameter estimation**: extensive — estimate mass/CoM/inertia from tactile/vision/encoders; structural vs practical
   **identifiability**. Focus = *estimate the parameter* (point). Also a density-field dataset (XDen-1K, 2512.10668).

## The gap (defensible, sharpened by Codex)
Prior work estimates stable-pose distributions from **known** geometry / known-or-sampled CoM; we **learn a conditional
distribution from point-cloud observations where the mass distribution is HIDDEN**, and show this hidden inertial latent
**only matters in near-boundary regimes**. The pieces (the combination is the novelty, not any single one):
- **Causal source** of multimodality = hidden CoM / mass distribution, *not* shape symmetry.
- **Regime characterization** = natural drop is point-predictable; near-boundary drop is distributional.
- **Learned world model** from point-cloud/release observations (not given exact geometry + mass).
- **Boundary focus** = the latent matters specifically around basin transitions.

## Crisp novelty (Codex)
> *We characterize **when** resting-pose prediction requires a **distributional** world model — because an **unobserved CoM**,
> not visible geometry, controls **basin selection** near stable-pose boundaries.*

Position as a **"when do we need distributions"** result (a causal criterion for distributional modeling), **not** a
CoM-estimation paper. It is niche if sold as "CoM affects drops"; it is broad if sold as "hidden physical latents create
*irreducible* multimodality only in sensitive contact regimes." **Terminology:** "hidden-latent distributional prediction",
not "belief-state" (no sequential evidence update).

## Positioning risks to pre-empt (Codex)
- **Selection optics** — CoM-sensitive subset may read as cherry-picked → pre-registered V2 criterion + full object audit +
  **CoM-insensitive negative controls** (must show NO distribution advantage).
- **Prior-art optics** — classical part-feeding / stable-pose tools already know CoM affects landing → acknowledge, then
  distinguish the *learned, hidden-latent, near-boundary, point-vs-distribution* result.
- **Synthetic optics** — hidden CoM must be a clearly-labeled controlled causal intervention (per-hull physical density = a
  later realism layer).
- **Cross-object** transfer failure = an open generalization problem, not a failed mechanism.

**Verdict:** gap real + defensible + worth the scale-up.
