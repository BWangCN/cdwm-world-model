# Drop-phase TODO — object density / center-of-gravity diversity (2026-07-21)

Branch: `drop`. Motivation from colleague discussion: the drop corpus was sim'd in MuJoCo with **uniform density**
(400 kg/m³ × convex-hull-volume proxy for 71/80 objects; measured mass for only the 9 YCB). With uniform density there is
**no center-of-gravity diversity** to study — every object's CoM sits at its geometric centroid. This doc scopes adding
realistic / diverse mass distributions so settling becomes genuinely CoM-dependent.

---

## ✅ Converged plan (self-decided → Codex-reviewed 2026-07-21, session `019f8664`)

Codex validated the physics claim and the L1/L2/L3 framing, and sharpened four things. Resolution:

1. **Target = L3, but grounded in REAL latent states** (fill level, hidden ballast, tool head/handle material, plausible
   per-object internal mass maps) — NOT arbitrary randomization, which a reviewer discounts as synthetic uncertainty.
2. **Honest scope of the claim:** a single cloud→final-pose task with hidden CoM is **aleatoric distributional
   prediction**, not true belief-state system-ID (that needs *sequential / interactive* observations to update on). So
   the defensible L3 thesis is the direct heir of the slip-MDN result: **geometry + release → a calibrated distribution
   over resting pose, whose irreducible spread is CoM uncertainty.** Do not overclaim "belief state / system-ID".
3. **Knobs:** per-hull heterogeneous density = the **main** dataset knob (physical, reuses CoACD, inertia/CoM stay coupled
   through MuJoCo). Explicit `<inertial>` CoM offset = **ablation/control** knob only (fast sensitivity sweeps).
4. **Pilot-first, OFF the critical path.** Biggest risk = burning weeks re-simulating before proving CoM shifts matter
   (many basins are shape/contact-dominated → CoM perturbation may barely move the outcome). So: (a) build the **baseline
   drop WM** on the existing corpus first; (b) run a **tiny A/B pilot** (~4–6 high-leverage objects) to prove CoM-induced
   multimodality is large *before* building any full dataset; (c) only then layer CoG diversity as a **separate
   contribution**. Gate order below reflects this.

---

## ⚠️ Load-bearing physics fact (must anchor the whole plan)

MuJoCo `compiler inertiafromgeom="true"` + a single uniform density ⇒ **body CoM = the composite-hull volumetric centroid,
independent of the density VALUE.** (Codex caveat: it is the centroid of MuJoCo's *modeled* mass geometry — the union of
the CoACD hulls, which can differ from the true visual-mesh centroid if hulls overlap or approximate unevenly — but the
density still cancels under uniform scaling.) Scaling every geom's density by the same factor scales total mass and inertia
magnitude but does **not move the CoM**. The dataset README + `admission_matrix.md` confirm empirically: mass only moved
the resting-penetration tail (heavy flat objects), it **changed no admit/reject verdict** and does not alter which way an
object tips.

**Consequence:** "update each category's (uniform) density to its real value" — the literal first reading of the task —
buys **mass realism only** (better penetration realism, correct contact-impulse magnitude). It creates **zero CoG
diversity**. CoG diversity requires **non-uniform** mass distribution.

**The clean lever:** every object is already decomposed into **≤24 CoACD convex hulls** (`assets573` MJCFs). Assigning
**different densities to different hulls** shifts the mass-weighted CoM with **no new geometry** — reuse what exists. A
heavy-head hammer = dense head-hull + light handle-hull; a bottle = dense base + light body.

## The density ladder (this is the decision fork for Codex)

| level | what | CoM effect | WM consequence | cost |
|---|---|---|---|---|
| **L0** | current: uniform 400 kg/m³ proxy | centroid | deterministic; CoM=centroid | shipped |
| **L1** | uniform **real** density per object | centroid (unchanged) | mass realism only, no CoG signal | low |
| **L2** | **fixed non-uniform** per-object internal mass map | CoM shifts, **fixed per object** | CoG diversity ACROSS objects, but still a deterministic fn of object identity → learnable point-predictor, **no hidden latent** (same "belief-state moot" verdict as grasp-rigid) | med |
| **L3** | **randomized non-uniform** — per-episode CoM/mass sampled, NOT visible in the cloud | CoM varies within one object | genuine **hidden latent** → aleatoric multimodality → resting pose is a **distribution** for fixed observation → the **belief-state / distributional WM** the grasp phase pointed at becomes *necessary* | med-high |

The stated goal ("CoG diversity to study deeply") is **L2 or L3**, not L1. L3 is where the project north star
(physics-structured belief-state contact WM + system-ID, per `execution_log.md` §10 and `cdwm-world-model` memory) actually
bites: a water bottle that is randomly full/empty looks identical in the point cloud but settles differently → the WM must
*represent* that uncertainty rather than predict a point. **Recommend surfacing L1/L2/L3 to Codex and picking the target
before building anything.**

---

## Object taxonomy (survey — DONE this turn)

80 meshes, from `manifest.json` (`mass_kg`, `obb_mm`, `shape_class` per object):

- **24 YCB-numbered** (`003_cracker_box`, `005_tomato_soup_can`, `013_apple`, `048_hammer`, `065-*_cups`, …) →
  **published measured masses** (Calli et al. 2015, YCB object&model set — every object weighed). Direct copy.
- **56 Google Scanned Objects** (`Schleich_*`, `Nintendo_*`, `Perricone_MD_*`, `Dell_*`, `Epson_*`, …) → GSO ships
  **real-world mass metadata** for scanned objects (weighed on scan). Cross-reference the GSO release.
- Mass range currently 0.011–1.36 kg (median 0.138). `shape_class`: concave 30 / convex 27 / bottle-can 15 / handled 2 /
  other-? 6. Episode `object_class`: {flat-bottom, roll-prone}.
- Only 9 YCB have *measured* mass in-sim; 71 use the 400 kg/m³×hull proxy → these are the ones to fix.

## Mass/density sourcing — where real values come from

1. **YCB (24 objs)** — official YCB benchmark mass table (per-object measured). Highest confidence.
2. **GSO (56 objs)** — Google Scanned Objects per-model metadata often includes measured mass; also `model.config`/
   thumbnails give material cues. Medium confidence.
3. **Existing MuJoCo/MJCF ports** — check robosuite YCB MJCFs, `obj2mjcf` outputs, ManiSkill/IsaacGym YCB assets,
   `mujoco_menagerie` for any ready `<geom density>` / `<inertial mass>` we can copy verbatim (the "copy XML if it exists"
   step). Note: **no MJCF ships in the HF drop/grasp datasets** — assets573 CoACD MJCFs live on the colleague's box.
4. **Deduce (fallback)** — for anything without a measured mass, a **material-density prior by category** (cardboard box
   hollow-effective ~150–250, plastic figure ~950, wood ~600, steel tool ~7800, filled can ~1000) × hull volume. This is
   the "simple analysis to deduce density" step. Sanity-check deduced mass against `obb_mm` and hand-feel plausibility.

---

## TODO list (gated order per converged plan — cheapest disproof first)

**Gate A — is CoG diversity even worth building? (do BEFORE any full dataset work)**
- [x] **T0 Survey** — object inventory + current masses. *(done; `density/object_inventory.csv`.)*
- [x] **T1 Source real masses** — `density/object_inventory.csv` filled. **24/24 YCB** matched to real masses from
      [`elpis-lab/ycb_dataset`](https://github.com/elpis-lab/ycb_dataset) (`ycb_mass_elpis.json`); proxy off >15% for 18/24
      (hammer 2×, cups way over). 56 GSO left at proxy, flagged for Gate-B metadata/priors (pilot uses YCB).
- [ ] **T2 MJCFs for the pilot objects only** — need per-object CoACD MJCFs to edit density. **Shortcut found:**
      `elpis-lab/ycb_dataset` ships ready `ycb/<obj>.xml` + 5 CoACD hulls/object (drop its placeholder `<inertial>`, set
      per-geom `density`). Else colleague `assets573`, or regen from `objects/<id>/mesh.obj` via CoACD + `obj2mjcf`.
- [x] **T3 MuJoCo density mechanics** — **PROVEN** by `density/test_density.py` (mujoco 2.3.7): uniform density → CoM at
      centroid *independent of value* (1000 vs 5000 → CoM 0.0000, mass ×5); per-geom hetero density → CoM = analytic
      mass-weighted centroid (8000/1000 → −0.389). `<geom density>` = main knob, `<inertial>` offset = ablation.
      *(Primitive 2-box stand-in; real-hull confirm folds into T2/T4.)*
- [x] **T4 A/B PILOT (the go/no-go)** — DONE on the hammer (`density/pilot/`, [`RESULTS.md`](../density/pilot/RESULTS.md)).
      **Conditional GO** (Codex-converged, session `019f8664`): an 83 mm CoM shift gives **0% basin change in the corpus's
      stable-pose+small-tilt regime** (shape-dominated, confirms README "mass nearly irrelevant") but is **decisive near
      stability boundaries** (100% basin disagreement balanced-on-end, 50% at 5° perturbation). **CoG matters only where
      dynamics are sensitive to it** → adding densities to the *existing* corpus alone changes ~nothing; CoG becomes a
      hidden latent only if release sampling also visits **near-boundary / metastable** poses.

**Gate A.5 — 4 stress tests BEFORE any dataset engineering (Codex)**
- [x] **V1 out-of-plane sensitivity** — FOLDED into V2: the sweep offsets CoM along **all three** principal axes
      (long/mid/short), so out-of-plane (mid/short) shifts are covered. Confirmed the basin-relevant direction is often
      lateral, not along the long axis.
- [x] **V2 object diversity** — DONE (`density/v2/`, [`RESULTS.md`](../density/v2/RESULTS.md)). Pivoted from noisy drop
      sims to the **analytic** `stable_pose_sensitivity.py` (trimesh; CoM is an input). Over 558 objects: CoM-sensitivity
      is a **shape** property — flat objects (plates/cases) CoM-ROBUST (tv≈0.02, 2 stable poses); elongated/asymmetric
      (spoon/scissors/clamp/mouse/bull) CoM-SENSITIVE (tv≈0.9–1.0). **median tv_max 0.28, 28% >0.5 → not hammer-only.**
      Key: **`003_cracker_box` tips 40% in corpus but is CoM-ROBUST** → transition rate ≠ CoG-sensitivity (Codex's warning
      confirmed); target objects by *this* ranking. Negative controls (plates) behave. `drop_sweep.py` kept for dynamical
      confirmation.
- [ ] **V3 boundary-sampler validation** — PARTLY IN PLACE: `compute_stable_poses` gives the stable-pose graph analytically
      (used in V2). Still to do: a sampler that emits releases near the **saddles between adjacent stable poses** and shows
      those poses have high CoM-sensitivity under tiny perturbations. Refine the degeneracy flag (round-object edge case).
- [ ] **V4 distributional necessity** — identical visible observation → genuinely multimodal rest under hidden mass maps,
      and a **distributional** predictor beats a point predictor on **likelihood/calibration** (not just mean error).
      *(Go/no-go metric = paired basin-transition probability near objectively-derived boundaries, NOT avg rest-angle.)*

**Gate B — only if Gate A.5 confirms the effect is general + physical**
- [ ] **T5 Physically-derived near-boundary + heterogeneous-density dataset** — density maps grounded in real latent
      states (fill level, head/handle material, base-heavy containers, ballast); L3 = sample the latent per episode
      (not visible in the cloud). Release from V3's boundary sampler. **Report BOTH** natural/stable and boundary
      distributions; paired counterfactuals; negative controls. Honest claim = "belief-state matters in the *subset* of
      contacts where hidden inertia controls basin selection", NOT "drops are always multimodal".
- [ ] **T6 Re-simulation** — colleague's corpus-v1 harness, or scale up the validated mini-harness.
- [ ] **T7 CoG-aware distributional WM** — geometry + release → a **calibrated distribution** over resting pose (heir of
      slip-MDN), irreducible spread = CoM uncertainty; energy score / CRPS, never point-only. Separate contribution,
      **after** the baseline drop WM.

## Sequencing (where this sits vs the baseline drop WM)

Baseline drop WM (outcome classification + resting-pose prediction on the *existing* L0 corpus) stays the **critical
path** — see `drop_phase_log.md` §5. This density/CoG track is **parallel and gated**: T1–T4 (Gate A) can run cheaply
alongside the baseline; T5–T7 only unlock if the pilot proves CoM-induced multimodality is real and physically
defensible. Do not block the baseline on it.
