# Drop-Phase World Model

Predict where a released rigid object comes to rest (its resting SE(3), and which stable **basin** it settles into) from the object point cloud plus release parameters. The drop regime completes the CDWM story: the natural corpus is largely a deterministic geometric mapping (a point predictor suffices), while near a stability boundary a hidden center-of-mass makes the outcome multimodal and a distributional world model becomes necessary.

Shared encoder throughout: `common.dit_local.DiTWMLocal` (the grasp encoder); only the head changes. `drop_net` is the point/regression head, `drop_diffusion.GateBDiT` the DiT diffusion head.

## Baseline drop WM (`drop.train_drop` / `drop.eval_drop`)
On the natural corpus the mapping is well posed. Held-out test (object-disjoint):
- resting-orientation error **2.38 degrees vs a no-motion predictor at 7.45** (non-overlapping CIs; unlike grasp, the point model beats no-motion), basin-transition AUROC 0.973.
- the **cloud is essential** (pose-only is at chance), and canonicalizing into the **release frame** roughly halves the rotation error versus the object frame.

So on the corpus as sampled, a point prediction is adequate. The interesting failure is the near-boundary, hidden-CoM regime below.

## Gate B: distributional WM (`drop.train_gateb` / `drop.eval_gateb`)
On 88 CoM-sensitive objects at near-boundary releases with a per-episode **hidden CoM** (a controlled explicit-inertial offset, mass and inertia held fixed so the CoM is the only varied quantity):
- **distribution >> point**, significant and large, and it **transfers** to unseen objects; the diffusion model is **well calibrated (ECE 0.015 vs the point predictor 0.227)**.
- the hidden CoM is **demonstrably causal**, three independent ways: model-free mutual information I(basin;CoM) 0.165 on CoM-sensitive vs 0.046 on negative controls; an oracle-latent model that is significant and specific to CoM-sensitive objects (null on controls); per-object distribution>point on 88/88.

Arms: `point` (regressor) | `diff` (no-latent diffusion) | `diff_oracle` (diffusion conditioned on the true CoM).

## Cross-object latent transfer (`drop.eval_transfer` / `drop.eval_transfer_hardening`)
Does knowing the CoM help on an object never seen in training (object-disjoint split, 19 held-out objects)? The CoM-to-basin mapping is object-geometry-specific, so an abstract `[axis, delta]` encoding does **not** transfer. Feeding the CoM **geometry-grounded** (for every cloud point, the vector to the CoM plus distance) does:
- grounded > no-latent **+0.184**, grounded > abstract **+0.191**, grounded > shuffled-CoM **+0.160** (all significant, paired object-bootstrap CIs).
- broad (grounded beats the baselines on **18/19** objects) and well calibrated (grounded ECE 0.019). The shuffled-CoM control rules out extra-channel capacity.

Four arms: `no_latent` | `abstract_oracle` | `grounded_oracle` | `shuffled_oracle`. The latent is the oracle CoM supplied at test; estimating the CoM from observations is future work.

## Physical-density realism (`drop.multi_physical_density`)
Validation that the controlled explicit-inertial offset is a faithful proxy for real mass distribution, not a synthetic artifact. Across 6 CoACD-decomposed real-mesh objects, a randomized heavy-end per-hull **physical density** causally controls the basin in every object (paired near-boundary drops), reproducing the hammer result (I(basin;density) 0.161 vs the controlled-offset 0.165). Effect size tracks shape and is reported as a heuristic-sampled lower bound.

## Demos: the hidden CoM decides the basin (physics counterfactuals, NOT model output)
> These clips are **MuJoCo ground-truth re-simulations, not drop-model rollouts.** They sweep the hidden CoM to show the *causal mechanism* behind the learned basin distribution (same object + release, different hidden CoM → different resting basin). The drop WM predicts the terminal resting pose, so it has no settling video of its own; these motivate *why* the outcome is multimodal and are **not samples from the diffusion model**.

Rendered from the point-cloud hull via `drop.render_drop_demo`; per-object basin references from `drop.render_basins`.

![banana CoM sweep](../figures/011_banana_com_sweep.gif)
*Banana: basin 1 → 0 → 0 → 1 across the four swept CoM conditions.*

![medium clamp CoM sweep](../figures/050_medium_clamp_com_sweep.gif)
*Medium clamp: basin 1 → 1 → 1 → 2.*

![gaming mouse CoM sweep](../figures/Razer_Taipan_Black_Ambidextrous_Gaming_Mouse_com_sweep.gif)
*Gaming mouse: basin 1 → 1 → 0 → 1.*

Basin indices are the object's stable resting poses ranked by probability (see `../figures/*_basins.png`).

## Data
The near-boundary hidden-CoM episodes live under `drop/gateb/*_gateb_s0.npz` (compact: release quaternion, drop height, hidden CoM `[axis, delta]`, resting quaternion, basin). This 13 MB summary **deterministically regenerates the full trajectories** from `com_sim` (the quantized CoM the simulator used) given the point clouds and MuJoCo 2.3.7; `drop/gateb/verify_gateb.py` confirms a 100% basin match. See `drop/gateb/README.md`.

## Scope and limitation
The drop WM is a **terminal-state distributional world model**: it predicts a calibrated distribution over the *final resting pose / basin*, not the free-fall / impact / settling *trajectory*. In the CDWM family, grasp is an H=32 closing-trajectory WM and slip a K=10 lift-onset rollout WM; drop shares the same encoder but a single-step (H=1) head. Rolling out the settling dynamics for drop — so the model *imagines* the settle and diverges into different basins under different CoM — is a natural extension, not required for the hidden-CoM claim (the free-fall is near-deterministic; the multimodality is in the settle outcome). A trained rollout extension is shown next.

### Geometry: the convex-hull scope
Both the simulator and the world model operate on the object's point-cloud **convex hull**, not a detailed mesh. This is a deliberate scoped choice. Two of the reasons are intrinsic: a rigid-body simulator collides **convex** geometry natively (a non-convex mesh must first be convex-decomposed), and the hull is matched to the world model's input cloud, so the model and the simulator share one shape representation with no geometry mismatch between them. The third reason is practical, not fundamental: our local mesh set covers only the 29 Gate B objects that are also GRASP-project objects; the other 60 are standard Google-Scanned-Objects / YCB items whose meshes are **publicly available** but were not downloaded and standardized for this study. So the hull is a scoped modeling choice for most of the corpus, **not a data impossibility** (the meshes exist upstream); obtaining them for all objects is future work.

**Global shape audit.** A data-derived audit (`drop.concavity_audit`) over the 29 objects that do have meshes measures the approximation with two global proxies. The **global convexity ratio** (mesh volume over hull volume) has median 0.52 and a natural gap near 0.74. The **contact-facing proxy** counts stable resting modes of the real mesh versus its hull (`trimesh.compute_stable_poses`): the hull loses a **median of only 1 mode** (mean 5.3 vs 4.4); it collapses modes mainly for a strongly concave minority (the Hereford bull: 8 modes to 3; a therizinosaurus: 8 to 4). These are honest measurements, but — as the boundary-regime study below shows — they answer "does this object look right at rest in general," not "does the hull distort the specific near-boundary episodes Gate B trains and evaluates on." The two questions turned out to have different answers.

**Boundary-regime measurement (`drop.hull_vs_coacd`).** For the same 29 objects, we re-simulated the actual Gate B near-boundary episodes with a CoACD convex-decomposition of the real mesh in place of the single hull — identical release and identical hidden CoM in both arms, so only the collision geometry differs — plus a 1.5°-release-wobble control that isolates ordinary near-boundary sensitivity from a geometry effect. Result: the release-wobble floor is high (agreement 0.95), confirming these episodes are genuinely boundary-sensitive; on top of that floor, **faithful geometry flips the resting basin 16–19% more often than the wobble alone**, even for objects the global proxy calls hull-faithful (convex-ish subset: **+0.16** gap; strongly concave: **+0.38**). So the global stable-pose proxy **under-states** hull sensitivity in the regime that matters: two bottles it rated "hull-faithful" turned out to have large boundary gaps (+0.31, +0.13), while `051_large_clamp` — which the global proxy flagged as too concave and which the first demo pass excluded — has one of the **smallest** gaps in the set (+0.05). This mirrors an earlier lesson from the density study (transition rate ≠ CoM-sensitivity): a global shape statistic does not predict in-regime dynamical sensitivity; only a direct test in the trained regime does.

**Does the Gate B mechanism survive this?** Yes. Re-stratifying the core Gate B result (distribution beats point, hidden CoM is causal; known-object reldisjoint split, 6,758 episodes over the 28 objects with both measurements) by the boundary-regime geometry gap instead of the global proxy: dist>point gain is positive and stable across low/mid/high-gap thirds (**+0.473 / +0.520 / +0.425**), and the CoM-causality gain (oracle beats no-latent) is if anything **largest in the high-gap third** (+0.033 vs +0.023 low-gap); the per-object correlation between geometry gap and dist>point gain is weak (−0.27). The mechanism does not concentrate in, or depend on, the hull-sensitive objects. What the hull approximation costs is the **fidelity of the predicted resting pose for a geometry-sensitive minority**, not the qualitative claim that a hidden latent drives multimodal outcomes.

**Limitation, stated precisely.** For a meaningful share of objects — not only the visibly concave ones — the single convex hull is not a faithful stand-in for the object's true near-boundary contact dynamics; roughly 1 in 5–6 boundary episodes would rest differently under faithful geometry. This is a genuine scope limitation on the *predicted resting pose*, not on the *distribution-over-point / hidden-latent-causality* claims, which hold under the more faithful geometry. The natural fix is convex decomposition (CoACD) integrated end-to-end — the world model's input cloud **re-sampled from the same decomposed geometry** used by the simulator — which remains future work; the boundary-regime study here is a diagnostic, not yet a fix.

![scope boundary: the bull rests differently on its real mesh versus its convex hull](../figures/scope_boundary.png)
*The convex hull collapses the bull's 8 stable resting modes to 3 — an extreme, easily visible case of the geometry effect measured above. The simulator and world model reason about the hull (right), a legless blob; the detailed mesh (left) is the object we recognize but never enters the model.*

## Model-imagined rollout (extension)
Beyond the endpoint, a rollout variant (`GateBDiT` with **H=16**, trained on re-simulated settling trajectories) lets the model **imagine the settle itself**. Below, for three **held-out** objects, each row is: the MuJoCo **ground truth** (left), the trained model's **imagined** settle for the same release (middle), and the model's basin samples (right). Under a near-boundary release the samples **diverge into different basins** — the same distribution the endpoint model captures, now made temporal. Genuine model output (contrast the physics-counterfactual demos above). The three objects (a skillet lid, a large clamp, a conditioner bottle) were picked by the **boundary-regime geometry gap** above (0.03 / 0.05 / 0.13, the three lowest of the 29 measured), not the global convexity proxy — an earlier pass picked two bottles the global proxy called hull-faithful that turned out to have large boundary gaps (0.31, 0.13); this selection is corrected accordingly.

![drop rollout grid: three held-out objects, ground truth vs model imagination, samples diverge into different basins](../figures/drop_grid.gif)

Pipeline: `gen_rollout` (trajectory targets from `com_sim`) → `my_dataset_roll` (RollDS, K=16 target) → `train_roll` (`GateBDiT` H=K) → `render_grid` (textured-mesh GT-vs-model grid).

## Reproduce
```bash
python -m drop.train_gateb   --arm grounded_oracle --tag grounded_oracle   # train an arm
python -m drop.eval_transfer                                               # cross-object transfer eval
python -m drop.eval_transfer_hardening                                     # per-object breakdown + ECE
python -m drop.concavity_audit                                             # global convex-hull scope audit (two proxies)
python -m drop.hull_vs_coacd --obj <object> --n 250 --out <out.json>       # boundary-regime hull-vs-CoACD gap (per object)
python -m drop.aggregate_hvc                                               # aggregate the 29-object hull_vs_coacd runs
MUJOCO_GL=egl python -m drop.render_drop_demo                              # regenerate the demo videos
# rollout WM (imagine the settle; K=16 trajectory targets regenerate from com_sim, ~75 MB, not shipped):
python -m drop.gen_rollout   --obj <object>                                # settling-trajectory targets -> gateb/<obj>_roll_s0.npz
python -m drop.train_roll    --arm grounded_oracle                         # train the rollout WM (GateBDiT H=16)
MUJOCO_GL=egl python -m drop.render_roll_pred                              # model rollout vs ground truth (held-out objects)
MUJOCO_GL=egl python -m drop.render_grid                                   # textured GT-vs-model grid (convex held-out objects)
MUJOCO_GL=egl python -m drop.render_scope                                  # scope-boundary figure (bull: real mesh vs convex hull)
```

## Terminology
- **no-motion** — trivial baseline: predict the resting pose equals the release pose (the object does not move). The honest floor any model must beat.
- **basin** — one stable resting pose of the object (which face it settles onto); the `basin` field is the id of the nearest stable pose (see the demos and `*_basins.png`).
- **Gate B** — the core distributional-WM experiment on CoM-sensitive objects at near-boundary releases (Gate A was the earlier density/CoM feasibility study).
- **NLL / Brier** — proper scores over the predicted basin distribution, lower is better: negative log-likelihood of the true basin, and squared error of the basin probabilities.
- **ECE** — expected calibration error: the average gap between predicted confidence and actual accuracy (near 0 = well-calibrated).
- **I(basin; CoM)** — mutual information; a model-free measure of how much the hidden CoM determines the basin.
- **arms** — `point` (regressor) · `no_latent` / `diff` (distribution, no CoM) · `abstract_oracle` (CoM as an `[axis, delta]` vector) · `grounded_oracle` (CoM as per-point vector-to-CoM + distance) · `shuffled_oracle` (grounded features but a wrong CoM, the control).
- **object-disjoint split** — train and test share no objects, so cross-object claims are about genuinely unseen geometry.
