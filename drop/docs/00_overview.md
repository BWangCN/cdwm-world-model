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
Both the simulator and the world model operate on the object's point-cloud **convex hull**, not a detailed mesh. This is deliberate and, for most of the corpus, forced: of the 89 Gate B objects **60 are point-cloud-only** (no source mesh exists), so the hull is the only geometry that can be built for them; a rigid-body simulator collides convex geometry natively; and the hull is matched to the world model's input cloud, so the model and the simulator share one shape representation with no geometry mismatch between them.

A data-derived audit (`drop.concavity_audit`) over the 29 objects that do have meshes measures how good this approximation is, with two proxies. The **global convexity ratio** (mesh volume over hull volume) has median 0.52 and a natural gap near 0.74. The **contact-facing proxy** counts stable resting modes of the real mesh versus its hull (`trimesh.compute_stable_poses`): the hull loses a **median of only 1 mode** (mean 5.3 vs 4.4), so for most objects it preserves the resting structure; it collapses modes only for a strongly concave minority, where the effect is large (the Hereford bull: 8 modes to 3; a therizinosaurus: 8 to 4). The two proxies correlate only weakly (-0.11): a low volume ratio does not by itself flag the problem objects (a thin plate has ratio 0.06 but loses no modes), whereas the stable-pose proxy does. So the hull is a faithful stand-in for most of the set and fails specifically on legged or handled shapes.

The learned effect **does not appear to be driven by this approximation**. Stratifying the cross-object transfer gain (no-latent NLL minus grounded NLL) by hull fidelity, the grounded-CoM benefit **remains positive in every audit-defined stratum**: hull-faithful objects **+0.204**, hull-unfaithful **+0.320**, point-cloud-only **+0.165** (overall +0.181). The mesh-having held-out cells are small (3 faithful, 1 unfaithful object), so this is a **robustness check, not a powered subgroup claim**; the gain does not collapse on any subset. Separately, the model-free I(basin;CoM) evidence concerns the hull's own dynamics, a distinct line of support that the mesh mismatch does not bear on.

The honest limitation is **qualitative resting fidelity for concave real objects**. For an object with legs or a handle the hull rests on faces the real object never rests on, so rendering the detailed mesh at a hull-predicted pose looks unnatural even when the predicted hull basin is correct. The figure below shows this for the bull. Faithful resting prediction for such objects is out of the current scope. The natural extension is convex decomposition (CoACD) on the mesh-having subset, with the world model's input cloud **re-sampled from the same decomposed geometry** so the model and simulator stay geometry-consistent; reconstructed meshes for point-cloud-only objects would be a separate, clearly labeled approximation layer.

![scope boundary: the bull rests differently on its real mesh versus its convex hull](../figures/scope_boundary.png)
*The convex hull collapses the bull's 8 stable resting modes to 3. The simulator and world model reason about the hull (right), a legless blob; the detailed mesh (left) is the object we recognize but never enters the model. Concave objects are out of scope for faithful resting; see the audit.*

## Model-imagined rollout (extension)
Beyond the endpoint, a rollout variant (`GateBDiT` with **H=16**, trained on re-simulated settling trajectories) lets the model **imagine the settle itself**. Below, for three **held-out** objects, each row is: the MuJoCo **ground truth** (left), the trained model's **imagined** settle for the same release (middle), and the model's basin samples (right). Under a near-boundary release the samples **diverge into different basins** — the same distribution the endpoint model captures, now made temporal. Genuine model output (contrast the physics-counterfactual demos above). The three objects (a curl-lotion bottle, a conditioner bottle, a skillet lid) are **convex-ish, hull-faithful** shapes chosen from the concavity audit above, so the textured mesh rests naturally at the predicted hull pose; the scope-boundary figure shows why strongly concave objects (the bull) are excluded.

![drop rollout grid: three held-out objects, ground truth vs model imagination, samples diverge into different basins](../figures/drop_grid.gif)

Pipeline: `gen_rollout` (trajectory targets from `com_sim`) → `my_dataset_roll` (RollDS, K=16 target) → `train_roll` (`GateBDiT` H=K) → `render_grid` (textured-mesh GT-vs-model grid).

## Reproduce
```bash
python -m drop.train_gateb   --arm grounded_oracle --tag grounded_oracle   # train an arm
python -m drop.eval_transfer                                               # cross-object transfer eval
python -m drop.eval_transfer_hardening                                     # per-object breakdown + ECE
python -m drop.concavity_audit                                             # convex-hull scope audit (two proxies)
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
