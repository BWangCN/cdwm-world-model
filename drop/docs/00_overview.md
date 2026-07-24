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

## Model-imagined rollout (extension)
Beyond the endpoint, a rollout variant (`GateBDiT` with **H=16**, trained on re-simulated settling trajectories) lets the model **imagine the settle itself**. The top row below is the MuJoCo ground truth; the rest are samples from the trained model for the *same* held-out release, each descending and tumbling per the model's *predicted* orientation trajectory. Under a near-boundary release the samples **diverge into different basins** — the same distribution the endpoint model captures, now made temporal. This is genuine model output (contrast the physics-counterfactual demos above).

![model rollout vs ground truth: three samples diverge into different basins](../figures/roll_pred_triceratops.gif)

Pipeline: `gen_rollout` (trajectory targets from `com_sim`) → `my_dataset_roll` (RollDS, K=16 target) → `train_roll` (`GateBDiT` H=K) → `render_roll_pred`.

## Reproduce
```bash
python -m drop.train_gateb   --arm grounded_oracle --tag grounded_oracle   # train an arm
python -m drop.eval_transfer                                               # cross-object transfer eval
python -m drop.eval_transfer_hardening                                     # per-object breakdown + ECE
MUJOCO_GL=egl python -m drop.render_drop_demo                              # regenerate the demo videos
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
