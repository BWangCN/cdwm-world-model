# Why the grasp settle WM is strong — loss, generalization, and what it actually learned

Analysis recorded 2026-08-01. Concerns the `colorless_s{0,1,2}/frame` `DiTWM` ensemble (the
model `gs-native-datagen` consumes). Evidence cited by `file:line`.

## 1. Is the optimized loss "purely diffusion"? Is the eval metric the diffusion error?

**Training loss — essentially diffusion, plus one small regularizer** (`train.py:6`):

```
loss = eps_MSE  +  lam_aux · geodesic(R(x0_hat), R(target))
```

- `eps_MSE` (`train.py:103`, `((eps_hat - noise)**2).mean()`) = standard DDPM
  **noise-prediction** loss: model sees a trajectory noised at a random diffusion timestep and
  predicts the injected noise. Dominant term.
- `lam_aux · geodesic` = small rotation-geometry regularizer pulling recovered rotation toward
  target. The pure-diffusion baseline ("stable-187") omits it.

**The eval metric is NOT the training loss — and the difference matters.**
- Training loss = noise prediction at a *random* timestep → a surrogate; its value is not
  interpretable as trajectory error.
- Eval metrics (ADE/FDE, ADD/ADD-S, geodesic) run the **full DDIM rollout** (`dit.py:79`
  `ddim_sample` → x0), un-z-score, and compare to ground truth in **physical units**
  (degrees, cm). Stricter and more meaningful. So FDE 3.16° means "the sampled prediction is
  ~3° off," not "the denoiser residual is small."

## 2. Why so strong? Overfitting? Object/trajectory counts? Real physics?

**Not overfitting — strongest single fact:** splits are **object-disjoint**
(`my_dataset_outcomes.py:90-91` asserts train/val/test object sets are disjoint). Test metrics
are therefore on **objects never seen in training** → genuine generalization.

**Scale** (`outcomes_v2/outcomes_index.csv`, `outcome_split.csv`):
- **562 objects** (YCB ~77 + Google Scanned Objects ~485).
- **53,917 episodes**; outcome split: RIGID 12,584 / TRANSIENT_SLIP 12,884 /
  PERSISTENT_SLIP 10,665 / CLOSED_NEVER_LIFTED 15,092 / LIFTED_DROPPED 2,669 /
  CLEARANCE_VIOLATION 23 (dropped).
- Trajectory WM supervises on **RIGID only = 12,584** (test rigid ≈ 1,967). Model ≈ **4.9M
  params** (`dit.py:3`). ~12k trajectories, hundreds of splittable objects, 4.9M params — the
  setup cannot memorize at that scale.

**Why it is so *accurate* (honest nuance):** the task is the tame regime by design — the
in-gripper settle of a **rigid** grasp. Once jaws close on a rigidly held object, motion over
H=32 is small and low-entropy (residual motion ~0.014 cm) and is nearly a smooth function of
the conditioning (gripper-frame cloud + grasp pose). Hard/chaotic cases (slip, drop) are
**filtered out** before the WM sees them. So it learned a **well-conditioned, low-variance
regression on the stable manifold** — which is why sub-degree / sub-mm accuracy is attainable.

**Did it learn "real physics"? Partially, narrowly, genuinely:**
- ✅ A data-driven physical regularity — local contact geometry → short-horizon rigid settle —
  that **generalizes to unseen objects** and respects support planes (ood ~0.02,
  non-penetration enforced downstream via `reject_reasons()`).
- ❌ NOT a general physics engine: no slip dynamics (excluded), no long-horizon or multi-body
  contact. It interpolates within the rigid-settle manifold it was trained on.

For the datagen purpose the narrow competence is **exactly enough**: we only need the tame
in-gripper settle; `reject_reasons()` catches anything outside the manifold. That is *why* the
pipeline works with this single component. See [[datagen_adaptation_plan]].
