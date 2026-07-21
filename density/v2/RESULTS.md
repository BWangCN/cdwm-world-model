# V2 — object-diversity CoG-sensitivity (which objects' resting pose depends on center of gravity?)

Two tools, converging answer:
- `drop_sweep.py` — paired uniform-vs-CoM-shifted MuJoCo drops (dynamical). **Superseded as the primary metric**:
  arbitrary-object basins are noisy to define from drops (low settle rates, yaw/symmetry confounds). Kept for
  dynamical confirmation on a few objects.
- `stable_pose_sensitivity.py` — **analytic + rigorous + whole-corpus**. `trimesh.compute_stable_poses` takes the CoM
  as input, so we compare the resting-pose distribution at the geometric centroid vs a plausible CoM offset
  (0.35 × half-extent) along each principal axis. Metric = TV distance between the two distributions (max over axes),
  with a degeneracy guard (near-round objects rest anywhere → flagged, not counted).

## Result (558 objects; 74 flagged near-round)

CoM-sensitivity is a **shape** property, and it separates cleanly:

| CoM-ROBUST (tv_max ≈ 0.02) | CoM-SENSITIVE (tv_max ≈ 0.9–1.0) |
|---|---|
| plates, DVD/game cases, tablets | spoon, banana, scissors, spatula, clamp |
| flat objects: exactly **2 stable poses** (front/back), ~50/50, CoM-invariant | Razer mouse, hair straightener, Schleich bull |
| a plate lands flat regardless of CoM | elongated/asymmetric: CoM flips which end/face rests down |

- **Non-degenerate tv_max: median 0.23, 41% > 0.3, 24% > 0.5.** A substantial subpopulation (elongated / asymmetric)
  has a genuinely CoM-dependent resting distribution — **the effect is NOT hammer-only** (V2's goal). The hammer pilot
  generalizes. (Numbers after the degeneracy-flag fix below; the clean top list = banana, scissors, spatula, mouse,
  clamp, spoon, bull, hair straightener, pitchers.)
- **Confirms Codex's warning — high corpus-transition ≠ CoM-sensitive.** `003_cracker_box` tips in **40%** of corpus
  episodes yet is **CoM-ROBUST** (tv_max ≈ 0.1): a box always rests on one of its 6 faces; the CoM shift only reweights
  them. Corpus transition rate is a *shape/stability* signal; CoG-sensitivity is a *distinct* one. Target objects by
  CoG-sensitivity (this ranking), not by transition rate.

## Verdict

**GO for a CoG contribution, targeted at elongated/asymmetric objects** — the CoM-sensitive subpopulation is real and
sizable (~a quarter to a half of objects), across many distinct shapes. Combined with the pilot + audit: CoG matters (a)
for CoM-sensitive shapes and (b) near stability boundaries; it is negligible for flat/round objects and for the corpus's
gentle stable-pose sampling.

## Caveats
- This is the **quasi-static resting-pose prior** (drop from uniform-random orientation) = the CoG-sensitivity **ceiling**.
  The corpus's stable-pose + U[0,15°] sampling realizes far less of it (audit: 2.3% transitions) — hence the need for a
  near-boundary sampler to actually *elicit* the sensitivity (V3).
- Round objects (orange, baseball, apple) are now correctly **flagged and excluded** by a two-part degeneracy test:
  `n_eff > 8` (many near-equal poses) **or** `isotropy = min/max half-extent > 0.7` (near-spherical → rests anywhere).
  Verified: orange (isotropy 0.98), baseball (0.99), apple (0.92) flagged; spoon (0.13), clamp (0.21), plate (0.12) kept.
- TV on down-normal histograms is coarse; `drop_sweep.py` on the top/bottom objects is the dynamical confirmation.
