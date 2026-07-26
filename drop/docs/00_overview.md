# Drop-Phase World Model

> 中文说明（设计示意图 + 我们相比 v1 的改动 + 88 物体清单）见 [`../exploratory/hidden_com_zh.md`](../exploratory/hidden_com_zh.md). Exploratory hidden-CoM extension (Gate B, our 88-object set): [`../exploratory/hidden_com.md`](../exploratory/hidden_com.md).

The task, per the corpus formulation (`BWangCN/cdwm-drop-corpus`): a world model of **where an object comes to rest after being released above a table** — object released → free-fall → impact → settle. The corpus records the full trajectory (world-frame pose at 31.25 Hz); the **learned target is the terminal resting pose** (its resting SE(3), and which stable **basin** it settles into), predicted from the object point cloud (the same Gaussian-cluster assets as the grasp WM) plus the release parameters — the fall itself is near-deterministic ballistics, so the modeling weight goes where the physics is interesting: the settle outcome.

This overview covers the **corpus task** — the primary deliverable on the colleague's v1 dataset (natural releases, CoACD collision, uniform density). The natural corpus is largely a deterministic geometric mapping where a point predictor suffices; the **diffusion rollout below** handles the multimodal tail (near a stability boundary an object can tip either way) by imagining the settle as a *distribution*. A separate, exploratory extension that deliberately constructs a hidden-CoM multimodal regime (Gate B, our own 88-object set) lives in [`../exploratory/hidden_com.md`](../exploratory/hidden_com.md).

Shared encoder throughout: `common.dit_local.DiTWMLocal` (the grasp encoder); only the head changes. `drop_net` is the point/regression head, `drop_diffusion.GateBDiT` the DiT diffusion head.

## Baseline drop WM (`drop.train_drop` / `drop.eval_drop`)
On the natural corpus the mapping is well posed. Held-out test (object-disjoint):
- resting-orientation error **2.38 degrees vs a no-motion predictor at 7.45** (non-overlapping CIs; unlike grasp, the point model beats no-motion), basin-transition AUROC 0.973.
- the **cloud is essential** (pose-only is at chance), and canonicalizing into the **release frame** roughly halves the rotation error versus the object frame.

So on the corpus as sampled, a point prediction is adequate for the bulk. The interesting failure is the multimodal tail (which way a near-boundary object tips), addressed by the diffusion rollout.

### Does supervising the settle trajectory help? (`drop.train_droptraj`)
A natural question is whether predicting **intermediate frames** — not just the endpoint — improves the resting-pose prediction. One reduction first: under the corpus's zero-velocity kinematic release, orientation is *constant* during free-fall and height is closed-form ballistics, so during-fall frames carry **no learning signal**; intermediate supervision meaningfully means the **contact-onset → settle transient**. We tested exactly that: same encoder, split, sampler, and protocol as the baseline, only the rotation head changed to K=8 anchor orientations spanning contact onset (detected as the first orientation deviation, which cannot occur before contact) to settle, equal chordal loss, endpoint read from the last anchor. Replicated across two seeds (paired per-episode dumps, object-bootstrap over the 12 held-out objects):

| arm | endpoint median | ≥15° tail median (n=313) | paired per-object gain vs baseline |
|---|---|---|---|
| `full_release` (H=1 endpoint baseline) | 2.38° | 16.51° (≈ no-motion 16.05°) | — |
| `traj_k8` seed 0 (K=8 transient supervision) | **2.01°** | 16.31° | endpoint **+0.28° [+0.08, +0.79] SIG** (10/12 objects) · tail +0.59 [−0.04, +2.42] ns |
| `traj_k8` seed 1 (replication) | **1.91°** | 21.74° | endpoint **+0.33° [+0.13, +1.07] SIG** (10/12 objects) · tail +0.74 [−0.88, +1.90] ns |

**Transient supervision significantly improves the endpoint, replicated**; it does **not** improve the ≥15° tail in either seed — the pooled tail median even swings between seeds (16.3° / 21.7°, both ≈ the 16.0° no-motion floor), itself evidence that a deterministic head lands arbitrarily on the which-way question. The tail failure is *which-way multimodality* — the distributional rollout's job, not supervision density. `traj_k8` is the preferred endpoint-head candidate (`drop/models/baseline/traj_k8.pt`); `full_release` remains the reference baseline.

## Primary demo: the corpus task (`drop.render_corpus_demo`)
The formulation, end to end, on three **held-out** objects. Left: the **recorded corpus trajectory replayed** — real free-fall, impact, and settle from `obj_pos`/`obj_quat` at 31.25 Hz, no re-simulation. Middle: the endpoint WM's **predicted resting pose**, computed from the cloud and release parameters alone — before the drop happens. Right: errors, as-is.

![corpus demo: recorded free-fall/impact/settle vs the model's predicted resting pose](../figures/drop_corpus_demo.gif)

Selection policy (uniform, disclosed): basin transitions exist for exactly one held-out object — the supplement bottle (44 tips; the corpus is shape-dominated) — which shows its *first* transition episode; the other objects have no transitions at these releases and show their *median*-net-rotation episode (a typical settle). The trio is representative of the per-object medians: 9 of 12 held-out objects are clear wins like the cube (1.9° vs 7.8°) and apple (4.6° vs 7.5°), and 2 are weak — the toy airplane (median 8.2°, tying no-motion) and the supplement bottle shown here as the honest tail: the point model cannot predict which way it tips (72.6°, tying no-motion), which is precisely the failure that motivates the diffusion rollout below. Objects rest on their natural flat sides here because these are **natural corpus releases with the corpus's CoACD physics**.

## Corpus diffusion rollout: imagining the settle distribution (`drop.train_droproll`) ★
The endpoint demo above ends on the supplement bottle as the honest failure of a *point* model: it cannot predict which way a near-boundary bottle tips. This is the direct answer to that failure — a **distributional rollout** that imagines the settle as a *sample-able distribution*, not a single pose. It changes exactly one thing versus the trajectory arm: the head. The `traj_k8` head deterministically regresses the K=8 contact→settle anchors; here a **`GateBDiT` diffusion head (H=8)** *denoises* the same K=8 anchors, so the comparison isolates the head family (deterministic → distributional) with everything else — encoder, split, anchors — held fixed. No latent is used (the corpus has no hidden per-episode CoM; the multimodality is intrinsic to shape + release).

Results (test set, DDIM sampling): validation diffusion loss **0.027**; the endpoint read off the final anchor lands within **0.52°** for the best of S and **0.71°** for the median sample — the distribution concentrates tightly wherever the outcome is unimodal. The distributional payoff is on the multimodal tail: on the supplement bottle's tip episodes, **at least one of S samples reaches the tip mode 78% of the time** (data-derived tip cutoff), and at S=8 **34 of 36 (94%)** of its tip episodes produce a sample distribution that *straddles* the cutoff — some imagined futures settle, some tip. This is exactly the ≥15° tail the deterministic heads provably tie no-motion on: a which-way *multimodality* the distributional head resolves by covering both outcomes.

![corpus diffusion rollout: same three held-out objects, recorded settle vs two of S=16 imagined settles; on the supplement bottle the imagined futures diverge — most settle, a minority tip, covering the tip that actually happened](../figures/drop_roll_corpus.gif)

Same three held-out objects as the primary demo. Per row: **left** = the recorded corpus trajectory replayed (real free-fall/impact/settle, no re-simulation); **middle ×2** = two of **S=16** diffusion-sampled imagined settles. This is genuine model output, with two honesty conventions stated in-frame: the deterministic ballistic **fall is prepended in the render only** — the model does not generate the fall (zero-velocity release ⇒ constant orientation + closed-form height), it generates the contact→settle anchors — and the tip/settle **cutoff is data-derived** (largest gap in the object's ground-truth net-rotation distribution, ≈47°). The cube and apple settle in **every** sample (0/16 tip), matching their unimodal outcome; the supplement bottle **diverges** — 13/16 imagined futures settle back, 3/16 tip — the same release yielding different futures, and the distribution **covers the tip that actually happened**. The displayed pair is one settling + one tipping sample when both exist; the full S=16 tally is in the remark. Fixed seed for reproducibility.

Pipeline: `precompute_droptraj` (K=8 contact→settle anchors) → `DropRollDS` (same anchors, formatted as the DiT target) → `train_droproll` (`GateBDiT` H=8, diffusion loss; test = DDIM best-of-S / median-of-S / tip-recall / divergence) → `render_droproll_demo` (recorded settle vs S imagined settles).

## Data (corpus)
**Corpus (`BWangCN/cdwm-drop-corpus`): 80 unique objects** (108 object×config combinations; 153,996 settled episodes). **CoACD collision + uniform density** (400 kg/m³ proxy; measured total mass for 9 YCB) → **CoM = the CoACD volume centroid** (geometrically realistic; no per-object override). **Natural releases** (a stable pose + ≤15° tilt, 5–80 mm height), so the corpus is shape-dominated (objects mostly settle back). Used by the baseline endpoint WM, the trajectory-supervision arm, and the corpus diffusion rollout.

## Scope and limitation
The drop WM predicts a **terminal-state distribution** over the *final resting pose / basin*, not the free-fall / impact / settling *trajectory* frame-by-frame (the corpus diffusion rollout imagines the K=8 contact→settle anchors, not the fall — which is near-deterministic ballistics). In the CDWM family, grasp is an H=32 closing-trajectory WM and slip a K=10 lift-onset rollout WM. The corpus assumes **uniform density**; a material-density realism study (see the exploratory doc) found that matters for only a narrow subset of objects (standable containers with variable contents), so uniform density is a benign assumption for the corpus task as a whole.

## Reproduce
```bash
MUJOCO_GL=egl python -m drop.render_corpus_demo                            # PRIMARY demo (corpus replay vs predicted rest)
python -m drop.precompute_droptraj                                         # K=8 contact->settle anchors (trajectory arm)
python -m drop.train_droptraj --tag traj_k8                                # trajectory-supervision arm (endpoint +0.3 deg)
python -m drop.train_droproll --tag roll_corpus                            # ★ corpus DIFFUSION rollout (distributional settle; GateBDiT H=8)
MUJOCO_GL=egl python -m drop.render_droproll_demo                          # recorded settle vs S imagined settles (divergence demo)
```
Exploratory (hidden-CoM Gate B) reproduce lines: see [`../exploratory/hidden_com.md`](../exploratory/hidden_com.md).

## Terminology
- **no-motion** — trivial baseline: predict the resting pose equals the release pose. The honest floor any model must beat.
- **basin** — one stable resting pose of the object (which face it settles onto).
- **NLL / Brier** — proper scores over the predicted basin distribution (lower better).
- **ECE** — expected calibration error: gap between predicted confidence and actual accuracy (near 0 = well-calibrated).
- **object-disjoint split** — train and test share no objects, so cross-object claims are about genuinely unseen geometry.
- **best-of-S / median-of-S** — over S diffusion samples per episode, the min / median endpoint error (the diffusion rollout is distributional, so it is scored over samples).
