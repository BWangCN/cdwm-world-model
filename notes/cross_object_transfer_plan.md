# Cross-object latent transfer — study plan (2026-07-22)

The drop-phase Gate B open problem, now being explored (user-chosen; the one high-impact remaining item).

## The problem (grounded)
The resting **basin** (which stable pose the object settles into) depends on the **visible** state (geometry + release
pose) and a **hidden** inertial latent (the CoM / mass distribution). Established results (`notes/gateb_results.md`):
- For **known** objects, the hidden CoM is *usable*: oracle-latent > no-latent is significant (+0.017; also model-free
  I(basin;CoM)=0.165 on CoM-sensitive).
- For **unseen** objects, the true CoM is *useless*: oracle > no-latent is **ns cross-object** (+0.009, eval 1105263).

Why: the mapping **CoM → basin is object-geometry-specific.** "Mass shifted +2 cm along axis 1" only determines the basin
if you know how that shift meets *this* object's stable-pose structure (faces / tipping edges / saddles). A head-heavy
hammer tips toward its head; a head-heavy pair of scissors does something else. The model learns per-object CoM→basin
combinations and can't apply them to a novel shape.

**Goal:** get the latent to help on *unseen* objects — learn a general `f(geometry, hidden-CoM) → basin distribution` that
transfers. Metric = **oracle-latent > no-latent on the object-disjoint split** (currently +0.009 ns → target: significant),
object-bootstrap CI; keep the known-object (reldisjoint) result as the control.

## Root cause: how the latent is fed
Currently the CoM latent is an **abstract offset vector** `[axis_idx, delta]` added to the conditioning; the model must
*learn* to bind it to geometry (works per-object, doesn't transfer). We already feed geometry (the cloud); the fix is to
feed the latent **geometry-grounded** so the encoder perceives *where the mass sits within the shape* (object-agnostic).

## Approaches (cheap → deep)
1. **Data lever (cheapest sanity):** does *more object diversity* alone close the oracle-transfer gap? (retrain with more
   generated objects.) Tells us if it's a coverage problem vs a representation problem.
2. **★ Geometry-grounded latent encoding (first real attempt):** feed the CoM as a **spatial location co-registered with
   the cloud** — e.g. a per-cloud-point feature = the point's position/distance relative to the CoM. The PointNet encoder
   then sees "the heavy region is *here* in the shape," which transfers. (oracle = true CoM location; no-latent = feature
   zeroed/absent. A pure representation change, no new architecture.)
3. **Stable-pose-structured conditioning (deeper):** encode the CoM relative to the object's stable-pose graph (distance to
   tipping edges / the saddle it sits nearest) — a physics-structured inductive bias.

**First study:** approach 2 (geometry-grounded CoM feature) — highest leverage, cheap (edit `build_feats_drop` to add a
CoM-relative per-point feature; retrain the 3 arms object-disjoint; re-eval oracle-transfer). Fall back to 1 (more data)
and 3 (stable-pose structure) if 2 is insufficient.

## Paper framing (which paper does this belong to)
- **Folds into the drop paper** if a modest fix (approach 1 or 2) closes the gap → the current *limitation* ("latent
  causal for known objects, doesn't transfer") becomes a *result* ("...and generalizes to novel objects"). Same thesis,
  strictly stronger.
- **Could anchor a separate methods paper** if a substantial geometry-structured inductive bias (approach 3 / an
  identification framework) is required — because such a method is **cross-cutting** (applies to grasp + slip + drop, the
  whole CDWM family), so drop stays the regime-characterization paper and the transfer method stands alone.
- We won't know which until we try; the cheap data-lever + representation tests disambiguate "data/representation" (fold
  in) vs "needs a real new inductive bias" (potential new paper) early.
