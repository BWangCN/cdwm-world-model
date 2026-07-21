# Corpus audit — how much of the drop corpus is near-boundary / metastable?

`audit_corpus_boundary.py` over all **153,996 valid** train episodes (of 160,200; 96.1% SETTLED). Near-boundary proxy =
`basin_transition` (object ends in a **different** stable basin than release = it crossed a boundary). Lower bound:
episodes that teetered and fell back are not counted.

## Headline: the corpus is overwhelmingly *inside* stable basins

- **basin_transition = 2.3%** (3,505 / 153,996). `net_rot_from_release`: **p50 = 7.7°** (≈ the release tilt → object just
  returns to its release basin), p90 = 14°, only **2.1% tip >30°**, 0.5% > 90°.
- This is the dataset-level confirmation of the hammer pilot: **the corpus's stable-pose + U[0,15°] sampling is
  shape-dominated; CoG is screened off for ~98% of episodes.**

## But the near-boundary tail is real and CONCENTRATED

- Rises with tilt: 1.3% (0–5°) → 2.3% (5–10°) → **3.2% (10–15°)**. More tilt → nearer boundaries.
- Extremely object-dependent — a few tall/asymmetric shapes tip constantly, most never do:

| tips a lot (near-boundary already) | rate | ~never tips (deep-stable → negative controls) | rate |
|---|---|---|---|
| Android_Figure_Chrome | 45% | 005_tomato_soup_can | 0.0% |
| Razer_Naga_Gaming_Mouse | 44% | Schleich_African_Black_Rhino | 0.0% |
| **003_cracker_box** | 40% | Thomas_Friends_Woodan_Railway_Henry | 0.0% |
| Quercetin_500 (pill bottle) | 31% | bottle-can shape_class (28k eps) | 0.0% |
| Prostate_Optimizer | 21% | handled shape_class (5.6k eps) | 0.0% |

- **72 of 80 objects transition <1% of the time.** By shape_class: `other` 21%, `?` 7.6%, concave 2.1%, convex 1.5%,
  bottle-can **0.0%**, handled **0.0%**.

## What this means for the plan

1. **Confirms CoG is a NO-GO on the corpus as-sampled** (2.3% near-boundary) — you cannot study a CoG latent where 98% of
   episodes are geometry-determined. This is the empirical backing for "co-design a near-boundary sampler."
2. **Gives V2 its object list for free:** the ~5 high-transition shapes (cracker_box, Android figure, Razer mouse,
   pill bottles) are where near-boundary dynamics *already* occur → CoG-sensitive candidates; the 0.0% shapes
   (tomato_soup_can, Schleich animals, bottle-can class) are ready-made **negative controls**.
3. The high-transition objects are also a cheap way to test CoG **without** a new sampler first: upweight/condition on
   their existing near-boundary episodes and check whether CoG changes their outcome, before building anything.
