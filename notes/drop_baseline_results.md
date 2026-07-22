# Baseline drop WM — results (2026-07-21)

The deterministic point-prediction reference on the existing (uniform-density) corpus, against which the CoG /
distributional contribution is measured. Reuses the slip `local` encoder; predicts **basin_transition** (tip vs stay)
and **resting orientation** (rest6d, rel. release) from (release-frame cloud + release params). Trained on the
object-disjoint split (58 train / 10 val / 12 held-out-mesh test), array job `1105057`/`1105100` (V100, `qos=batch`).
Eval vs a **no-motion** baseline (predict identity rotation) — the honest bar. Code: `train_drop.py`, `my_dataset_drop.py`,
`wm/drop_net.py`, `eval_drop.py`.

## Three-arm comparison (held-out test, n=19,881, 12 objects, 44 basin-transition positives)

| arm | basin-transition AUROC | resting-orientation geo median | reads |
|---|---|---|---|
| **full / release** (cloud + release params) | **0.973** | **2.38°** | the baseline |
| pose_only / release (release params, cloud zeroed) | 0.548 | 14.56° | control: is the cloud needed? |
| full / obj-frame (object-canonical cloud) | 0.971 | 4.59° | ablation: does the release frame help? |
| — no-motion (predict identity rotation) | — | 7.45° | the honest bar |

## Findings (object-level bootstrap 95% CIs on the winner)

1. **Beats no-motion (unlike the grasp baseline).** geo median **2.38° [1.78, 3.82]** vs no-motion **7.45° [7.18, 7.63]**
   — non-overlapping CIs. The drop task is a largely well-posed deterministic mapping (release orientation strongly sets
   the resting pose), so a point predictor is *adequate* on the natural corpus — exactly what the CoG stress tests
   predicted (the interesting failure is the near-boundary / hidden-CoM regime, not this one).
2. **The cloud is essential — no release-parameter shortcut.** Zeroing the cloud (pose_only) collapses basin-transition
   AUROC to **0.548 ≈ chance** and makes resting-orientation error (14.56°) *worse than no-motion*. Object geometry
   carries essentially all the signal (same clean result as the slip pose-only control).
3. **The release frame helps** the rotation prediction: **2.38° (release) vs 4.59° (object-frame)** — the frame
   canonicalization roughly halves the error, the drop analogue of the grasp gripper-frame win (smaller, ~2× not 3×).
   It barely changes basin-transition AUROC (0.973 vs 0.971) — the cloud handles tip/stay in either frame; the frame
   matters for the finer rotation regression.
4. **Basin-transition is predictable** despite rarity: AUROC **0.973 [0.898, 0.977]** (object-bootstrap; 634/1000
   resamples had both classes — the CI is honest about the 44-positive sparsity, but the lower bound 0.90 ≫ chance).

## Caveats / next
- 44 held-out positives over 12 objects → basin-transition CIs are wide by construction; a larger held-out set or a
  transition-enriched eval split would tighten them.
- Point-prediction only (the baseline's job). The CoG contribution (Gate B) adds the distributional head where hidden
  CoM makes the resting pose multimodal (V4) — measured against *this* reference.
- Rotation reported as median geodesic; add the stratified `wm/metrics.py` bands (0–2/2–5/5–15/15–30/30+) for the tail.
