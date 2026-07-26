# Two Gate B extensions — CoACD end-to-end geometry & CoM-from-observation (plan, 2026-07-25)

User approved proceeding on both. Codex design-checked (high level). Scope, design, and readouts below.

## Item 1 — CoACD end-to-end geometry (close the hull scope limitation)
**Claim to test (ABSOLUTE, not hull-vs-CoACD comparison):** the Gate B mechanism — distribution ≫ point, and hidden CoM is causal (grounded_oracle > no_latent) with cross-object transfer — **survives on more faithful collision geometry**, when sim AND world-model input are geometry-consistent. (Codex: don't call CoACD ground truth; "more faithful multi-convex collision.")

**Scope:** the **29** Gate B objects that have local meshes (same set as `hull_vs_coacd`). 61 of 90 are cloud-only, out of scope.

**Design (confound-controlled, reuse `hull_vs_coacd.py` machinery):**
- **Physics:** CoACD multi-hull decomposition of the real mesh (`build_coacd` / `align_mesh` / `settle_robust` — the 3 CoACD gotchas already solved: orientation-based settle, hull-vertex-mean centering, direct basin_diff). SAME explicit-inertial CoM offset distribution, SAME near-boundary release sampler, SAME axis convention as the hull generator.
- **WM input cloud:** re-sampled from the **REAL MESH surface** (trimesh `sample_surface`), NOT the CoACD surface — Codex: CoACD seams/part boundaries change sampling density and the model could learn simulator artifacts; real-mesh surface is the clean consistent input. (End-to-end consistency = sim collides CoACD-of-real-mesh, WM sees real-mesh-surface samples; both derive from one real mesh.)
- **Arms (minimal first cut, 3):** `point` / `no_latent` diffusion / `grounded_oracle` (per-point vector-to-CoM). Add `abstract_oracle` / `shuffled` / cross-object transfer only if the core ordering survives.
- **Split:** object-disjoint over the 29; paired per-object deltas; **object-bootstrap CIs** (N=29 is small — no aggregate-episode-only significance).

**Readout:** object-disjoint NLL, ECE, basin accuracy; dist>point and grounded>no_latent per-object paired deltas with bootstrap CI. Report settle rate / ambiguous-orientation / timeout / exclusion counts **per object** (settle behavior changes under multi-hull).

**Pitfalls (Codex):** target distribution shifts (hull vs CoACD NOT directly comparable — hence absolute claim); settle failures change by geometry (report them); CoM identical across arms; cloud-density leakage (→ real-mesh surface sampling); N=29 small (bootstrap over objects); normals consistency if used.

**Pipeline:** new `gateb/generate_coacd.py` (CoACD physics gen) → real-mesh cloud cache → CoACD-aware `GateBDS` variant (loads CoACD episodes + real-mesh cloud) → `train_gateb` 3 arms → `eval_gateb` + per-object bootstrap.

## Item 2 — CoM estimation from observations (drop the oracle)
**Goal:** infer the hidden CoM POSTERIOR from OBSERVED drops and beat the no_latent marginal, approaching the oracle. Novelty: a world model that **adapts to an object instance's hidden mass distribution from a few observed contact outcomes**.

**Framing (Codex-confirmed minimal):** FEW-SHOT, fixed-CoM instance. CoM is hidden but STABLE per object instance; multiple drops reveal it. (A single-shot trajectory-posterior is a different, dynamics-based problem — deferred.)

**Data (must regenerate — current data samples fresh CoM per episode):** "object instances" = (object, FIXED CoM); each instance gets M drops at varied releases. Store per instance so a k-shot set is drawable.

**Model:** amortized posterior `q(c | {(release_i, basin_i)}_{i≤k})` via a **DeepSets** set-encoder (observations are exchangeable — DeepSets before any transformer). Output = a POSTERIOR over CoM (axis, delta), NOT a point estimate (basin is many-to-one in CoM → multimodal posterior).

**Predictive procedure (the readout target is predictive, not CoM error):**
1. infer q(c | k observed drops)
2. sample CoM hypotheses from q
3. feed each through the EXISTING grounded per-point vector-to-CoM latent WM
4. marginalize the WM predictions for the (k+1)-th drop

**Readout — the one plot:** **NLL vs k** (number of observed drops), with `no_latent` (k=0 / no-evidence lower bound) and `grounded_oracle` (true-CoM upper bound) as references; object-bootstrap CI; ECE companion panel (guard against improving NLL by overconfidence).

**Pitfalls (Codex):** fresh-CoM data can't answer this (regenerate fixed-CoM instances); report the FULL adaptation curve over k (don't cherry-pick a favored k); **posterior collapse** — verify posterior entropy DECREASES with k and sampled CoMs actually change predictions; release informativeness varies (measure, don't hand-pick informative drops); split by object; principal-axis sign/symmetry ambiguities (predictive NLL first, CoM-coordinate error secondary/misleading).

## Item 2 — IMPLEMENTATION UPDATE (2026-07-25): analytic Bayesian posterior, NO data regen
Key data finding that reshaped the build: the existing hull Gate B episodes ALREADY contain fixed-CoM instances — the generator caches the sim model by a CoM quantized to 3 mm, so grouping episodes by `(object, com_key=(axis, round(delta,3mm)))` yields ~100 drops of the SAME hidden CoM at varied releases. **843 such instances (≥13 drops) across the 19 held-out objects.** No regeneration, no fixed-CoM data-gen needed.

So the sharpest first cut is training-free: **analytic Bayesian CoM posterior using the frozen grounded WM as the likelihood** `P(basin | CoM, release)`. Over a CoM grid (axis × delta), Bayesian-update on k observed drops of an instance: `p(CoM|obs) ∝ ∏ P(basin_i|CoM,rel_i)`; predict the query by marginalizing over the posterior grid. Bounds: `no_latent` arm (separate marginal) and the exact-CoM point oracle. `com_infer.py` (+ `slurm/com_infer.sbatch`). The amortized DeepSets encoder Codex suggested is a faster FOLLOW-UP; the analytic posterior is simpler AND the information-theoretic ceiling, so it leads.

**★ VALIDATION (job 1105950, 5 objects, exact oracle):** adaptation curve is clean and monotonic —
`no_latent 0.780 | k=0 0.602 | k=1 0.580 | k=2 0.532 | k=4 0.512 | oracle(point) 0.636`; posterior entropy 3.21→3.16→3.06 (sharpens with k, no collapse). Two bonus findings: (1) even the uniform-CoM-marginalized grounded WM (k=0, 0.602) beats the separately-trained no_latent model (0.780); (2) k=4 (0.512) SURPASSES the point oracle (0.636) — marginalizing the CoM posterior is better-calibrated than committing to one CoM. Full 19-object run + object-bootstrap CIs: job 1105951.

## Execution order
Item 1 first (concrete, closes a known limitation, reuses existing CoACD machinery), then Item 2 (analytic Bayesian posterior over existing instances). Both pipelines built + validated; full runs in flight. Log final results here as each lands.

## ★★ FINAL RESULTS (2026-07-25)

### Item 1 — CoACD end-to-end geometry: the Gate B mechanism SURVIVES on more faithful geometry
Data: `generate_coacd.py` (29 objects, CoACD physics + mesh basins, ~2500 attempts/obj, settle 38–100%), real-mesh clouds (`build_mesh_cloud.py`), `CDWM_GATEB_SRC=coacd` dataset path (28 objects after hammer exclusion; object-disjoint 16/5/7). Train `slurm/train_gateb_coacd.sbatch` (job 1105944, 3 arms, 120 ep). Eval `eval_coacd.py` (job 1105954, N=60 samples, mesh-stable-pose basins, object-bootstrap over the 7 test objects).

| arm | ALL NLL | top-1 | cover | ECE | BNDRY NLL |
|---|---|---|---|---|---|
| point | 1.447 | 0.56 | 0.56 | 0.166 | 1.783 |
| no_latent (distribution) | 1.056 | 0.63 | 0.96 | 0.045 | 1.608 |
| grounded (hidden CoM) | **0.822** | **0.71** | 0.97 | 0.050 | **1.281** |

- **dist>point: +0.357 NLL [+0.195, +0.539] SIG (7/7 objects).**
- **grounded>no_latent (hidden CoM causal): +0.242 NLL [+0.179, +0.291] SIG (7/7 objects).**
- The CoM-causal gain is LARGER on CoACD (+0.242) than on the single hull (+0.184, transfer study) — faithful multi-hull geometry has more stable resting modes (mesh n_stable 2–8), so the hidden CoM decides the basin MORE often. The mechanism is not a hull artifact; it strengthens under faithful geometry. (NLL magnitudes are higher than the hull task because the target has more basins — the GAINS are the claim; target shift means hull-vs-CoACD numbers are not directly comparable, per Codex.)
- Absolute claim CONFIRMED: distribution>point and hidden-CoM-causality both hold, significantly, when sim AND WM input are geometry-consistent (CoACD physics + real-mesh cloud).

**SPECIFICITY arms on CoACD (job 1105967 train, 1105974 eval) — full arm story replicated on faithful geometry.** grounded 0.821 | no_latent 1.057 | abstract 1.028 | shuffled 1.054 | point 1.447. grounded>no_latent +0.245 [+0.179,+0.296]; grounded>abstract **+0.212** [+0.156,+0.254]; grounded>shuffled **+0.236** [+0.177,+0.291] — all SIG 7/7. abstract (1.028) and wrong-CoM shuffled control (1.054) BOTH collapse to no_latent (1.057): the effect requires the CORRECT geometry-grounded CoM, not extra channels — exactly the hull finding, now on faithful geometry. Ported code (eval_coacd) handles all 5 arms.

### Item 2 — CoM from observations: the WM adapts to an instance's hidden CoM from a few drops
`com_infer.py` (job 1105953): analytic Bayesian posterior over a CoM grid, frozen grounded WM as likelihood, 843 fixed-CoM instances on the 19 held-out objects. N_SAMP=40, grid axis×11δ, KS=[0,1,2,4,8], N_QUERY=4, MAX_INST=8.

| | predictive NLL |
|---|---|
| no_latent arm (separate marginal) | 0.756 |
| k=0 (grounded WM, uniform-CoM prior) | 0.680 |
| k=1 | 0.655 |
| k=2 | 0.631 |
| k=4 | 0.597 |
| k=8 | **0.558** |
| oracle (exact CoM, point) | 0.630 |

- **Paired adaptation gain k=0→k=8: +0.120 NLL [+0.049, +0.231] SIG (17/19 objects).**
- Posterior entropy monotonically sharpens 3.39→3.29→3.11→2.79 with k (no collapse — Codex's check passes).
- k=8 (0.558) SURPASSES the point oracle (0.630): marginalizing the CoM posterior is better-calibrated than committing to one CoM.
- Even k=0 (grounded marginalized over uniform CoM, 0.680) beats the separately-trained no_latent arm (0.756).
- Headline: a world model that ADAPTS to an object instance's hidden mass distribution from a few observed contact outcomes — Bayesian inference under the WM likelihood, no oracle, no retraining.

**Follow-ups (deferred):** Item 2 amortized DeepSets encoder (faster, learned q(CoM|obs)); Item 1 extend to cloud-only objects (needs meshes) + more arms (abstract/shuffled/transfer). Both are candidate paper sections; consult colleague on framing.

### Codex sign-off (2026-07-25)
Both results judged SOUND, no over-reach. Recommended 3 lead claims: (1) CoM-conditioning improves prediction under faithful multi-hull geometry → the effect is NOT a single-hull artifact; (2) the frozen grounded WM can infer useful posterior beliefs over the hidden CoM from a handful of observed drops; (3) Bayesian marginalization over inferred CoM can OUTPERFORM plug-in of a single CoM under model misspecification (better NLL + calibration).
Caveats to state explicitly: CoACD-vs-hull absolute NLLs are NOT directly comparable (basin space changes); the 7-object CoACD test is strong but small-sample; frame Item 2 as **transductive / test-time instance adaptation** on held-out objects (same-object few-shot), NOT zero-shot object generalization.
Item 2 anti-leak / robustness suite to run before writeup (Codex): (a) leave-one-query-out per instance; (b) ensure no support/query release DUPLICATES, or stratify by data-derived release similarity; (c) ECE for no_latent/k=0/k=8/oracle; (d) posterior mass/rank around the true CoM vs k; (e) THE KEY CONTRAST — posterior-marginal predictive vs MAP/mean-CoM PLUG-IN (if marginal beats plug-in while calibration improves, the "marginalization helps" story is solid; if MAP also beats the oracle, scrutinize the oracle/likelihood eval). Also soften "exact oracle" → "matched CoM" (group key is 3mm-quantized).
Amortized encoder NOT required for the scientific claim (systems extension only).

### Amortized encoder — INFORMATIVE NULL (jobs 1105971 timed out / 1105987 capped): learns the prior, not the adaptation
`com_amortized.py`: DeepSets over {(release,basin)} + PointNet object embedding → MDN posterior over the 3D CoM offset; trained on 52 train objects' instances (MDN NLL to true offset), eval predictive NLL vs k through the frozen grounded WM. RESULT (52 obj train, 40 ep): NLL k=1 0.672 / k=2 0.666 / k=4 0.659 / k=8 **0.675** — FLAT; paired gain k=1→k=8 −0.003 [−0.030,+0.024] ns (11/19). Absolute NLL ~0.67 beats no_latent (0.744) and ≈ the analytic k=1 (0.655), so it DOES learn a useful geometry-conditioned CoM PRIOR — but it does NOT ADAPT to observations (no k-dependence), whereas the analytic Bayesian method does (0.678→0.566). Precise finding: the geometry→CoM-prior is amortizable; the observation→CoM-UPDATE (the adaptation) is not, with this DeepSets-MDN approach. Sharpens Item 2's framing: the contribution is specifically the ADAPTATION, delivered by the analytic WM-as-likelihood method. (Also: the full eval is prohibitively slow — WM-in-the-loop over 870 instances timed out at 2h; capped to 5 inst/obj for the number above.) Not pushed (null; code stays in dev).

### Item 2 robustness suite — DONE (job 1105956), all Codex checks pass
Paired gain replicated: k=0→k=8 **+0.111 NLL [+0.045, +0.212] SIG (17/19 obj)**.
- **Marginal vs MAP plug-in (the key contrast):** marginal beats MAP at EVERY k (k=8: 0.566 vs 0.600; k=0: 0.678 vs 0.834), and keeps calibration (ECE k8-marginal 0.043 vs k8-MAP 0.065). → "marginalization helps" confirmed on NLL AND calibration.
- **Both marginal (0.566) and MAP (0.600) beat the point oracle (0.639)** — NOT a bug: model-misspecification (Codex claim 3). Inference recovers the WM's EFFECTIVE CoM (best explains observed basins under the imperfect WM), which predicts better than the true PHYSICAL CoM fed as a point. State honestly; do not headline "beats oracle."
- **Posterior mass on the true-CoM grid cell rises with k: 0.03 → 0.05** (uniform=1/33≈0.03) — the posterior concentrates correctly toward the true CoM.
- **Calibration excellent throughout** (ECE ≈ 0.043 for no_latent / k0 / k8-marginal).
- **Leakage guard: 0/600** query drops excluded for release-duplication (<5°) — no support/query release leakage. Frame as transductive test-time instance adaptation on held-out objects (per Codex).
Soften "exact oracle" → "true-CoM point estimate" in writeups (group key is 3 mm-quantized).
