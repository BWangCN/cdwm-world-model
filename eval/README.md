# Eval scripts
Reproduce the base-vs-arm per-band dir_err comparison against the matched floor.
`eval_arm.py` selects each seed's checkpoint by held-out dir_err then scores per band+shape; `compare_arms.py` scores the pre-registered criteria vs `floor_summary.json`.
**Paths** at the top of these files (`--wm_root`, `D`) point to training-run output dirs — set them to your `wm_runs/`.
