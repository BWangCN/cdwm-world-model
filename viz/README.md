# Visualization scripts
`analyze_direction.py` is standalone (dir-error distributions/plots). `render_direction_video.py` and `select_and_rollout.py` render 3DGS+MuJoCo rollouts and **depend on our simulation harness** (`make_videos`, `build_grasps`, `config_aug`, `dist_harness`) which is **not** included here — they are provided as reference for how the direction-annotated videos were produced.
