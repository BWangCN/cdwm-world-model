# Drop-phase demo videos — the hidden CoM decides the basin

These clips illustrate the drop-phase thesis directly: for a near-boundary release, an object's resting **basin** is
controlled by an **unobservable inertial latent** (the center of mass), not by the visible geometry alone. Each video
holds the object and its release pose fixed and sweeps the hidden CoM offset along a principal axis; the object then
settles into a different resting basin. This is why a point predictor is inadequate here and a distributional world
model is required (the quantitative claims and confidence intervals live in `gateb_results.md` and
`cross_object_transfer_plan.md`; these videos are a controlled illustration of the mechanism, not a measurement).

Rendered with MuJoCo (2.3.7, EGL headless) from the point-cloud convex hull that the world model also sees; the CoM
enters as an explicit inertial offset (mass and inertia held fixed, so the CoM is the only varied quantity). Regenerate
with `MUJOCO_GL=egl python render_drop_demo.py` (SLURM: `slurm/render_demo.sbatch`, job 1105467).

| Object | Video | CoM sweep (-35% / -15% / +15% / +35% of half-extent) | Resting basin |
|--------|-------|------------------------------------------------------|---------------|
| Banana | [`011_banana_com_sweep.mp4`](demo_videos/011_banana_com_sweep.mp4) | offset along the long axis | **1 → 0 → 0 → 1** |
| Medium clamp | [`050_medium_clamp_com_sweep.mp4`](demo_videos/050_medium_clamp_com_sweep.mp4) | offset along a principal axis | **1 → 1 → 1 → 2** |
| Gaming mouse | [`Razer_Taipan..._com_sweep.mp4`](demo_videos/Razer_Taipan_Black_Ambidextrous_Gaming_Mouse_com_sweep.mp4) | offset along a principal axis | **1 → 1 → 0 → 1** |

Each clip concatenates the four CoM conditions from the same release, captioned in-frame with the offset (mm) and the
resulting basin. Because the release pose is identical across the four drops, any change in the resting basin is due
solely to the hidden CoM, which the point cloud does not reveal.

**To fold into the paper write-up:** these are relative-path links; keep the `demo_videos/` folder next to the write-up
markdown, or convert to GIFs if the venue needs inline playback.
