# Density / center-of-gravity workstream

Artifacts for the drop-phase object-density challenge. Plan + rationale: [`../notes/drop_density_todo.md`](../notes/drop_density_todo.md)
(Chinese colleague brief: [`../notes/drop_density_plan_zh.md`](../notes/drop_density_plan_zh.md)). Gated & off the baseline
drop critical path.

## Files
- `object_inventory.csv` — 80 drop-corpus objects (24 YCB + 56 GSO): current sim mass (proxy/measured), obb, shape_class,
  held_out, and **real_mass_kg** for all 24 YCB.
- `ycb_mass_elpis.json` — 78 YCB real masses (kg), from [`elpis-lab/ycb_dataset`](https://github.com/elpis-lab/ycb_dataset)
  (`scripts/ycb_mass.json`). Same repo ships **79 ready MuJoCo XMLs** with per-object 5-hull CoACD collision meshes.
- `test_density.py` — T3 mechanism self-check (runnable, asserts). **PASSES.**

## What's done (Gate A, T0/T1/T3)
- **T1 masses (YCB): 24/24 matched.** The 400 kg/m³ proxy is **off >15% for 18/24** — e.g. hammer real 0.665 vs proxy
  0.333 (2×), cups real ~13–35 g vs proxy 50–190 g, foam_brick 28 g vs 79 g. Mass realism is genuinely wrong, *but* mass ≠ CoG.
- **T3 mechanism proven** (`test_density.py`): uniform density → CoM pinned at the centroid **regardless of the value**
  (1000 vs 5000 kg/m³ both give CoM_x = 0.0000, mass scales 5×); heterogeneous per-geom density → CoM moves to the exact
  analytic mass-weighted centroid (8000/1000 → −0.389). Confirms: per-category *uniform* density = zero CoG diversity;
  **per-CoACD-hull density is the lever.**

## Resource note for T2/T3 (existing MJCFs)
`elpis-lab/ycb_dataset` `ycb/<obj>.xml` = ready MuJoCo model + 5 CoACD hulls per object, but it overrides inertia with an
explicit `<inertial pos="0 0 0" mass=... diaginertia="0.001 0.001 0.001"/>` — **placeholder CoM at the origin + isotropic
inertia**, i.e. mass-correct but CoG-fake. For the CoG study we **drop that `<inertial>` and set per-geom `density`** so
MuJoCo derives a real, hull-weighted CoM. These XMLs unblock a real-object pilot without waiting on the colleague's
`assets573`.

## GSO (56 objects)
Left at proxy for now (`confidence=proxy(400kg/m3)`, flagged `TODO: GSO metadata / material prior`). Precise GSO masses are
a **Gate-B** item; the Gate-A A/B pilot uses YCB objects (hammer / can / clamp) which have real masses.

## Next
- **T2** pilot MJCFs (pull elpis-lab hammer hulls, or colleague `assets573`).
- **T4** A/B pilot (go/no-go): uniform vs CoM-shifted resting-pose distribution on ~4–6 high-leverage shapes.
