# Object selection for CDWM grasp data generation

Constraints on which objects are suitable for the grasp world model + the
`gs-native-datagen` pipeline (parallel-jaw Robotiq 2F-85, tabletop scenes).

## Exclude flat objects (e.g. plates)

**Rule (project lead, 2026-07-31):** do **not** include flat objects such as
**plates**. Their viable grasp contact points sit only a **short distance above
the tabletop**, so a parallel-jaw gripper cannot seat on them cleanly — the jaws
would have to close almost at table level, which does not fit the gripper.

This matches what our own feasibility analysis found when the plate was carried
as a test object (see `notes/vla_pipeline.md`, now archived, and the corpus
report): the plate is **side-/edge-grasp only** (no top-down grasp on a flat
disk), the grasped rim forces the FR3 wrist through a strained, joint-limit-bound
region during transport, and a bottom corner dips below the table when the disk is
tilted. It was correctly **rejected by the feasibility gate** and kept only as an
informative negative. The lead's remark generalizes that finding into a dataset
rule rather than a per-episode rejection.

**Practical criterion.** Skip an object when its stable grasp band lies within
roughly one gripper-finger height of the support surface, or when the object
affords no approach that keeps the jaws clear of the table (flat disks, thin
lids, shallow trays). Objects with graspable structure standing well above the
table (cans, boxes, bottles, bottle necks, tool handles) are fine.

## Object inventory

- **WM training set:** the `cdwm-grasp-dataset` (171 objects) — the grasp world
  model was trained across these.
- **`gs-native-datagen` released example scene:** **2 objects / 2 categories** —
  `003_cracker_box` (box) and `005_tomato_soup_can` (can), both YCB. These are the
  two segmented 3DGS clusters and the two grasp banks
  (`config_assets/grasps_env1/kept_00{3,5}_*.json`) shipped with that repo. New
  scenes add objects via its recon + segmentation upstream, subject to the
  flat-object exclusion above.
