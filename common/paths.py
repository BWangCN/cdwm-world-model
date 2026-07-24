"""Centralized data-root resolution (single source of truth for every external + repo-relative path).
Public defaults are repo-relative (`./data/<dataset>`); override any root via its `CDWM_*` env var to point at a
shared / scratch copy. A move or rename touches ONE file, not the whole tree.

The DROP models reuse the GRASP dataset's point clouds + meshes (the world model sees the same clouds), and the
baseline drop WM reads the drop corpus. Fetch both from HuggingFace (BWangCN/cdwm-grasp-dataset,
BWangCN/cdwm-drop-corpus) into `./data/` or point the env vars at an existing copy.
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))    # parent of common/
_DATA = os.environ.get("CDWM_DATA", os.path.join(REPO, "data"))      # dataset root (repo-relative default)

# --- external datasets (override via env) ---
GRASP_ROOT  = os.environ.get("CDWM_HF",      os.path.join(_DATA, "cdwm-grasp-dataset"))
PCDIR       = os.environ.get("CDWM_PCDIR",   os.path.join(GRASP_ROOT, "outcomes_v2", "point_clouds"))
OBJECTS     = os.environ.get("CDWM_OBJECTS", os.path.join(GRASP_ROOT, "objects"))
OUTCOMES    = os.environ.get("CDWM_OV",      os.path.join(GRASP_ROOT, "outcomes_v2"))
DROP_CORPUS = os.environ.get("CDWM_DROP",    os.path.join(_DATA, "cdwm-drop-corpus"))
DERIVED     = os.path.join(DROP_CORPUS, "derived")

# --- repo-relative ---
GATEB_DIR  = os.path.join(REPO, "drop", "gateb")                      # compact gateb episodes
MODELS     = os.path.join(REPO, "drop", "models")                    # released drop checkpoints
NORM_STATS = os.path.join(MODELS, "norm_stats")                      # shipped feature/target stats


def resolve_report():                                                # path-resolution check for the smoke gate
    return {k: (v, os.path.exists(v)) for k, v in
            dict(GRASP_ROOT=GRASP_ROOT, PCDIR=PCDIR, OBJECTS=OBJECTS, DROP_CORPUS=DROP_CORPUS,
                 GATEB_DIR=GATEB_DIR).items()}
