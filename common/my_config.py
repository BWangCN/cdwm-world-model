"""Portable config for the cross-object colorless WM. Data dir via env CDWM_DATA (default ./example_data)."""
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
RUN = _HERE                                                    # feat_stats.npz lives at repo root
DATA_DIR = os.environ.get("CDWM_DATA", os.path.join(_HERE, "example_data"))
CLUSTERS = os.path.join(DATA_DIR, "clusters")                  # gaussians_<obj>_clean.npz (colorless 12-feat; = HF package point_cloud.npz)
CHUNKS_CSV = os.path.join(DATA_DIR, "chunk_index", "chunks_corrected.csv")
DIR_WM = os.environ.get("CDWM_WM", os.path.join(_HERE, "wm_runs"))
H = 32
N_FEAT = 12                                                    # colorless: mu(3)+rot_quat(4)+scale(3)+opacity(1)+is_completed(1)
os.makedirs(DIR_WM, exist_ok=True)
