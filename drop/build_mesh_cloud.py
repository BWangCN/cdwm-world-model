"""Item 1 (CoACD end-to-end): the WM's geometry-consistent input cloud. Sample points from the REAL MESH surface
(the SAME aligned mesh whose CoACD decomposition the simulator collides), so sim geometry and WM input derive from
one mesh. Codex: sample from the real-mesh surface, NOT the CoACD surface (CoACD seams change sampling density and
the WM could learn simulator artifacts). Stored as gateb_coacd_clouds/<obj>.npz {mu, sig2} in the aligned-mesh
frame (same frame as the stored hull_center/axes in the CoACD episodes).

    python build_mesh_cloud.py --obj 011_banana            # one object
    python build_mesh_cloud.py --all                       # all 29 mesh objects (objlist_mesh.txt)
"""
import os, argparse
import numpy as np, trimesh
from scipy.spatial import cKDTree
HERE = os.path.dirname(os.path.abspath(__file__))
from drop.hull_vs_coacd import align_mesh
OUT = os.path.join(HERE, "gateb_coacd_clouds")
N_PTS = 8192


def build(obj, n=N_PTS, seed=0):
    m = align_mesh(obj)                                                  # real mesh in the aligned (hull) frame
    rng = np.random.RandomState(seed)
    mu, _ = trimesh.sample.sample_surface(m, n, seed=seed)              # uniform surface samples
    mu = np.asarray(mu, np.float32)
    d, _ = cKDTree(mu).query(mu, k=2)                                   # nearest-neighbour spacing -> isotropic width
    sig2 = float(np.median(d[:, 1]) ** 2)                              # scalar isotropic covariance for build_feats
    return mu, sig2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj"); ap.add_argument("--all", action="store_true"); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    objs = [l.strip() for l in open(f"{HERE}/objlist_mesh.txt")] if a.all else [a.obj]
    for o in objs:
        if not o: continue
        mu, sig2 = build(o)
        np.savez(f"{OUT}/{o}.npz", mu=mu, sig2=np.float32(sig2))
        print(f"[{o}] {len(mu)} surface pts, sig={sig2**0.5:.4f}  -> {OUT}/{o}.npz")


if __name__ == "__main__":
    main()
