#!/usr/bin/env python3
"""Self-contained inference demo for the cross-object colorless WM. Loads a base700 checkpoint and predicts how each
bundled example grasp settles/tilts in the gripper, vs the simulated ground truth. Runs on GPU if available, else CPU.

    python infer_example.py

Uses the mini dataset in ./example_data (2 objects, 6 grasps). No external data needed."""
import os, sys, csv
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("CDWM_DATA", os.path.join(HERE, "example_data"))     # my_config reads this
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "eval"))
import numpy as np, torch
import my_config as C
from wm.common import rot_angle_axis, geodesic_deg
from eval_lib import predict                                                # reuses the exact best-of-8 DDIM eval path


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = list(csv.DictReader(open(C.CHUNKS_CSV)))
    for r in rows:
        r["episode"] = r["episode"].replace("EXDIR", C.DATA_DIR)           # resolve the portable placeholder
    st = np.load(os.path.join(HERE, "checkpoints", "target_norm_stats.npz"))
    sd, mu = st["std"], st["mean"]
    ckpt = os.path.join(HERE, "checkpoints", "base700_s0.pt")
    print(f"device {dev} | {len(rows)} example grasps | checkpoint base700_s0.pt\n")
    P = predict(ckpt, sd, mu, rows, dev, N=8, steps=50, ds_mod="my_dataset")   # best-of-8
    print(f"{'object':16s} {'grasp':>6s} {'GT tilt':>8s} {'pred tilt':>10s} {'axis err':>9s} {'geo err':>8s}")
    for i, r in enumerate(rows):
        ag, axg = rot_angle_axis(P["Rg"][i]); ap, axp = rot_angle_axis(P["Rp"][i])
        axis_err = np.degrees(np.arccos(np.clip(abs(float(axp @ axg)), 0, 1)))
        geo = float(geodesic_deg(P["Rp"][i][None], P["Rg"][i][None])[0])
        gi = r["episode"].split("_g")[-1].replace(".npz", "")
        print(f"{r['object']:16s} {gi:>6s} {ag:7.1f}° {ap:9.1f}° {axis_err:8.1f}° {geo:7.1f}°")
    print("\nGT tilt = simulated net rotation over the settle window; pred = WM best-of-8 prediction.")


if __name__ == "__main__":
    main()
