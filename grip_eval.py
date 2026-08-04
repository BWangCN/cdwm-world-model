"""Proper grip-head metrics on the trained best.pt over the FULL val set: MAE (rad), phase-sliced MAE
(close vs settled-hold), and tolerance accuracy (within 0.05 / 0.1 rad) — the metrics Codex recommended
to lead with instead of exact-1%-bin acc100. Read-only eval; writes grip_eval.json next to the ckpt."""
import os, json, numpy as np, torch
from torch.utils.data import DataLoader
from my_dataset_grasp_mtl import GraspMTLDS, GRIP_MAX
from wm.grasp_mtl import GraspMTL

dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load("gated_runs/grasp_mtl/best.pt", map_location=dev)
va = GraspMTLDS("val", stats=ck["stats"])
model = GraspMTL(n_feat=13, H=32, D=256, depth=4).to(dev); model.load_state_dict(ck["model"]); model.eval()
dl = DataLoader(va, 128, num_workers=4)

GP, GT, PH = [], [], []
with torch.no_grad():
    for b in dl:
        cond = model.encode(*[b[k].to(dev) for k in ("pts", "base_rel", "closing", "table")])
        GP.append(model.predict_grip(cond).clamp(0, 1).cpu().numpy()); GT.append(b["grip_tgt"].numpy()); PH.append(b["grip_phase"].numpy())
gp, gt, ph = np.concatenate(GP), np.concatenate(GT), np.concatenate(PH)      # (N,H) normalized [0,1]
err = np.abs(gp - gt) * GRIP_MAX                                             # abs error in rad
close, hold = ph == 1, ph == 2
r = dict(n_ep=int(gp.shape[0]),
         mae_rad=float(err.mean()), mae_close_rad=float(err[close].mean()), mae_hold_rad=float(err[hold].mean()),
         frac_within_0p05rad=float((err < 0.05).mean()), frac_within_0p10rad=float((err < 0.10).mean()),
         frac_within_0p05_hold=float((err[hold] < 0.05).mean()), frac_within_0p05_close=float((err[close] < 0.05).mean()),
         mae_pct_of_range=float(err.mean() / GRIP_MAX * 100))
json.dump(r, open("gated_runs/grasp_mtl/grip_eval.json", "w"), indent=1)
print(json.dumps(r, indent=1))
# sanity: hold-phase should be predicted tighter (plateau) than the ramp; overall MAE must match training (~0.054)
assert r["mae_rad"] < 0.15, "grip MAE unexpectedly high"
