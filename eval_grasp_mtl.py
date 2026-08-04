"""Proper eval of the clean MTL grasp WM (gated_runs/grasp_mtl/best.pt) on the TEST split: trajectory endpoint geodesic
(N=1 + best-of-8, 50 DDIM steps) on rigid test episodes, and the boolean slip head AUROC/acc/slip-recall on all test.
    python eval_grasp_mtl.py [--S 8]
"""
import os, sys, json, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from my_dataset_grasp_mtl import GraspMTLDS
from wm.grasp_mtl import GraspMTL
from wm.dit import cosine_acp, ddim_sample
from wm.common import sixd_to_R, geodesic_deg
from torch.utils.data import DataLoader


def auroc(y, s):
    o = np.argsort(-s); y = y[o]; P = y.sum(); N = len(y) - P
    return float(np.trapz(np.cumsum(y) / P, np.cumsum(1 - y) / N)) if P and N else float("nan")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--S", type=int, default=8); a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0)
    ck = torch.load("gated_runs/grasp_mtl/best.pt", map_location=dev)
    m = GraspMTL(n_feat=13, H=32, D=256, depth=4).to(dev); m.load_state_dict(ck["model"]); m.eval()
    stats = ck["stats"]; sd, mu = stats["std"].astype(np.float32), stats["mean"].astype(np.float32)
    acp = cosine_acp(1000, device=dev)
    te = GraspMTLDS("test", stats=stats); dl = DataLoader(te, 128, num_workers=6)
    G1, G8, PS, YS = [], [], [], []
    for b in dl:
        cond = m.encode(*[b[k].to(dev) for k in ("pts", "base_rel", "closing", "table")])
        PS.append(torch.softmax(m.classify(cond), 1)[:, 1].cpu().numpy()); YS.append(b["y_slip"].numpy())
        rig = b["is_rigid"].bool()
        if not rig.any(): continue
        cr = cond[rig]; gt = b["target"].numpy()[rig.numpy(), -1, :] * sd + mu; Rg = sixd_to_R(gt[:, 3:9])
        best = np.full(len(Rg), 1e9); first = None
        for s in range(a.S):
            x0 = ddim_sample(m, cr, 32, acp, steps=50, device=dev)[:, -1, :].cpu().numpy() * sd + mu
            gg = geodesic_deg(sixd_to_R(x0[:, 3:9]), Rg)
            if s == 0: first = gg
            best = np.minimum(best, gg)
        G1.append(first); G8.append(best)
    g1 = np.concatenate(G1); g8 = np.concatenate(G8); P = np.concatenate(PS); Y = np.concatenate(YS); pred = (P > 0.5).astype(int)
    out = dict(traj_geo_n1_med=round(float(np.median(g1)), 3), traj_geo_best8_med=round(float(np.median(g8)), 3), n_rigid=len(g1),
               bool_auroc=round(auroc(Y, P), 4), bool_acc=round(float((pred == Y).mean()), 4),
               bool_slip_recall=round(float(pred[Y == 1].mean()), 4), bool_rigid_kept=round(float((pred[Y == 0] == 0).mean()), 4), n_test=len(Y))
    print(json.dumps(out, indent=1)); json.dump(out, open("gated_runs/grasp_mtl/eval.json", "w"), indent=1)


if __name__ == "__main__":
    main()
