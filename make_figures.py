"""Paper figures. Okabe-Ito colorblind-safe categorical palette; single-hue sequential for the alignment heatmap.
Outputs figures/*.png. Needs GPU (MDN train+test inference for the proper MDN-derived risk).
    python make_figures.py
"""
import os, sys, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
sys.path.insert(0, ".")
from my_dataset_traj import TrajDS
from wm.trajnet import MDNTrajNet
from eval_traj import MODES
OI = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})
os.makedirs("figures", exist_ok=True)


@torch.no_grad()
def mdn_weights(model, split, dev):
    ds = TrajDS(split, modality="full"); dl = DataLoader(ds, 512, shuffle=False, num_workers=6); W, Y = [], []
    model.eval()
    for b in dl:
        lg, _, _ = model(b["pts"].to(dev), b["base_rel"].to(dev), b["closing"].to(dev), b["table"].to(dev))
        W.append(torch.softmax(lg, -1).cpu()); Y.append(b["y5"])
    return torch.cat(W).numpy(), torch.cat(Y).numpy()


def rc_curve(sev, fail):
    o = np.argsort(-sev); xs = np.linspace(0, 0.5, 26)
    return xs, np.array([fail[o[int(r * len(o)):]].mean() for r in xs])


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    K = int(np.load("traj_runs/mdn_full/test_pred.npz")["K"])
    m = MDNTrajNet(K=K).to(dev); m.load_state_dict(torch.load("traj_runs/mdn_full/best.pt", map_location=dev))
    wtr, ytr = mdn_weights(m, "train", dev); wte, yte = mdn_weights(m, "test", dev)
    fail_te = (yte >= 2).astype(float); pfail_comp = (wtr * (ytr >= 2)[:, None]).sum(0) / (wtr.sum(0) + 1e-9)
    mdn_risk = wte @ pfail_comp
    cls = dict(np.load("cls_runs/full/test_pred.npz", allow_pickle=True)); assert np.array_equal(cls["y5"], yte)

    # --- Fig 1: emergent MDN components vs physical modes (row-normalized), alive components only ---
    comp = wte.argmax(1); alive = [k for k in range(K) if (comp == k).sum() > 30]
    A = np.array([[np.mean(yte[comp == k] == mi) for mi in range(5)] for k in alive])
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    im = ax.imshow(A, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(5)); ax.set_xticklabels(MODES, rotation=30, ha="right")
    ax.set_yticks(range(len(alive))); ax.set_yticklabels([f"comp {k}\nP(fail)={pfail_comp[k]:.2f}" for k in alive])
    for i in range(len(alive)):
        for j in range(5):
            if A[i, j] > 0.08: ax.text(j, i, f"{A[i,j]*100:.0f}", ha="center", va="center", color="white" if A[i, j] > 0.5 else "black", fontsize=9)
    ax.set_title("Emergent MDN components align with physical failure modes\n(unsupervised; % of each component's grasps per mode)")
    fig.colorbar(im, label="fraction"); fig.tight_layout(); fig.savefig("figures/fig1_emergent_modes.png"); plt.close(fig)

    # --- Fig 2: risk-coverage (reject riskiest x% -> KEPT failure rate) ---
    fig, ax = plt.subplots(figsize=(6.2, 4.0)); rng = np.random.default_rng(0)
    series = [("classifier P(fail)", cls["p2"][:, 1], OI[0]), ("MDN-derived P(fail)", mdn_risk, OI[2]),
              ("pose severity", np.load("traj_runs/direct_pose/test_pred.npz")["pred_mean"][:, 9], OI[1]),
              ("random", rng.random(len(yte)), OI[6])]
    for name, sev, c in series:
        x, y = rc_curve(sev, fail_te); ax.plot(x * 100, y, lw=2, color=c, label=name)
    ax.axhline(fail_te.mean(), ls=":", color="gray", lw=1)
    ax.set_xlabel("% riskiest grasps rejected"); ax.set_ylabel("failure rate of KEPT grasps"); ax.set_ylim(0.4, 0.62)
    ax.set_title("Grasp-risk ranking: MDN matches the dedicated classifier"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig("figures/fig2_risk_coverage.png"); plt.close(fig)

    # --- Fig 3: distributional coverage (settle-rot median error) ---
    labels = ["MDN\ntop-K oracle", "MDN\nmixture-mean", "single-mean\n(mode-agnostic)", "labeled\nmode-mixture"]
    vals = [8.35, 13.18, 12.97, 15.26]; cols = [OI[2], OI[2], OI[1], OI[0]]
    fig, ax = plt.subplots(figsize=(6.2, 3.8)); b = ax.bar(labels, vals, color=cols, width=0.6)
    ax.bar_label(b, fmt="%.1f°", padding=3)
    ax.axhline(124.5, ls=":", color="gray"); ax.text(3.3, 118, "per-mode median = 124.5°", ha="right", color="gray", fontsize=9)
    ax.set_ylabel("settle-rotation error (°, median)"); ax.set_ylim(0, 20)
    ax.set_title("The distribution covers the true outcome:\nMDN's best component beats the label-supervised model")
    fig.tight_layout(); fig.savefig("figures/fig3_coverage.png"); plt.close(fig)

    # --- Fig 4: rigid regime ladder (Part 1) ---
    cfg = ["base", "objgrip", "+frame", "+cov", "local", "frame+geo", "local_geo"]; geo = [12.26, 7.40, 4.49, 4.13, 4.02, 3.77, 3.46]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    b = ax.bar(cfg, geo, color=[OI[6] if g > 9.57 else OI[0] for g in geo], width=0.62); ax.bar_label(b, fmt="%.2f", padding=3, fontsize=9)
    ax.axhline(9.57, ls=":", color=OI[1]); ax.text(6.4, 9.9, "no-motion baseline 9.57", ha="right", color=OI[1], fontsize=9)
    ax.set_ylabel("held-out single-shot geodesic error (°)"); ax.set_title("Rigid regime: gripper-frame conditioning is the 3× lever")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right"); fig.tight_layout(); fig.savefig("figures/fig4_rigid_ladder.png"); plt.close(fig)
    print("wrote figures/fig1_emergent_modes.png fig2_risk_coverage.png fig3_coverage.png fig4_rigid_ladder.png")


if __name__ == "__main__":
    main()
