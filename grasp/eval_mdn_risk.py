"""Derive a proper P(fail) from the MDN mixture (not the crude E[max-drift]): use the EMERGENT components as a soft
clustering, label each component by its TRAIN failure rate (train labels only), then risk = sum_k w_k(test) P(fail|comp_k).
Test whether the distributional WM matches/beats the classifier P(fail) as a grasp ranker -> unify WM + deployment.
    python eval_mdn_risk.py    (loads traj_runs/mdn_full/best.pt; needs GPU for inference)
"""
import sys, numpy as np, torch
from torch.utils.data import DataLoader
from grasp.my_dataset_traj import TrajDS
from grasp.trajnet import MDNTrajNet


@torch.no_grad()
def weights(model, ds, dev):
    dl = DataLoader(ds, 512, shuffle=False, num_workers=6); W, Y, OB = [], [], []
    model.eval()
    for b in dl:
        lg, _, _ = model(b["pts"].to(dev), b["base_rel"].to(dev), b["closing"].to(dev), b["table"].to(dev))
        W.append(torch.softmax(lg, -1).cpu()); Y.append(b["y5"]); OB += list(b["object"])
    return torch.cat(W).numpy(), torch.cat(Y).numpy(), np.array(OB)


def rc_auc(sev, fail):
    o = np.argsort(-sev); return round(float(np.mean([fail[o[int(r * len(o)):]].mean() for r in np.linspace(0, 0.5, 11)])), 3)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"; K = int(np.load("traj_runs/mdn_full/test_pred.npz")["K"])
    model = MDNTrajNet(K=K).to(dev); model.load_state_dict(torch.load("traj_runs/mdn_full/best.pt", map_location=dev))
    wtr, ytr, _ = weights(model, TrajDS("train", modality="full"), dev)
    wte, yte, _ = weights(model, TrajDS("test", modality="full"), dev)
    ftr = (ytr >= 2).astype(float); fte = (yte >= 2).astype(float)
    pfail_comp = (wtr * ftr[:, None]).sum(0) / (wtr.sum(0) + 1e-9)        # TRAIN weighted failure rate per component
    risk = wte @ pfail_comp                                              # test P(fail) from the mixture
    cls = dict(np.load("cls_runs/full/test_pred.npz", allow_pickle=True))
    assert np.array_equal(cls["y5"], yte), "align mismatch"
    print(f"per-component TRAIN P(fail): {np.round(pfail_comp, 2)}  (base {ftr.mean():.3f})")
    print(f"\n=== risk-coverage AUC (lower=better; base fail {fte.mean():.3f}) ===")
    print(f"  MDN-derived P(fail)   {rc_auc(risk, fte)}")
    print(f"  classifier P(fail)    {rc_auc(cls['p2'][:, 1], fte)}")
    print(f"  random                {rc_auc(np.random.default_rng(0).random(len(fte)), fte)}")
    # reject-30% detail
    for name, sev in [("MDN-P(fail)", risk), ("classifier", cls["p2"][:, 1])]:
        o = np.argsort(-sev); n = int(0.3 * len(o))
        print(f"  {name:12s} reject30%: KEPT-fail {fte[o[n:]].mean():.3f}  REJECT-precision {fte[o[:n]].mean():.3f}")


if __name__ == "__main__":
    main()
