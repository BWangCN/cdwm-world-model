"""(#1 push) Train the MDN over continuous trajectory summaries (mixture NLL, mode-agnostic, no threshold labels).
Uniform sampling (density should reflect natural mode proportions). Saves TEST mixture params (un-z-scored means/scales).
    CDWM_MODALITY=full     python train_mdn.py --tag mdn_full
    CDWM_MODALITY=pose_only python train_mdn.py --tag mdn_pose
"""
import os, sys, json, argparse, time
import numpy as np, torch
from torch.utils.data import DataLoader
from grasp.my_dataset_traj import TrajDS
from grasp.trajnet import MDNTrajNet, mdn_nll


@torch.no_grad()
def infer(model, ds, dev, smean, sstd, bs=512):
    dl = DataLoader(ds, bs, shuffle=False, num_workers=6); W, M, S, Y, SR, OB = [], [], [], [], [], []
    model.eval()
    for b in dl:
        lg, mu, ls = model(b["pts"].to(dev), b["base_rel"].to(dev), b["closing"].to(dev), b["table"].to(dev))
        W.append(torch.softmax(lg, -1).cpu()); M.append(mu.cpu()); S.append(ls.cpu()); Y.append(b["y5"]); SR.append(b["summ_raw"]); OB += list(b["object"])
    w = torch.cat(W).numpy(); mu = torch.cat(M).numpy() * sstd + smean; sc = np.exp(torch.cat(S).numpy()) * sstd   # un-z-score
    return w, mu, sc, torch.cat(Y).numpy(), torch.cat(SR).numpy(), np.array(OB)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="mdn_full"); ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=256); ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--K", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; mod = os.environ.get("CDWM_MODALITY", "full")
    torch.manual_seed(a.seed); np.random.seed(a.seed); out = f"traj_runs/{a.tag}"; os.makedirs(out, exist_ok=True)
    tr = TrajDS("train", modality=mod, seed=a.seed); te = TrajDS("test", modality=mod); smean, sstd = tr.smean, tr.sstd
    dl = DataLoader(tr, a.bs, shuffle=True, num_workers=8, drop_last=True, persistent_workers=True)   # uniform
    print(f"[mdn {a.tag} mod={mod} K={a.K}] train {len(tr)} test {len(te)}", flush=True)
    model = MDNTrajNet(K=a.K).to(dev); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs); t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        for b in dl:
            lg, mu, ls = model(b["pts"].to(dev), b["base_rel"].to(dev), b["closing"].to(dev), b["table"].to(dev))
            loss = mdn_nll(lg, mu, ls, b["summ"].to(dev))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        sch.step()
        if ep % 20 == 0 or ep == a.epochs - 1: print(f"  ep{ep:3d} nll {loss.item():.3f} ({(time.time()-t0)/60:.1f}m)", flush=True)
    torch.save(model.state_dict(), f"{out}/best.pt")
    w, mu, sc, y, sr, ob = infer(model, te, dev, smean, sstd)
    np.savez(f"{out}/test_pred.npz", weight=w, means=mu, scales=sc, y5=y, summ_raw=sr, obj=ob, K=a.K, mod=mod)
    json.dump(dict(tag=a.tag, mod=mod, K=a.K, n_test=len(te)), open(f"{out}/summary.json", "w"))
    print(f"DONE {a.tag} -> {out}/test_pred.npz", flush=True)


if __name__ == "__main__":
    main()
