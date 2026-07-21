"""Phase-2 trainer: mode CE + per-mode summary regression on the GT-mode expert (teacher forcing, Codex). Saves TEST
predictions (mode probs + per-mode RAW summaries + GT summary + mode + object) for eval_traj.
    CDWM_MODALITY=full     python train_traj.py --tag traj_full
    CDWM_MODALITY=pose_only python train_traj.py --tag traj_pose
"""
import os, sys, json, argparse, time
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from my_dataset_traj import TrajDS
from wm.trajnet import TrajNet


@torch.no_grad()
def infer(model, ds, dev, smean, sstd, bs=512):
    dl = DataLoader(ds, bs, shuffle=False, num_workers=6); MP, PS, Y, SR, OB = [], [], [], [], []
    model.eval()
    for b in dl:
        ml, ps = model(b["pts"].to(dev), b["base_rel"].to(dev), b["closing"].to(dev), b["table"].to(dev))
        MP.append(torch.softmax(ml, 1).cpu()); PS.append(ps.cpu()); Y.append(b["y5"]); SR.append(b["summ_raw"]); OB += list(b["object"])
    ps = torch.cat(PS).numpy() * sstd + smean                 # un-z-score per-mode summaries
    return torch.cat(MP).numpy(), ps, torch.cat(Y).numpy(), torch.cat(SR).numpy(), np.array(OB)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="traj_full"); ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--bs", type=int, default=256); ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; mod = os.environ.get("CDWM_MODALITY", "full")
    torch.manual_seed(a.seed); np.random.seed(a.seed); out = f"traj_runs/{a.tag}"; os.makedirs(out, exist_ok=True)
    tr = TrajDS("train", modality=mod, seed=a.seed); te = TrajDS("test", modality=mod)
    smean, sstd = tr.smean, tr.sstd
    y5, _ = tr.labels(); freq5 = np.bincount(y5, minlength=5)
    w5 = torch.tensor((freq5.sum() / (5 * np.maximum(freq5, 1))).astype(np.float32)).to(dev)
    samp = WeightedRandomSampler(torch.as_tensor((1.0 / np.maximum(freq5, 1))[y5]), len(y5), replacement=True)
    dl = DataLoader(tr, a.bs, sampler=samp, num_workers=8, drop_last=True, persistent_workers=True)
    print(f"[traj {a.tag} mod={mod}] train {len(tr)} test {len(te)} | 5-way {freq5}", flush=True)
    model = TrajNet().to(dev); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs); t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        for b in dl:
            ml, ps = model(b["pts"].to(dev), b["base_rel"].to(dev), b["closing"].to(dev), b["table"].to(dev))
            y = b["y5"].to(dev); gt_exp = ps[torch.arange(len(y)), y]                 # GT-mode expert (teacher forcing)
            loss = F.cross_entropy(ml, y, weight=w5) + F.mse_loss(gt_exp, b["summ"].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        if ep % 20 == 0 or ep == a.epochs - 1: print(f"  ep{ep:3d} loss {loss.item():.3f} ({(time.time()-t0)/60:.1f}m)", flush=True)
    torch.save(model.state_dict(), f"{out}/best.pt")
    mp, ps, y, sr, ob = infer(model, te, dev, smean, sstd)
    np.savez(f"{out}/test_pred.npz", mode_prob=mp, pred_summ=ps, y5=y, summ_raw=sr, obj=ob, mod=mod)
    json.dump(dict(tag=a.tag, mod=mod, n_test=len(te)), open(f"{out}/summary.json", "w"))
    print(f"DONE {a.tag} -> {out}/test_pred.npz", flush=True)


if __name__ == "__main__":
    main()
