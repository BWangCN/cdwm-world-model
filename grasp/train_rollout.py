"""(colleague's suggestion #2) Train the full-rollout WM: DiT denoises the full H x 9 object-in-gripper trajectory over
ALL v2 outcomes + a read-off boolean head. Kendall-weighted DDPM-eps + CE. Saves best.pt, TEST boolean P(fail),
z-score stats, and TEST raw targets for eval_rollout (per-frame coverage, generative calibration, boolean calibration).
    CDWM_MODALITY=full      python train_rollout.py --tag roll_full
    CDWM_MODALITY=pose_only python train_rollout.py --tag roll_pose
"""
import os, sys, json, argparse, time
import numpy as np, torch
from torch.utils.data import DataLoader
from grasp.my_dataset_trajfull import TrajFullDS
from grasp.dit_rollout import DiTRollout, rollout_loss
from common.dit import cosine_acp


@torch.no_grad()
def infer_cls(model, ds, dev, bs=512):
    dl = DataLoader(ds, bs, shuffle=False, num_workers=6); P2, Y, TR, OB = [], [], [], []
    model.eval()
    for b in dl:
        cond = model.encode(b["pts"].to(dev), b["base_rel"].to(dev), b["closing"].to(dev), b["table"].to(dev))
        P2.append(torch.softmax(model.classify(cond), -1).cpu()); Y.append(b["y5"]); TR.append(b["target_raw"]); OB += list(b["object"])
    return torch.cat(P2).numpy(), torch.cat(Y).numpy(), torch.cat(TR).numpy(), np.array(OB)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="roll_full"); ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--bs", type=int, default=256); ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--T", type=int, default=1000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; mod = os.environ.get("CDWM_MODALITY", "full")
    torch.manual_seed(a.seed); np.random.seed(a.seed); out = f"roll_runs/{a.tag}"; os.makedirs(out, exist_ok=True)
    tr = TrajFullDS("train", modality=mod, seed=a.seed); te = TrajFullDS("test", modality=mod)
    dl = DataLoader(tr, a.bs, shuffle=True, num_workers=8, drop_last=True, persistent_workers=True)   # uniform (natural dist)
    yfa = (tr.labels()[0] >= 2).astype(np.int64)
    cw = torch.tensor([1.0 / max((yfa == 0).mean(), 1e-6), 1.0 / max((yfa == 1).mean(), 1e-6)], dtype=torch.float32)
    cw = (cw / cw.sum() * 2).to(dev)
    acp = cosine_acp(a.T, device=dev)
    print(f"[rollout {a.tag} mod={mod}] train {len(tr)} test {len(te)} fail {yfa.mean():.3f} K={tr.K} T={a.T}", flush=True)
    model = DiTRollout(H=tr.K).to(dev); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs); t0 = time.time()
    for ep in range(a.epochs):
        model.train(); ld = lc = 0.0
        for b in dl:
            cond = model.encode(b["pts"].to(dev), b["base_rel"].to(dev), b["closing"].to(dev), b["table"].to(dev))
            yfail = (b["y5"] >= 2).long().to(dev)
            loss, dmse, ce = rollout_loss(model, cond, b["target"].to(dev), yfail, acp, cw)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            ld, lc = dmse.item(), ce.item()
        sch.step()
        if ep % 20 == 0 or ep == a.epochs - 1:
            print(f"  ep{ep:3d} dmse {ld:.4f} ce {lc:.3f} logs {model.log_s.detach().cpu().numpy().round(2)} ({(time.time()-t0)/60:.1f}m)", flush=True)
    torch.save(model.state_dict(), f"{out}/best.pt")
    p2, y, tr_raw, ob = infer_cls(model, te, dev)
    np.savez(f"{out}/test_pred.npz", p2=p2, y5=y, target_raw=tr_raw, obj=ob,
             tmean=te.tmean, tstd=te.tstd, T=a.T, K=te.K, mod=mod)
    json.dump(dict(tag=a.tag, mod=mod, n_test=len(te)), open(f"{out}/summary.json", "w"))
    print(f"DONE {a.tag} -> {out}/test_pred.npz", flush=True)


if __name__ == "__main__":
    main()
