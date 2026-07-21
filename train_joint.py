"""(colleague's suggestion, validation control) JOINT: shared local encoder -> MDN summary head + binary success/fail
head, one model, Kendall-weighted joint loss. Saves TEST MDN params (un-z-scored) AND binary P(fail) in the same npz
schema as train_mdn so eval_joint can compare joint-vs-separate directly.
    CDWM_MODALITY=full      python train_joint.py --tag joint_full
    CDWM_MODALITY=pose_only python train_joint.py --tag joint_pose
"""
import os, sys, json, argparse, time
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from my_dataset_traj import TrajDS
from wm.trajnet import JointMDNOutcomeNet, joint_loss


@torch.no_grad()
def infer(model, ds, dev, smean, sstd, bs=512):
    dl = DataLoader(ds, bs, shuffle=False, num_workers=6); W, M, S, P2, Y, SR, OB = [], [], [], [], [], [], []
    model.eval()
    for b in dl:
        lg, mu, ls, c2 = model(b["pts"].to(dev), b["base_rel"].to(dev), b["closing"].to(dev), b["table"].to(dev))
        W.append(torch.softmax(lg, -1).cpu()); M.append(mu.cpu()); S.append(ls.cpu())
        P2.append(torch.softmax(c2, -1).cpu()); Y.append(b["y5"]); SR.append(b["summ_raw"]); OB += list(b["object"])
    w = torch.cat(W).numpy(); mu = torch.cat(M).numpy() * sstd + smean; sc = np.exp(torch.cat(S).numpy()) * sstd
    return w, mu, sc, torch.cat(P2).numpy(), torch.cat(Y).numpy(), torch.cat(SR).numpy(), np.array(OB)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="joint_full"); ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=256); ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--K", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; mod = os.environ.get("CDWM_MODALITY", "full")
    torch.manual_seed(a.seed); np.random.seed(a.seed); out = f"traj_runs/{a.tag}"; os.makedirs(out, exist_ok=True)
    tr = TrajDS("train", modality=mod, seed=a.seed); te = TrajDS("test", modality=mod); smean, sstd = tr.smean, tr.sstd
    dl = DataLoader(tr, a.bs, shuffle=True, num_workers=8, drop_last=True, persistent_workers=True)   # uniform
    # binary success/fail class weights from TRAIN frequency (fail = y5>=2); no hand-tuning
    yfa = (tr.labels()[0] >= 2).astype(np.int64)
    cw = torch.tensor([1.0 / max((yfa == 0).mean(), 1e-6), 1.0 / max((yfa == 1).mean(), 1e-6)], dtype=torch.float32)
    cw = (cw / cw.sum() * 2).to(dev)
    print(f"[joint {a.tag} mod={mod} K={a.K}] train {len(tr)} test {len(te)} fail-rate {yfa.mean():.3f} cw {cw.tolist()}", flush=True)
    model = JointMDNOutcomeNet(K=a.K).to(dev); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs); t0 = time.time()
    for ep in range(a.epochs):
        model.train(); ln = lc = 0.0
        for b in dl:
            lg, mu, ls, c2 = model(b["pts"].to(dev), b["base_rel"].to(dev), b["closing"].to(dev), b["table"].to(dev))
            yfail = (b["y5"] >= 2).long().to(dev)
            loss, nll, ce = joint_loss(lg, mu, ls, c2, b["summ"].to(dev), yfail, model.log_s, cw)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            ln, lc = nll.item(), ce.item()
        sch.step()
        if ep % 20 == 0 or ep == a.epochs - 1:
            print(f"  ep{ep:3d} nll {ln:.3f} ce {lc:.3f} logs {model.log_s.detach().cpu().numpy().round(2)} ({(time.time()-t0)/60:.1f}m)", flush=True)
    torch.save(model.state_dict(), f"{out}/best.pt")
    w, mu, sc, p2, y, sr, ob = infer(model, te, dev, smean, sstd)
    np.savez(f"{out}/test_pred.npz", weight=w, means=mu, scales=sc, p2=p2, y5=y, summ_raw=sr, obj=ob, K=a.K, mod=mod)
    json.dump(dict(tag=a.tag, mod=mod, K=a.K, n_test=len(te)), open(f"{out}/summary.json", "w"))
    print(f"DONE {a.tag} -> {out}/test_pred.npz", flush=True)


if __name__ == "__main__":
    main()
