"""Train the grasp-outcome classifier (OutcomeNet on the local encoder). Class-balanced CE + weighted sampler (no focal,
Codex), post-hoc temperature scaling per head on object-disjoint val, then save TEST probs/labels/objects for analysis.

    CDWM_MODALITY=full     python train_cls.py --tag full
    CDWM_MODALITY=pose_only python train_cls.py --tag pose
"""
import os, sys, json, argparse, time
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from grasp.my_dataset_outcomes import OutcomeDS, FIVE
from grasp.classifier import OutcomeNet


def bal_acc(logits, y, K):
    p = logits.argmax(1); rec = []
    for c in range(K):
        m = y == c
        if m.sum(): rec.append((p[m] == c).float().mean().item())
    return float(np.mean(rec))


@torch.no_grad()
def infer(model, ds, dev, bs=512):
    dl = DataLoader(ds, bs, shuffle=False, num_workers=6)
    L5, L2, Y5, Y2, OBJ = [], [], [], [], []
    model.eval()
    for b in dl:
        l5, l2 = model(b["pts"].to(dev), b["base_rel"].to(dev), b["closing"].to(dev), b["table"].to(dev))
        L5.append(l5.cpu()); L2.append(l2.cpu()); Y5.append(b["y5"]); Y2.append(b["y2"]); OBJ += list(b["object"])
    return torch.cat(L5), torch.cat(L2), torch.cat(Y5), torch.cat(Y2), np.array(OBJ)


def fit_temp(logits, y):                                        # 1-D temperature scaling (minimize NLL on val)
    T = torch.ones(1, requires_grad=True); opt = torch.optim.LBFGS([T], lr=0.1, max_iter=50)
    def clo():
        opt.zero_grad(); loss = F.cross_entropy(logits / T.clamp_min(0.05), y); loss.backward(); return loss
    opt.step(clo); return float(T.detach().clamp_min(0.05))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="full"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=80); ap.add_argument("--bs", type=int, default=256); ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    mod = os.environ.get("CDWM_MODALITY", "full")
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = f"cls_runs/{a.tag}"; os.makedirs(out, exist_ok=True)
    tr = OutcomeDS("train", modality=mod, seed=a.seed); va = OutcomeDS("val", modality=mod); te = OutcomeDS("test", modality=mod)
    y5, y2 = tr.labels()
    print(f"[cls tag={a.tag} mod={mod}] train {len(tr)} val {len(va)} test {len(te)} | 5-way {np.bincount(y5)}", flush=True)
    # class-balanced: inverse-freq weights (loss) + weighted sampler (5-way, exposes rare DROP)
    freq5 = np.bincount(y5, minlength=5); w5 = torch.tensor((freq5.sum() / (5 * np.maximum(freq5, 1))).astype(np.float32))   # guard empty class (lifted-only)
    freq2 = np.bincount(y2, minlength=2); w2 = torch.tensor((freq2.sum() / (2 * np.maximum(freq2, 1))).astype(np.float32))
    samp_w = (1.0 / np.maximum(freq5, 1))[y5]; sampler = WeightedRandomSampler(torch.as_tensor(samp_w), len(y5), replacement=True)
    dltr = DataLoader(tr, a.bs, sampler=sampler, num_workers=8, drop_last=True, persistent_workers=True)
    model = OutcomeNet().to(dev); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    w5, w2 = w5.to(dev), w2.to(dev); t0 = time.time(); best = -1
    for ep in range(a.epochs):
        model.train()
        for b in dltr:
            l5, l2 = model(b["pts"].to(dev), b["base_rel"].to(dev), b["closing"].to(dev), b["table"].to(dev))
            loss = F.cross_entropy(l5, b["y5"].to(dev), weight=w5) + F.cross_entropy(l2, b["y2"].to(dev), weight=w2)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        if ep % 10 == 0 or ep == a.epochs - 1:
            l5, l2, yy5, yy2, _ = infer(model, va, dev)
            b2 = bal_acc(l2, yy2, 2); b5 = bal_acc(l5, yy5, 5)
            print(f"  ep{ep:3d} val bal-acc 2tier {b2:.3f} 5way {b5:.3f}  ({(time.time()-t0)/60:.1f}m)", flush=True)
            if b2 > best: best = b2; torch.save(model.state_dict(), f"{out}/best.pt")
    # reload best, calibrate on val, save TEST probs
    model.load_state_dict(torch.load(f"{out}/best.pt"))
    vl5, vl2, vy5, vy2, _ = infer(model, va, dev); T5 = fit_temp(vl5, vy5); T2 = fit_temp(vl2, vy2)
    tl5, tl2, ty5, ty2, tobj = infer(model, te, dev)
    p5 = torch.softmax(tl5 / T5, 1).numpy(); p2 = torch.softmax(tl2 / T2, 1).numpy()
    np.savez(f"{out}/test_pred.npz", p5=p5, p2=p2, y5=ty5.numpy(), y2=ty2.numpy(), obj=tobj, T5=T5, T2=T2, mod=mod)
    json.dump(dict(tag=a.tag, mod=mod, val_bal2=best, T5=T5, T2=T2, n_test=len(te)), open(f"{out}/summary.json", "w"), indent=1)
    print(f"DONE {a.tag}: val bal2 {best:.3f}  T5 {T5:.2f} T2 {T2:.2f} -> {out}/test_pred.npz", flush=True)


if __name__ == "__main__":
    main()
