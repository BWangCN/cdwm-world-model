"""Baseline drop WM training. Predicts basin_transition (tip vs stay) + resting orientation (rest6d) from
(release-frame cloud + release params). Reuses the slip encoder + train_cls structure (balanced sampler,
weighted CE, cosine LR, temperature scaling, checkpoint on val).

Controls via env CDWM_MODALITY (full | pose_only) and --frame (release | obj). Reports vs a NO-MOTION baseline
(predict identity rotation) — the honest bar, since ~98% of episodes barely rotate.

    CDWM_MODALITY=full python train_drop.py --tag full --frame release
"""
import os, sys, json, time, argparse
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from my_dataset_drop import DropDS
from wm.drop_net import DropNet, sixd_to_R, rot_loss


def geodesic(Ra, Rb):
    Rr = torch.matmul(Ra, Rb.transpose(-1, -2))
    tr = torch.clamp((Rr[..., 0, 0] + Rr[..., 1, 1] + Rr[..., 2, 2] - 1) / 2, -1, 1)
    return torch.rad2deg(torch.arccos(tr))


def auroc(scores, y):                                      # rank-based AUROC (binary)
    y = np.asarray(y); s = np.asarray(scores)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return float("nan")
    r = s.argsort().argsort() + 1                          # ranks
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


@torch.no_grad()
def evaluate(model, ds, dev, bs=512, max_batches=None):
    dl = DataLoader(ds, bs, shuffle=False, num_workers=6, pin_memory=True)
    model.eval(); BT, P, GEO, GEO0, OBJ = [], [], [], [], []
    for bi, b in enumerate(dl):
        if max_batches and bi >= max_batches: break
        lbt, rot = model(b["pts"].to(dev, non_blocking=True), b["base_rel"].to(dev, non_blocking=True), b["closing"].to(dev, non_blocking=True), b["table"].to(dev, non_blocking=True))
        Rp = sixd_to_R(rot); Rg = sixd_to_R(b["rest6d"].to(dev, non_blocking=True))
        I = torch.eye(3, device=dev).expand_as(Rg)
        GEO.append(geodesic(Rp, Rg).cpu()); GEO0.append(geodesic(I, Rg).cpu())   # model vs no-motion
        P.append(torch.softmax(lbt, 1)[:, 1].cpu()); BT.append(b["y_bt"]); OBJ += list(b["object"])
    bt = torch.cat(BT).numpy(); p = torch.cat(P).numpy()
    geo = torch.cat(GEO).numpy(); geo0 = torch.cat(GEO0).numpy()
    pred = (p > 0.5).astype(int)
    tp = ((pred == 1) & (bt == 1)).sum(); tn = ((pred == 0) & (bt == 0)).sum()
    bal = 0.5 * (tp / max((bt == 1).sum(), 1) + tn / max((bt == 0).sum(), 1))
    return dict(bal_bt=float(bal), auroc_bt=auroc(p, bt), geo_med=float(np.median(geo)),
                nomotion_med=float(np.median(geo0)), geo_mean=float(geo.mean()), nomotion_mean=float(geo0.mean()),
                n=len(bt), n_pos=int(bt.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full"); ap.add_argument("--frame", default="release")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=256); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=1.0); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mod = os.environ.get("CDWM_MODALITY", "full")
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = f"drop_runs/{a.tag}"; os.makedirs(out, exist_ok=True)

    tr = DropDS("train", modality=mod, frame=a.frame, seed=a.seed)
    va = DropDS("val", modality=mod, frame=a.frame)
    y = tr.labels()
    print(f"[drop tag={a.tag} mod={mod} frame={a.frame}] train {len(tr)} val {len(va)} | bt pos {y.sum()} ({y.mean():.1%})")
    freq = np.bincount(y, minlength=2); wbt = torch.tensor((freq.sum() / (2 * np.maximum(freq, 1))).astype(np.float32)).to(dev)
    sampw = (1.0 / np.maximum(freq, 1))[y]
    sampler = WeightedRandomSampler(torch.as_tensor(sampw), len(y), replacement=True)
    dltr = DataLoader(tr, a.bs, sampler=sampler, num_workers=8, drop_last=True, persistent_workers=True, pin_memory=True)

    model = DropNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    t0 = time.time(); best = -1
    for ep in range(a.epochs):
        model.train()
        for bi, b in enumerate(dltr):
            lbt, rot = model(b["pts"].to(dev, non_blocking=True), b["base_rel"].to(dev, non_blocking=True), b["closing"].to(dev, non_blocking=True), b["table"].to(dev, non_blocking=True))
            Rg = sixd_to_R(b["rest6d"].to(dev, non_blocking=True))
            loss = F.cross_entropy(lbt, b["y_bt"].to(dev, non_blocking=True), weight=wbt) + a.lam * rot_loss(rot, Rg)
            opt.zero_grad(); loss.backward(); opt.step()
            if a.smoke and bi >= 3: break
        sch.step()
        if ep % 5 == 0 or ep == a.epochs - 1 or a.smoke:
            m = evaluate(model, va, dev, max_batches=6 if a.smoke else None)
            score = m["bal_bt"] + max(0, m["nomotion_med"] - m["geo_med"]) / 20   # bt acc + rotation gain over no-motion
            print(f"  ep{ep:3d} val bal_bt {m['bal_bt']:.3f} auroc {m['auroc_bt']:.3f} "
                  f"geo_med {m['geo_med']:.2f}° (no-motion {m['nomotion_med']:.2f}°) ({(time.time()-t0)/60:.1f}m)")
            if score > best: best = score; torch.save(model.state_dict(), f"{out}/best.pt")
        if a.smoke: break
    model.load_state_dict(torch.load(f"{out}/best.pt"))
    te = DropDS("test", modality=mod, frame=a.frame)
    m = evaluate(model, te, dev, max_batches=6 if a.smoke else None)
    print(f"[TEST tag={a.tag}] {json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()})}")
    json.dump(dict(tag=a.tag, mod=mod, frame=a.frame, **m), open(f"{out}/summary.json", "w"), indent=1)

if __name__ == "__main__":
    main()
