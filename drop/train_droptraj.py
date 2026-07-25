"""Trajectory-supervision arm (Codex-converged design): identical encoder/split/protocol to train_drop, but the
rotation head predicts K=8 anchor orientations spanning contact-onset -> settle (equal chordal loss over anchors,
no endpoint weighting -- ablate only if the endpoint degrades). Endpoint readout = anchor K-1 (= the resting pose),
directly comparable to the H=1 baseline. Decisive question: does transient supervision improve the endpoint median
and, separately, the >=15deg tail stratum (where the point baseline ties no-motion)?

    python -m drop.train_droptraj --tag traj_k8 --frame release
"""
import os, sys, json, time, argparse
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from drop.my_dataset_drop import DropTrajDS
from drop.drop_net import DropTrajNet, sixd_to_R, rot_loss
from drop.train_drop import geodesic, auroc


@torch.no_grad()
def evaluate(model, ds, dev, bs=512, max_batches=None, dump=None):
    dl = DataLoader(ds, bs, shuffle=False, num_workers=6, pin_memory=True)
    model.eval(); BT, P, GEO, GEO0, OBJ = [], [], [], [], []
    for bi, b in enumerate(dl):
        if max_batches and bi >= max_batches: break
        lbt, rot = model(b["pts"].to(dev, non_blocking=True), b["base_rel"].to(dev, non_blocking=True),
                         b["closing"].to(dev, non_blocking=True), b["table"].to(dev, non_blocking=True))
        Rp = sixd_to_R(rot[:, -1])                                     # ENDPOINT = anchor K-1 (the resting pose)
        Rg = sixd_to_R(b["rest6d"].to(dev, non_blocking=True))
        I = torch.eye(3, device=dev).expand_as(Rg)
        GEO.append(geodesic(Rp, Rg).cpu()); GEO0.append(geodesic(I, Rg).cpu())
        P.append(torch.softmax(lbt, 1)[:, 1].cpu()); BT.append(b["y_bt"]); OBJ += list(b["object"])
    bt = torch.cat(BT).numpy(); p = torch.cat(P).numpy()
    geo = torch.cat(GEO).numpy(); geo0 = torch.cat(GEO0).numpy()
    if dump:
        np.savez(dump, geo=geo, geo0=geo0, pr=p, bt=bt, obj=np.array(OBJ))
    tail = geo0 >= 15                                                  # the documented failure stratum
    return dict(auroc_bt=auroc(p, bt), geo_med=float(np.median(geo)), nomotion_med=float(np.median(geo0)),
                tail_n=int(tail.sum()), tail_geo_med=float(np.median(geo[tail])) if tail.any() else float("nan"),
                tail_nomotion_med=float(np.median(geo0[tail])) if tail.any() else float("nan"),
                n=len(bt), n_pos=int(bt.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="traj_k8"); ap.add_argument("--frame", default="release")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=256); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=1.0); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = f"drop_runs/{a.tag}"; os.makedirs(out, exist_ok=True)

    tr = DropTrajDS("train", modality="full", frame=a.frame, seed=a.seed)
    va = DropTrajDS("val", modality="full", frame=a.frame)
    y = tr.labels(); K = tr.K
    print(f"[droptraj tag={a.tag} K={K} frame={a.frame}] train {len(tr)} val {len(va)} | bt pos {y.sum()} ({y.mean():.1%})")
    freq = np.bincount(y, minlength=2); wbt = torch.tensor((freq.sum() / (2 * np.maximum(freq, 1))).astype(np.float32)).to(dev)
    sampler = WeightedRandomSampler(torch.as_tensor((1.0 / np.maximum(freq, 1))[y]), len(y), replacement=True)
    dltr = DataLoader(tr, a.bs, sampler=sampler, num_workers=8, drop_last=True, persistent_workers=True, pin_memory=True)

    model = DropTrajNet(K=K).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    t0 = time.time(); best = -1
    for ep in range(a.epochs):
        model.train()
        for bi, b in enumerate(dltr):
            lbt, rot = model(b["pts"].to(dev, non_blocking=True), b["base_rel"].to(dev, non_blocking=True),
                             b["closing"].to(dev, non_blocking=True), b["table"].to(dev, non_blocking=True))
            Rg = sixd_to_R(b["traj6d"].to(dev, non_blocking=True).reshape(-1, 6))
            loss = F.cross_entropy(lbt, b["y_bt"].to(dev, non_blocking=True), weight=wbt) \
                 + a.lam * rot_loss(rot.reshape(-1, 6), Rg)            # equal chordal over ALL K anchors
            opt.zero_grad(); loss.backward(); opt.step()
            if a.smoke and bi >= 3: break
        sch.step()
        if ep % 5 == 0 or ep == a.epochs - 1 or a.smoke:
            m = evaluate(model, va, dev, max_batches=6 if a.smoke else None)
            score = max(0, m["nomotion_med"] - m["geo_med"]) / 20 + np.nan_to_num(m["auroc_bt"])   # auroc NaN if no positives in subset
            print(f"  ep{ep:3d} val geo_med {m['geo_med']:.2f}° (no-motion {m['nomotion_med']:.2f}°) "
                  f"tail {m['tail_geo_med']:.1f}/{m['tail_nomotion_med']:.1f}° (n={m['tail_n']}) "
                  f"auroc {m['auroc_bt']:.3f} ({(time.time()-t0)/60:.1f}m)", flush=True)
            if score > best: best = score; torch.save(model.state_dict(), f"{out}/best.pt")
        if a.smoke: break
    model.load_state_dict(torch.load(f"{out}/best.pt"))
    te = DropTrajDS("test", modality="full", frame=a.frame)
    m = evaluate(model, te, dev, max_batches=6 if a.smoke else None, dump=None if a.smoke else f"{out}/test_pred.npz")
    print(f"[TEST tag={a.tag}] {json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()})}")
    json.dump(dict(tag=a.tag, K=K, frame=a.frame, **m), open(f"{out}/summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
