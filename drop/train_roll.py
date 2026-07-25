"""Drop ROLLOUT WM training: GateBDiT with H=K denoises the K-step settling trajectory (RollDS), not just the
endpoint. Same latent arms as Gate B (no_latent / abstract / grounded / shuffled), object-disjoint split. The model
imagines the whole settle; under a hidden CoM the samples diverge into different basins.

    python train_roll.py --arm grounded_oracle --tag grounded_roll --epochs 120
"""
import os, sys, json, time, argparse
import numpy as np, torch
from torch.utils.data import DataLoader
from drop.my_dataset_roll import RollDS, K
from drop.drop_diffusion import GateBDiT, diffusion_loss, cosine_acp, ddim_sample

LAT = {"no_latent": ("none", False, 13), "abstract_oracle": ("abstract", True, 13),
       "grounded_oracle": ("grounded", False, 17), "shuffled_oracle": ("shuffled", False, 17)}


@torch.no_grad()
def val_metric(model, ds, dev, acp, use_lat, max_batches=None):
    dl = DataLoader(ds, 256, num_workers=6); model.eval(); vals = []
    for bi, b in enumerate(dl):
        if max_batches and bi >= max_batches: break
        pts, br, cl, tb = [b[k].to(dev) for k in ("pts", "base_rel", "closing", "table")]
        cond = model.cond(pts, br, cl, tb, b["latent"].to(dev) if use_lat else None)
        vals.append(diffusion_loss(model, b["target"].to(dev), cond, acp).item())
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="grounded_oracle", choices=list(LAT))
    ap.add_argument("--tag", default=None); ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--bs", type=int, default=128); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); tag = a.tag or a.arm
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(a.seed); np.random.seed(a.seed)
    lm, USE_LAT, NF = LAT[a.arm]; os.environ["CDWM_GATEB_LATENT"] = lm
    mode = os.environ.get("CDWM_GATEB_SPLIT", "object"); pre = "" if mode == "object" else f"{mode}_"
    out = f"roll_runs/{pre}{tag}"; os.makedirs(out, exist_ok=True)

    tr = RollDS("train", seed=a.seed); va = RollDS("val", stats=tr.stats)
    print(f"[roll arm={a.arm} K={K} nfeat={NF}] train {len(tr)} ({len(set(tr.obj))} obj) val {len(va)} ({len(set(va.obj))} obj)")
    st = {k: torch.tensor(v, device=dev) for k, v in tr.stats.items()}
    model = GateBDiT(n_feat=NF, use_latent=USE_LAT, H=K).to(dev)
    acp = cosine_acp(1000, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    dltr = DataLoader(tr, a.bs, shuffle=True, num_workers=8, drop_last=True, persistent_workers=True, pin_memory=True)
    t0 = time.time(); best = 1e9
    for ep in range(a.epochs):
        model.train()
        for bi, b in enumerate(dltr):
            pts, br, cl, tb = [b[k].to(dev, non_blocking=True) for k in ("pts", "base_rel", "closing", "table")]
            cond = model.cond(pts, br, cl, tb, b["latent"].to(dev) if USE_LAT else None)
            loss = diffusion_loss(model, b["target"].to(dev, non_blocking=True), cond, acp)
            opt.zero_grad(); loss.backward(); opt.step()
            if a.smoke and bi >= 3: break
        sch.step()
        if ep % 10 == 0 or ep == a.epochs - 1 or a.smoke:
            vm = val_metric(model, va, dev, acp, USE_LAT, max_batches=4 if a.smoke else None)
            print(f"  ep{ep:3d} val diff_loss {vm:.4f} ({(time.time()-t0)/60:.1f}m)")
            if vm < best: best = vm; torch.save(model.state_dict(), f"{out}/best.pt")
        if a.smoke: break
    b = next(iter(DataLoader(va, 8)))
    cond = model.cond(*[b[k].to(dev) for k in ("pts", "base_rel", "closing", "table")],
                      b["latent"].to(dev) if USE_LAT else None)
    x0 = ddim_sample(model, cond, K, acp, steps=25, device=dev)
    print(f"  sample check: ddim x0 {tuple(x0.shape)} finite={torch.isfinite(x0).all().item()}")
    json.dump(dict(arm=a.arm, tag=tag, K=K, best_val=best, n_train=len(tr)), open(f"{out}/summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
