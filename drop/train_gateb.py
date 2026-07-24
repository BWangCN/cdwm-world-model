"""Gate B distributional-WM training. Arms (--arm): point (deterministic regressor) | diff (no-latent diffusion,
CoM hidden) | diff_oracle (diffusion conditioned on the true CoM = ceiling). DiT diffusion head (H=1, resting SE(3)),
shared DiTWMLocal encoder. Object-disjoint split. Full basin-probability eval is in eval_gateb.py; here we train +
checkpoint on a per-arm val metric.

    python train_gateb.py --arm diff --tag diff --epochs 120
"""
import os, sys, json, time, argparse
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from drop.my_dataset_gateb import GateBDS
from drop.drop_diffusion import GateBDiT, GateBPoint, diffusion_loss, cosine_acp, ddim_sample
from drop.drop_net import sixd_to_R


def geodesic(Ra, Rb):
    Rr = torch.matmul(Ra, Rb.transpose(-1, -2))
    tr = torch.clamp((Rr[..., 0, 0] + Rr[..., 1, 1] + Rr[..., 2, 2] - 1) / 2, -1, 1)
    return torch.rad2deg(torch.arccos(tr))


def unnorm(x, st):                                            # z-scored (B,9) -> rotation matrix from first 6d
    return sixd_to_R((x * st["std"] + st["mean"])[..., :6])


@torch.no_grad()
def val_metric(model, ds, dev, arm, acp, st, use_lat=False, max_batches=None):
    dl = DataLoader(ds, 256, num_workers=6); model.eval(); vals = []
    for bi, b in enumerate(dl):
        if max_batches and bi >= max_batches: break
        pts, br, cl, tb = [b[k].to(dev) for k in ("pts", "base_rel", "closing", "table")]
        tgt = b["target"].to(dev)                            # (B,1,9)
        if arm == "point":
            pred = model.predict(pts, br, cl, tb)
            vals.append(geodesic(unnorm(pred, st), unnorm(tgt.squeeze(1), st)).cpu())
        else:
            cond = model.cond(pts, br, cl, tb, b["latent"].to(dev) if use_lat else None)
            vals.append(torch.tensor([diffusion_loss(model, tgt, cond, acp).item()]))
    v = torch.cat(vals)
    return float(v.median()) if arm == "point" else float(v.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="diff", choices=["point", "diff", "diff_oracle",
                    "no_latent", "abstract_oracle", "grounded_oracle", "shuffled_oracle"])
    ap.add_argument("--tag", default=None); ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--bs", type=int, default=256); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); tag = a.tag or a.arm
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    # cross-object transfer arms: (latent_mode, use_latent, n_feat). Set the latent-encoding env BEFORE GateBDS.
    LAT = {"no_latent": ("none", False, 13), "abstract_oracle": ("abstract", True, 13),
           "grounded_oracle": ("grounded", False, 17), "shuffled_oracle": ("shuffled", False, 17)}
    if a.arm in LAT:
        _lm, USE_LAT, NF = LAT[a.arm]; os.environ["CDWM_GATEB_LATENT"] = _lm
    else:
        USE_LAT, NF = (a.arm == "diff_oracle"), 13
    IS_POINT = (a.arm == "point")
    mode = os.environ.get("CDWM_GATEB_SPLIT", "object"); pre = "" if mode == "object" else f"{mode}_"
    out = f"gateb_runs/{pre}{tag}"; os.makedirs(out, exist_ok=True)

    tr = GateBDS("train", seed=a.seed); va = GateBDS("val", stats=tr.stats)
    print(f"[gateb arm={a.arm} latent={os.environ.get('CDWM_GATEB_LATENT','-')} nfeat={NF}] "
          f"train {len(tr)} ({len(set(tr.obj))} obj) val {len(va)} ({len(set(va.obj))} obj)")
    st = {k: torch.tensor(v, device=dev) for k, v in tr.stats.items()}
    model = (GateBPoint() if IS_POINT else GateBDiT(n_feat=NF, use_latent=USE_LAT)).to(dev)
    acp = cosine_acp(1000, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    dltr = DataLoader(tr, a.bs, shuffle=True, num_workers=8, drop_last=True, persistent_workers=True, pin_memory=True)
    t0 = time.time(); best = 1e9
    for ep in range(a.epochs):
        model.train()
        for bi, b in enumerate(dltr):
            pts, br, cl, tb = [b[k].to(dev, non_blocking=True) for k in ("pts", "base_rel", "closing", "table")]
            tgt = b["target"].to(dev, non_blocking=True)
            if IS_POINT:
                loss = F.mse_loss(model.predict(pts, br, cl, tb), tgt.squeeze(1))
            else:
                cond = model.cond(pts, br, cl, tb, b["latent"].to(dev) if USE_LAT else None)
                loss = diffusion_loss(model, tgt, cond, acp)
            opt.zero_grad(); loss.backward(); opt.step()
            if a.smoke and bi >= 3: break
        sch.step()
        if ep % 10 == 0 or ep == a.epochs - 1 or a.smoke:
            vm = val_metric(model, va, dev, "point" if IS_POINT else "diff", acp, st,
                            use_lat=USE_LAT, max_batches=4 if a.smoke else None)
            print(f"  ep{ep:3d} val {'geo°' if IS_POINT else 'diff_loss'} {vm:.4f} ({(time.time()-t0)/60:.1f}m)")
            if vm < best: best = vm; torch.save(model.state_dict(), f"{out}/best.pt")
        if a.smoke: break
    # smoke: verify sampling runs for diffusion arms
    if not IS_POINT:
        b = next(iter(DataLoader(va, 8)))
        cond = model.cond(*[b[k].to(dev) for k in ("pts", "base_rel", "closing", "table")],
                          b["latent"].to(dev) if USE_LAT else None)
        x0 = ddim_sample(model, cond, 1, acp, steps=25, device=dev)
        print(f"  sample check: ddim x0 shape {tuple(x0.shape)} finite={torch.isfinite(x0).all().item()}")
    json.dump(dict(arm=a.arm, tag=tag, best_val=best, n_train=len(tr)), open(f"{out}/summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
