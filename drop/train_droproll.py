"""Corpus diffusion ROLLOUT WM (Codex-converged): GateBDiT with H=K=8 denoises the contact->settle anchor
trajectory on the NATURAL corpus (no latent -- the corpus has no hidden CoM). Same anchors as the deterministic
traj_k8 arm for comparability. Final test reports: diffusion val loss, endpoint geodesic (best-of-S and
median-of-S) vs the deterministic arm, tip-recall on the CoQ10 transition episodes with a DATA-DERIVED cutoff
(largest gap in the CoQ10 ground-truth net-rotation distribution), and sample diversity.

    python -m drop.train_droproll --tag roll_corpus --epochs 30
"""
import os, json, time, argparse
import numpy as np, torch
from torch.utils.data import DataLoader
from drop.my_dataset_drop import DropRollDS
from drop.drop_diffusion import GateBDiT, diffusion_loss
from common.dit import cosine_acp, ddim_sample
from drop.drop_net import sixd_to_R
from drop.train_drop import geodesic
from common.paths import REPO


@torch.no_grad()
def val_loss(model, ds, dev, acp, max_batches=None):
    dl = DataLoader(ds, 256, num_workers=6); model.eval(); vals = []
    for bi, b in enumerate(dl):
        if max_batches and bi >= max_batches: break
        cond = model.cond(*[b[k].to(dev) for k in ("pts", "base_rel", "closing", "table")], None)
        vals.append(diffusion_loss(model, b["target"].to(dev), cond, acp).item())
    return float(np.mean(vals))


@torch.no_grad()
def test_metrics(model, te, dev, acp, S=4, max_batches=None, dump=None):
    K = te.K; mean = torch.tensor(te.stats["mean"], device=dev); std = torch.tensor(te.stats["std"], device=dev)
    dl = DataLoader(te, 128, num_workers=6); model.eval()
    GEO_B, GEO_M, NET, OBJ, GEO0 = [], [], [], [], []
    for bi, b in enumerate(dl):
        if max_batches and bi >= max_batches: break
        B = len(b["table"])
        cond = model.cond(*[b[k].to(dev) for k in ("pts", "base_rel", "closing", "table")], None)
        x0 = ddim_sample(model, cond.repeat_interleave(S, 0), K, acp, steps=25, device=dev)
        dR = sixd_to_R((x0 * std + mean)[:, -1, :6]).view(B, S, 3, 3)          # final anchor per sample
        Rg = sixd_to_R(b["rest6d"].to(dev)); I = torch.eye(3, device=dev).expand_as(Rg)
        g = geodesic(dR, Rg[:, None].expand_as(dR))                            # (B,S)
        GEO_B.append(g.min(1).values.cpu()); GEO_M.append(g.median(1).values.cpu())
        NET.append(geodesic(dR, I[:, None].expand_as(dR)).cpu())               # sampled net rotation (B,S)
        GEO0.append(geodesic(I, Rg).cpu()); OBJ += list(b["object"])
    geo_b = torch.cat(GEO_B).numpy(); geo_m = torch.cat(GEO_M).numpy()
    net = torch.cat(NET).numpy(); geo0 = torch.cat(GEO0).numpy(); obj = np.array(OBJ)
    if dump: np.savez(dump, geo_best=geo_b, geo_med=geo_m, net=net, geo0=geo0, obj=obj)
    out = dict(endpoint_bestS_med=float(np.median(geo_b)), endpoint_medS_med=float(np.median(geo_m)), n=len(obj))
    # CoQ10 tip-recall with a DATA-DERIVED cutoff (largest gap in the GT net-rotation distribution of that object)
    co = obj == "CoQ10_BjTLbuRVt1t"
    if co.any():
        g0 = np.sort(geo0[co]); gaps = np.diff(g0); gi = int(gaps.argmax())
        cut = float((g0[gi] + g0[gi + 1]) / 2)
        tipped_gt = geo0[co] > cut; sample_tips = net[co] > cut                # (n_co, S)
        rec = float((sample_tips[tipped_gt].any(1)).mean()) if tipped_gt.any() else float("nan")
        both = float(((sample_tips.any(1)) & (~sample_tips.all(1)))[tipped_gt].mean()) if tipped_gt.any() else float("nan")
        out.update(coq10_cutoff_deg=round(cut, 1), coq10_n_tips=int(tipped_gt.sum()),
                   coq10_tip_recall=rec, coq10_diverse_frac=both)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="roll_corpus"); ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=256); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = f"{REPO}/drop_runs/{a.tag}"; os.makedirs(out, exist_ok=True)
    tr = DropRollDS("train", seed=a.seed); va = DropRollDS("val", stats=tr.stats)
    K = tr.K
    print(f"[droproll tag={a.tag} K={K}] train {len(tr)} val {len(va)}", flush=True)
    model = GateBDiT(n_feat=13, use_latent=False, H=K).to(dev)
    acp = cosine_acp(1000, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    dltr = DataLoader(tr, a.bs, shuffle=True, num_workers=8, drop_last=True, persistent_workers=True, pin_memory=True)
    t0 = time.time(); best = 1e9
    for ep in range(a.epochs):
        model.train()
        for bi, b in enumerate(dltr):
            cond = model.cond(*[b[k].to(dev, non_blocking=True) for k in ("pts", "base_rel", "closing", "table")], None)
            loss = diffusion_loss(model, b["target"].to(dev, non_blocking=True), cond, acp)
            opt.zero_grad(); loss.backward(); opt.step()
            if a.smoke and bi >= 3: break
        sch.step()
        if ep % 5 == 0 or ep == a.epochs - 1 or a.smoke:
            vm = val_loss(model, va, dev, acp, max_batches=4 if a.smoke else None)
            print(f"  ep{ep:3d} val diff_loss {vm:.4f} ({(time.time()-t0)/60:.1f}m)", flush=True)
            if vm < best: best = vm; torch.save(model.state_dict(), f"{out}/best.pt")
        if a.smoke: break
    model.load_state_dict(torch.load(f"{out}/best.pt"))
    te = DropRollDS("test", stats=tr.stats)
    m = test_metrics(model, te, dev, acp, S=4, max_batches=4 if a.smoke else None,
                     dump=None if a.smoke else f"{out}/test_samples.npz")
    m["val_diff_loss"] = best
    print(f"[TEST tag={a.tag}] {json.dumps({k: round(v,3) if isinstance(v,float) else v for k,v in m.items()})}")
    json.dump(dict(tag=a.tag, K=K, **m), open(f"{out}/summary.json", "w"), indent=1)
    np.savez(f"{out}/stats.npz", **tr.stats)


if __name__ == "__main__":
    main()
