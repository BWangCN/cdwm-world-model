"""Train the clean multi-task grasp WM FROM SCRATCH (the locked ideal workflow):
  - ONE dataset  = outcomes_v2 (all outcomes), unified `GraspMTLDS`.
  - ONE model    = `GraspMTL` (shared trunk + private trajectory/boolean branches).
  - per-batch multi-task loss: diffusion eps+geodesic on the trajectory MASKED to rigid episodes + boolean CE on ALL
    outcomes, Kendall-weighted. No warm-start, no freeze.
  - use = classify -> gate (discard slip) -> roll out the trajectory.
    python train_grasp_mtl.py --tag grasp_mtl --epochs 200            # (smoke: --smoke)
"""
import os, sys, math, time, json, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from my_dataset_grasp_mtl import GraspMTLDS
from wm.grasp_mtl import GraspMTL
from wm.dit import cosine_acp, ddim_sample
from wm.common import sixd_to_R, geodesic_deg


def t_sixd_to_R(d):
    a, b = d[..., :3], d[..., 3:]
    x = a / (a.norm(dim=-1, keepdim=True) + 1e-8); b = b - (x * b).sum(-1, keepdim=True) * x
    y = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    return torch.stack([x, y, torch.cross(x, y, dim=-1)], -1)


def t_geo_rad(Rp, Rg):
    m = torch.matmul(Rp.transpose(-1, -2), Rg); tr = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    return torch.arccos(torch.clamp((tr - 1) / 2, -1 + 1e-6, 1 - 1e-6))


def auroc(y, s):
    o = np.argsort(-s); y = y[o]; P = y.sum(); N = len(y) - P
    return float(np.trapz(np.cumsum(y) / P, np.cumsum(1 - y) / N)) if P and N else float("nan")


@torch.no_grad()
def evaluate(model, dl, dev, acp, sd, mu, nb=12):
    model.eval(); G, PS, YS, GP, GT, GPH = [], [], [], [], [], []
    for i, b in enumerate(dl):
        if i >= nb: break
        cond = model.encode(*[b[k].to(dev) for k in ("pts", "base_rel", "closing", "table")])
        PS.append(torch.softmax(model.classify(cond), 1)[:, 1].cpu().numpy()); YS.append(b["y_slip"].numpy())
        GP.append(model.predict_grip(cond).clamp(0, 1).cpu().numpy()); GT.append(b["grip_tgt"].numpy()); GPH.append(b["grip_phase"].numpy())
        rig = b["is_rigid"].bool()
        if rig.any():                                                # endpoint geodesic on rigid only
            x0 = ddim_sample(model, cond[rig], 32, acp, steps=25, device=dev)[:, -1, :].cpu().numpy() * sd + mu
            gt = b["target"].numpy()[rig.numpy(), -1, :] * sd + mu
            G.append(geodesic_deg(sixd_to_R(x0[:, 3:9]), sixd_to_R(gt[:, 3:9])))
    g = np.concatenate(G) if G else np.array([float("nan")]); P = np.concatenate(PS); Y = np.concatenate(YS)
    gp, gt, gph = np.concatenate(GP), np.concatenate(GT), np.concatenate(GPH)     # (M,H)
    mae_rad = float(np.abs(gp - gt).mean() * 0.8)                                 # achieved-closure MAE in rad
    bp_, bt_ = np.clip((gp * 100).astype(int), 0, 99), np.clip((gt * 100).astype(int), 0, 99)   # 100-state [0,99]
    acc = float((bp_ == bt_).mean())
    close, hold = gph == 1, gph == 2                                             # phase-sliced (close ramp vs settled hold)
    acc_c = float((bp_[close] == bt_[close]).mean()) if close.any() else float("nan")
    acc_h = float((bp_[hold] == bt_[hold]).mean()) if hold.any() else float("nan")
    return float(np.median(g)), auroc(Y, P), mae_rad, acc, acc_c, acc_h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="grasp_mtl"); ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--bs", type=int, default=128); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam_aux", type=float, default=0.05); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = f"gated_runs/{a.tag}"; os.makedirs(out, exist_ok=True)

    tr = GraspMTLDS("train", seed=a.seed); va = GraspMTLDS("val", stats=tr.stats)
    sd = torch.tensor(tr.stats["std"], device=dev); mu = torch.tensor(tr.stats["mean"], device=dev)
    sdn, mun = tr.stats["std"], tr.stats["mean"]
    freq = np.array([(tr.is_rigid == 1).sum(), (tr.is_rigid == 0).sum()], float)          # [rigid, slip]
    cw = torch.tensor(freq.sum() / (2 * freq + 1e-6), dtype=torch.float32, device=dev)
    print(f"[grasp_mtl] train {len(tr)} (rigid {tr.rigid_frac():.2f}) val {len(va)} | cw {cw.tolist()}", flush=True)

    model = GraspMTL(n_feat=13, H=32, D=256, depth=4).to(dev)                              # FROM SCRATCH
    acp = cosine_acp(1000, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    dl = DataLoader(tr, a.bs, shuffle=True, num_workers=8, drop_last=True, persistent_workers=True, pin_memory=True)
    dlv = DataLoader(va, a.bs, num_workers=4)

    t0 = time.time(); best = 1e9
    for ep in range(a.epochs):
        model.train()
        for b in dl:
            cond = model.encode(*[b[k].to(dev, non_blocking=True) for k in ("pts", "base_rel", "closing", "table")])
            y = b["target"].to(dev, non_blocking=True); w = b["is_rigid"].to(dev)          # (B,) rigid mask
            t = torch.randint(0, len(acp), (y.shape[0],), device=dev)
            noise = torch.randn_like(y); at = acp[t][:, None, None]; xt = at.sqrt() * y + (1 - at).sqrt() * noise
            eps = model(xt, t, cond); dmse_p = ((eps - noise) ** 2).mean([1, 2])
            x0 = ((xt - (1 - at).sqrt() * eps) / at.sqrt()).clamp(-8, 8)
            aux_p = (t_geo_rad(t_sixd_to_R((x0 * sd + mu)[..., 3:9]), t_sixd_to_R((y * sd + mu)[..., 3:9])) / math.pi).mean(1)
            den = w.sum().clamp_min(1.0)
            dmse = (dmse_p * w).sum() / den; aux = (aux_p * w).sum() / den                 # trajectory MASKED to rigid
            ce = F.cross_entropy(model.classify(cond), b["y_slip"].to(dev), weight=cw)      # boolean on ALL
            grip_l = F.smooth_l1_loss(model.predict_grip(cond), b["grip_tgt"].to(dev, non_blocking=True))  # achieved closure on ALL (Huber)
            s = model.log_s
            loss = torch.exp(-s[0]) * (dmse + a.lam_aux * aux) + s[0] + torch.exp(-s[1]) * ce + s[1] + torch.exp(-s[2]) * grip_l + s[2]
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
            if a.smoke: break
        sch.step()
        if ep % 10 == 0 or ep == a.epochs - 1 or a.smoke:
            geo, au, gmae, gacc, gacc_c, gacc_h = evaluate(model, dlv, dev, acp, sdn, mun, nb=3 if a.smoke else 12)
            print(f"  ep{ep:3d} val traj_geo {geo:.2f} bool_auroc {au:.3f} | grip mae {gmae:.4f}rad acc100 {gacc:.3f} "
                  f"(close {gacc_c:.3f} hold {gacc_h:.3f}) ({(time.time()-t0)/60:.1f}m)", flush=True)
            score = geo + 8.0 * (1 - au)                                                    # want BOTH low geo and high auroc (grip = aux supervision)
            if score < best:
                best = score
                torch.save({"model": model.state_dict(), "H": 32, "T": 1000, "dim": 256, "depth": 4, "n_feat": 13,
                            "arch": "grasp_mtl", "stats": tr.stats, "ep": ep, "val_traj_geo": geo, "val_auroc": au,
                            "val_grip_mae_rad": gmae, "val_grip_acc100": gacc}, f"{out}/best.pt")
        if a.smoke: break
    json.dump(dict(tag=a.tag, best_score=best), open(f"{out}/summary.json", "w"), indent=1)
    print(f"[DONE {a.tag}] best {best:.3f} -> {out}/best.pt", flush=True)


if __name__ == "__main__":
    main()
