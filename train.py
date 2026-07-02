"""CLEAN A/B trainer for the direction-targeted geodesic-aux loss (V1). IDENTICAL to the baseline stable-187 colorless
trainer (imports the SAME my_config/my_dataset/DiTWM from out/2026-06-30_stable187-wm-train -> same data, split, 12-feat
representation, D256/depth4 architecture, H=32, per-channel standardization + z-clamp), EXCEPT the aux loss term.

THE ONLY VARIABLE is --lam_aux:
  loss = eps_MSE  +  lam_aux * mean_over_batch( mean_over_H( geodesic(R(x0_hat), R(target)) / pi ) )
where x0_hat is recovered from the predicted eps (clamp +-8, z-score space), un-z-scored, and the 6D rotation [3:9]
decoded via Gram-Schmidt. V1 = plain batch mean, all diffusion-t equal, no detach (the exact bowl-winning formulation;
geodesic in radians/pi in ~[0,1] so lam=0.05 is a true relative weight). lam_aux=0 -> byte-identical to the baseline.
Budget by EPOCHS (not wall-time) so parallel runs get equal training. Run: my_train_aux.py --seed S --lam_aux L --tag TAG."""
import os, sys, time, json, math, argparse
import numpy as np, torch
from torch.utils.data import DataLoader, WeightedRandomSampler
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
import my_config as C
from wm.dit import DiTWM, cosine_acp, ddim_sample
from wm.common import sixd_to_R, geodesic_deg
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt


def t_sixd_to_R(d):                                  # (...,6)->(...,3,3) Gram-Schmidt, columns [x,y,z] (matches wm.common.sixd_to_R)
    a, b = d[..., :3], d[..., 3:]
    x = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b = b - (x * b).sum(-1, keepdim=True) * x
    y = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    z = torch.cross(x, y, dim=-1)
    return torch.stack([x, y, z], dim=-1)


def t_geo_rad(Rp, Rg):                               # geodesic angle(Rp^T Rg), radians, differentiable
    Rr = Rp.transpose(-1, -2) @ Rg
    tr = Rr[..., 0, 0] + Rr[..., 1, 1] + Rr[..., 2, 2]
    return torch.arccos(((tr - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6))


@torch.no_grad()
def matters(model, ds, dev, acp, H, sd, mu, N=4, steps=40, thr=15.0):
    if len(ds) == 0: return float("nan"), float("nan"), 0
    dl = DataLoader(ds, 128, shuffle=False, num_workers=3); conds, gts, tl = [], [], []
    model.eval()
    for b in dl:
        conds.append(model.encode(b["pts"].to(dev), b["base_rel"].to(dev), b["closing"].to(dev), b["table"].to(dev)))
        gts.append(b["target"].numpy()); tl += list(b["tilt"].numpy())
    G = np.concatenate(gts) * sd + mu; tl = np.array(tl); M = len(G)
    cond = torch.cat(conds); geo = np.zeros((N, M)); pt = np.zeros((N, M))
    for n in range(N):
        x0 = np.concatenate([ddim_sample(model, cond[i:i+256], H, acp, steps=steps, device=dev).cpu().numpy() for i in range(0, M, 256)]) * sd + mu
        geo[n] = geodesic_deg(sixd_to_R(x0[:, -1, 3:]), sixd_to_R(G[:, -1, 3:]))
        pt[n] = geodesic_deg(sixd_to_R(x0[:, -1, 3:]), np.tile(np.eye(3), (M, 1, 1)))
    gb = geo.min(0); pred_tilt = pt.mean(0); m = tl > thr
    r = float(np.corrcoef(pred_tilt[m], tl[m])[0, 1]) if m.sum() >= 3 else float("nan")
    return (float(np.median(gb[m])) if m.sum() else float("nan")), r, int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--lam_aux", type=float, default=0.0)
    ap.add_argument("--tag", default="l000"); ap.add_argument("--var", default="v1")
    ap.add_argument("--outroot", default="wm_runs")
    ap.add_argument("--H", type=int, default=C.H); ap.add_argument("--time_budget_min", type=float, default=120)
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--T", type=int, default=1000); ap.add_argument("--epochs", type=int, default=460)   # fixed-epoch budget -> equal training across conditions
    ap.add_argument("--dim", type=int, default=256); ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--patience", type=int, default=100000); ap.add_argument("--eval_every", type=int, default=20)
    ap.add_argument("--epoch_draws", type=int, default=1024); ap.add_argument("--ckpt_every", type=int, default=0)   # save ep{N}.pt series for dir_err-vs-epoch
    ap.add_argument("--dataset_mod", default="my_dataset")   # swappable dataset (my_dataset | my_dataset_frame | ...) — the ONLY A/B variable
    a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; LAM = a.lam_aux
    _DS = importlib.import_module(a.dataset_mod); ChunkDS, load_rows, compute_stats = _DS.ChunkDS, _DS.load_rows, _DS.compute_stats
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    outdir = f"{a.outroot}/colorless_s{a.seed}/{a.tag}"; ck = f"{outdir}/ckpt"; os.makedirs(ck, exist_ok=True)
    def log(m): print(m, flush=True); open(f"{outdir}/train.log", "a").write(m + "\n")

    tr = load_rows("train"); va = load_rows("valgrasp"); ho = load_rows("heldout_object")
    stats = compute_stats(tr, a.H, seed=a.seed); np.savez(f"{outdir}/norm_stats.npz", **stats); sd, mu = stats["std"], stats["mean"]
    sd_t = torch.tensor(sd, device=dev, dtype=torch.float32); mu_t = torch.tensor(mu, device=dev, dtype=torch.float32)
    DStr = ChunkDS(tr, a.H, stats=stats, seed=a.seed); DSva = ChunkDS(va, a.H, stats=stats, fixed_subsample=True)
    DSho = ChunkDS(ho, a.H, stats=stats, fixed_subsample=True)
    log(f"[s{a.seed} lam={LAM} var={a.var}] n_feat={C.N_FEAT} train {len(tr)} valgrasp {len(va)} heldobj {len(ho)}")
    model = DiTWM(n_feat=C.N_FEAT, H=a.H, D=a.dim, depth=a.depth).to(dev)
    npar = sum(p.numel() for p in model.parameters()); log(f"{npar/1e6:.2f}M params")
    acp = cosine_acp(a.T, device=dev); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    def mk_loader(bs):
        w = DStr.weights(); samp = WeightedRandomSampler(torch.as_tensor(w), max(a.epoch_draws, len(w)), replacement=True)
        return DataLoader(DStr, bs, sampler=samp, num_workers=4, drop_last=True, persistent_workers=True)
    dlva = DataLoader(DSva, 64, shuffle=False, num_workers=2, persistent_workers=True)

    def loss_b(b, train):                            # returns (total, eps_mse, aux)  -- RNG order (t,noise) identical to baseline
        pts, br, cl, tb, y = (b[k].to(dev) for k in ["pts", "base_rel", "closing", "table", "target"])
        with torch.set_grad_enabled(train):
            cond = model.encode(pts, br, cl, tb); t = torch.randint(0, a.T, (y.shape[0],), device=dev)
            noise = torch.randn_like(y); at = acp[t][:, None, None]; xt = at.sqrt() * y + (1 - at).sqrt() * noise
            eps_hat = model(xt, t, cond); le = ((eps_hat - noise) ** 2).mean()
            if LAM == 0.0: return le, le, torch.zeros((), device=dev)
            x0 = ((xt - (1 - at).sqrt() * eps_hat) / at.sqrt()).clamp(-8, 8)      # x0_hat (z-score space), as in ddim_sample
            Rp = t_sixd_to_R((x0 * sd_t + mu_t)[..., 3:9]); Rg = t_sixd_to_R((y * sd_t + mu_t)[..., 3:9])
            aux_per = (t_geo_rad(Rp, Rg) / math.pi).mean(1)                        # (B,) mean over H, radians/pi
            la = aux_per.mean() if a.var == "v1" else (acp[t].sqrt().detach() * aux_per).sum() / (acp[t].sqrt().detach().sum() + 1e-8)
            return le + LAM * la, le, la

    def epoch(loader, train):
        model.train(train); tot = tle = tla = n = nskip = 0
        for b in loader:
            l, le, la = loss_b(b, train)
            if not torch.isfinite(l): nskip += 1; continue
            if train:
                opt.zero_grad(); l.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                if torch.isfinite(gn): opt.step()
                else: nskip += 1
            bs = b["target"].shape[0]; tot += l.item()*bs; tle += float(le)*bs; tla += float(la)*bs; n += bs
        return (tot/n, tle/n, tla/n) if n else (float("nan"),)*3

    def save(p, **e): torch.save({"model": model.state_dict(), "H": a.H, "T": a.T, "dim": a.dim, "depth": a.depth, "n_feat": C.N_FEAT, "lam_aux": LAM, **e}, p)
    batch = a.batch; dltr = mk_loader(batch); t0 = time.time(); best = 1e9; bgeo = 1e9; pat = 0; ep = 0; lc = t0; curve = []; geo_curve = []
    while ep < a.epochs and (time.time() - t0) / 60 < a.time_budget_min:
        try:
            ltr, ltr_e, ltr_a = epoch(dltr, True); lva, lva_e, lva_a = epoch(dlva, False); sched.step()
            if not (math.isfinite(ltr) and math.isfinite(lva)): raise FloatingPointError()
            curve.append([ep, round(ltr, 6), round(lva, 6), round(ltr_e, 6), round(ltr_a, 6)])
            if lva < best - 1e-5: best = lva; pat = 0; save(f"{ck}/best.pt", val=best)
            else: pat += 1
            if ep % a.eval_every == 0 and ep > 0:
                vg, r, nm = matters(model, DSva, dev, acp, a.H, sd, mu)
                geo_curve.append([ep, round(vg, 3) if math.isfinite(vg) else None, round(r, 3) if math.isfinite(r) else None])
                if math.isfinite(vg) and vg < bgeo: bgeo = vg
                log(f"  ep{ep} val {lva:.4f} (eps {lva_e:.4f} aux {lva_a:.4f}) valgrasp>15geo {vg:.2f} r {r:.2f} n{nm}")
            if time.time() - lc > 600: save(f"{ck}/periodic.pt"); lc = time.time()
            if a.ckpt_every and ep % a.ckpt_every == 0 and ep > 0: save(f"{ck}/ep{ep:04d}.pt", val=round(lva, 6))   # checkpoint series
            if ep % 25 == 0 or pat == 0: log(f"ep{ep:4d} tr {ltr:.4f}(eps{ltr_e:.4f} aux{ltr_a:.4f}) val {lva:.4f} best {best:.4f} {(time.time()-t0)/60:.1f}m")
            if pat >= a.patience: log("early stop"); break
            ep += 1
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); batch = max(16, batch // 2); dltr = mk_loader(batch); log(f"OOM -> batch {batch}")
        except FloatingPointError:
            newlr = opt.param_groups[0]["lr"] * 0.5; log(f"NaN epoch -> reload best + RESET optimizer + lr {newlr:.1e}")
            if os.path.exists(f"{ck}/best.pt"): model.load_state_dict(torch.load(f"{ck}/best.pt")["model"])
            opt = torch.optim.AdamW(model.parameters(), lr=newlr, weight_decay=1e-4); ep += 1
    if not os.path.exists(f"{ck}/best.pt"): save(f"{ck}/best.pt", val=best)
    model.load_state_dict(torch.load(f"{ck}/best.pt")["model"])
    vg, vr, vn = matters(model, DSva, dev, acp, a.H, sd, mu, N=8); hg, hr, hn = matters(model, DSho, dev, acp, a.H, sd, mu, N=8)
    peak = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0
    json.dump(dict(epochs=ep, curve=curve, geo_curve=geo_curve), open(f"{outdir}/curves.json", "w"))
    summ = dict(seed=a.seed, lam_aux=LAM, var=a.var, best_val=round(best, 6), sel="best.pt(val=eps+lam*aux)",
                valgrasp_geo15=round(vg, 3), valgrasp_r=round(vr, 3), heldobj_geo15=round(hg, 3), heldobj_r=round(hr, 3),
                n_train=len(tr), n_heldobj=len(ho), params=npar, epochs=ep, wall_min=round((time.time()-t0)/60, 1),
                peak_vram_gb=round(peak, 2), final_batch=batch, diverged=bool(not math.isfinite(best) or best > 0.5))
    json.dump(summ, open(f"{outdir}/train_summary.json", "w"), indent=2)
    cv = np.array(curve); fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(cv[:, 0], cv[:, 3], label="train eps-MSE", lw=1.2); ax[0].plot(cv[:, 0], cv[:, 2], label="val (eps+aux)", lw=1.2)
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].set_title(f"s{a.seed} lam={LAM}"); ax[0].legend(); ax[0].grid(alpha=.3)
    if geo_curve:
        gc = np.array([[g[0], g[1] if g[1] is not None else np.nan, g[2] if g[2] is not None else np.nan] for g in geo_curve], float)
        ax[1].plot(gc[:, 0], gc[:, 1], "-o", ms=3, label="valgrasp>15 geo"); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("geo(deg)"); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{outdir}/curve.png", dpi=120); plt.close()
    log(f"DONE s{a.seed} lam={LAM} best_val {best:.4f} | valgrasp>15 geo {vg:.2f} r {vr:.2f} | heldobj geo {hg:.2f} r {hr:.2f} | {ep}ep {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
