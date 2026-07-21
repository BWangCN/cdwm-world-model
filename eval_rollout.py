"""Eval the SHORT-window rollout WM (colleague's 'visualize the future', native K steps from lift-onset, target = motion
relative to onset). Metrics (Codex rigor bar):
 (1) per-frame reconstruction COVERAGE (best-of-N) vs no-motion (stays-seated=identity), reported over the HORIZON CURVE
     (steps 3/5/8/K) and by outcome -> shows the operating horizon from data, not a hardcoded K;
 (2) boolean read-off CALIBRATION (Brier/ECE/AUROC) vs the separate classifier + joint MDN;
 (3) sample DIVERSITY vs P(fail) -> generative WM captures aleatoric unc.
Honest framing: if the boolean isn't better-calibrated than the classifier it's a rollout VISUALIZATION WM, not a superior
risk model; the WM claim is 'near-term contact dynamics are predictable; late transient/persistent split is aleatoric'.
    python eval_rollout.py --tag roll_full [--N 8 --steps 25]
"""
import sys, argparse, numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, ".")
from my_dataset_trajfull import TrajFullDS
from wm.dit_rollout import DiTRollout
from wm.dit import cosine_acp, ddim_sample
from wm.common import sixd_to_R, geodesic_deg
from eval_traj import MODES
L = lambda p: dict(np.load(p, allow_pickle=True))


def brier(p, y): return float(np.mean((p - y) ** 2))
def ece(p, y, bins=15):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if m.any(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)
def auroc(p, y):
    o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(len(p))
    npos = y.sum(); nneg = len(y) - npos
    return float((r[y == 1].sum() - npos * (npos - 1) / 2) / (npos * nneg + 1e-9))


@torch.no_grad()
def sample_metrics(model, ds, dev, acp, N, steps, bs=256):
    dl = DataLoader(ds, bs, shuffle=False, num_workers=6)
    tmean = torch.tensor(ds.tmean); tstd = torch.tensor(ds.tstd)
    tm = ds.tmean; ts = ds.tstd
    G, G1, NOM, ES, NMES, spread, Y, OB = [], [], [], [], [], [], [], []
    Kf = model.H; I6 = np.array([1, 0, 0, 0, 1, 0], np.float32)
    nm_raw = np.tile(np.array([1, 0, 0, 0, 1, 0, 0, 0, 0], np.float32), (Kf, 1))   # no-motion raw (K,9)
    nmz = ((nm_raw - tm) / ts).reshape(-1)                                          # z-scored no-motion vector
    model.eval()
    for b in dl:
        cond = model.encode(b["pts"].to(dev), b["base_rel"].to(dev), b["closing"].to(dev), b["table"].to(dev))
        B = cond.shape[0]
        xs = ddim_sample(model.dit, cond.repeat_interleave(N, 0), Kf, acp, steps=steps, device=dev).cpu()
        xs = (xs * tstd + tmean).view(B, N, Kf, 9).numpy()                # un-z-scored (B,N,K,9)
        gt = b["target_raw"].numpy()                                     # (B,K,9) relative-to-onset
        Rgt = sixd_to_R(gt[..., :6]); Rs = sixd_to_R(xs[..., :6])
        g = geodesic_deg(Rs, Rgt[:, None])                              # (B,N,K)
        G.append(g.min(1)); G1.append(g[:, 0])                          # best-of-N and single-shot per frame (B,K)
        NOM.append(geodesic_deg(np.broadcast_to(sixd_to_R(I6), Rgt.shape), Rgt))   # no-motion=identity (B,K)
        # energy score (proper scoring rule) in z-scored trajectory space (B,) — lower=better
        Xz = ((xs - tm) / ts).reshape(B, N, -1); yz = ((gt - tm) / ts).reshape(B, -1)
        t1 = np.linalg.norm(Xz - yz[:, None], axis=-1).mean(1)          # E||X-y||
        t2 = 0.5 * np.linalg.norm(Xz[:, :, None] - Xz[:, None], axis=-1).mean((1, 2))   # 0.5 E||X-X'||
        ES.append(t1 - t2); NMES.append(np.linalg.norm(yz - nmz, axis=-1))             # no-motion point score
        spread.append(xs[:, :, -1, 8].std(1) * 1000)
        Y.append(b["y5"].numpy()); OB += list(b["object"])
    return (np.concatenate(G), np.concatenate(G1), np.concatenate(NOM),
            np.concatenate(ES), np.concatenate(NMES), np.concatenate(spread), np.concatenate(Y), np.array(OB))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="roll_full")
    ap.add_argument("--N", type=int, default=8); ap.add_argument("--steps", type=int, default=25); a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    P = L(f"roll_runs/{a.tag}/test_pred.npz"); T = int(P["T"]); Kf = int(P["K"])
    ds = TrajFullDS("test", modality=str(P["mod"]), kind="short")
    model = DiTRollout(H=Kf).to(dev); model.load_state_dict(torch.load(f"roll_runs/{a.tag}/best.pt", map_location=dev))
    acp = cosine_acp(T, device=dev)
    G, G1, NOM, ES, NMES, spread, y, obj = sample_metrics(model, ds, dev, acp, a.N, a.steps)
    fail = (y >= 2).astype(float)
    np.savez(f"roll_runs/{a.tag}/metrics.npz", G1=G1, Gbest=G, NOM=NOM, ES=ES, NMES=NMES, spread=spread, y5=y, obj=obj, K=Kf)
    hs = [h for h in [3, 5, 8, 10] if h <= Kf]                            # horizon steps (1-indexed)
    print(f"=== SHORT-WINDOW ROLLOUT WM ({a.tag}, n={len(y)}, K={Kf}, N={a.N} samples) ===\n")
    print("-- (1) reconstruction, median drift° AT each horizon: single-shot(N1) & best-of-N vs no-motion(stays-seated) --")
    print(f"  {'':22s}" + "".join(f"{'k'+str(h):>10s}" for h in hs))
    print(f"  {'MODEL single-shot N1':22s}" + "".join(f"{np.median(G1[:,h-1]):10.2f}" for h in hs))
    print(f"  {'MODEL best-of-N':22s}" + "".join(f"{np.median(G[:,h-1]):10.2f}" for h in hs))
    print(f"  {'no-motion baseline':22s}" + "".join(f"{np.median(NOM[:,h-1]):10.2f}" for h in hs))
    print("  -- single-shot by outcome (N1 | no-motion) at each horizon --")
    for mi, m in enumerate(MODES):
        g = y == mi
        if g.sum(): print(f"  {m:22s}" + "".join(f"{np.median(G1[g,h-1]):5.1f}|{np.median(NOM[g,h-1]):5.1f}" for h in hs))
    print("\n-- (1b) energy score (proper scoring rule, z-space, lower=better): model distribution vs no-motion point --")
    print(f"  {'outcome':22s} {'model ES':>9s} {'no-motion':>9s}")
    for mi, m in enumerate(MODES):
        g = y == mi
        if g.sum(): print(f"  {m:22s} {np.median(ES[g]):9.2f} {np.median(NMES[g]):9.2f}")
    print(f"  {'ALL':22s} {np.median(ES):9.2f} {np.median(NMES):9.2f}")

    print("\n-- (2) boolean read-off calibration vs separate heads --")
    rows = [("rollout read-off", P["p2"][:, 1])]
    for name, path in [("separate classifier", "cls_runs/full/test_pred.npz"), ("joint MDN+bool", "traj_runs/joint_full/test_pred.npz")]:
        try:
            D = L(path)
            if np.array_equal(D["y5"], y): rows.append((name, D["p2"][:, 1]))
            else: print(f"  [warn] {name} test order mismatch — skipping")
        except Exception: pass
    print(f"  {'model':22s} {'Brier':>7s} {'ECE':>6s} {'AUROC':>6s}")
    for name, p in rows: print(f"  {name:22s} {brier(p,fail):7.3f} {ece(p,fail):6.3f} {auroc(p,fail):6.3f}")

    print("\n-- (3) sample diversity vs P(fail) (generative WM captures aleatoric unc) --")
    pf = P["p2"][:, 1]
    print(f"  corr(P(fail), end-z sample-spread) = {float(np.corrcoef(pf, spread)[0,1]):+.3f}")
    print(f"  end-z spread mm: low-risk half {np.median(spread[pf<np.median(pf)]):.1f}  vs high-risk half {np.median(spread[pf>=np.median(pf)]):.1f}")


if __name__ == "__main__":
    main()
