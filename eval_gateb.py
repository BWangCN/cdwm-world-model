"""Gate B payoff eval — does the distribution beat the point predictor, and does oracle-latent beat no-latent?

For each held-out-object test episode (fixed visible observation: cloud + release, CoM hidden), sample N resting
poses from the diffusion arm, compose with the release orientation (R_rest = dR @ R_rel), assign each to the
object's nearest stable pose (basin) -> predicted P(basin). Score the TRUE basin: NLL, top-1, coverage, entropy.
Point arm = one pose -> a Laplace-smoothed one-hot P(basin). Criteria: diff NLL < point NLL (distribution needed);
diff_oracle NLL < diff NLL + lower entropy (hidden CoM is causal/learnable).

    python eval_gateb.py            # evaluates point, diff, diff_oracle -> gateb_runs/eval.json
"""
import os, sys, json
import numpy as np, torch
from torch.utils.data import DataLoader
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "gateb")); sys.path.insert(0, os.path.join(HERE, "density", "v2"))
from my_dataset_gateb import GateBDS
from wm.drop_diffusion import GateBDiT, GateBPoint, ddim_sample, cosine_acp
from wm.drop_net import sixd_to_R
import drop_sweep as DS
from generate_obj import stable_orientations, basin_of
from scipy.spatial.transform import Rotation as R

N_SAMP = 60


def stables(o):
    Rs, _ = stable_orientations(DS.hull(o)); return Rs


def boundaryness(te, k=40):
    """data-derived per-episode boundaryness = basin entropy among the k nearest releases (same object). High =
    a release neighborhood where multiple basins occur under different hidden CoM = where the distribution should win."""
    score = np.zeros(len(te.obj))
    for o in set(te.obj):
        idx = np.where(te.obj == o)[0]
        Rr = te.R_rel[idx]                                    # (m,3,3)
        tr = np.einsum("aij,bij->ab", Rr, Rr)                 # tr(Ri Rj^T) (releases share frame -> Ri.Rj elementwise)
        ang = np.arccos(np.clip((tr - 1) / 2, -1, 1))         # geodesic angle between releases
        bas = te.basin[idx]
        for a in range(len(idx)):
            nn = np.argsort(ang[a])[:k]; b = bas[nn]
            _, c = np.unique(b, return_counts=True); p = c / c.sum()
            score[idx[a]] = -(p * np.log(p)).sum()            # local basin entropy
    return score


@torch.no_grad()
def eval_arm(arm, dev, te, st):
    model = (GateBPoint() if arm == "point" else GateBDiT(use_latent=(arm == "diff_oracle"))).to(dev)
    mode = os.environ.get("CDWM_GATEB_SPLIT", "object"); pre = "" if mode == "object" else f"{mode}_"
    model.load_state_dict(torch.load(f"gateb_runs/{pre}{arm}/best.pt", map_location=dev)); model.eval()
    acp = cosine_acp(1000, device=dev); Rc = {}
    nll, top1, cover, ent, brier = [], [], [], [], []
    maxb = int(os.environ.get("GATEB_MAXB", "0"))
    for bi, b in enumerate(DataLoader(te, 128, num_workers=6)):
        if maxb and bi >= maxb: break
        pts, br, cl, tb = [b[k].to(dev) for k in ("pts", "base_rel", "closing", "table")]
        B = pts.shape[0]
        Rrel = sixd_to_R(br[:, 1:7]).cpu().numpy()            # release orientation, from base_rel
        objs = b["object"]; truebasin = b["basin"].numpy()
        if arm == "point":
            pred = model.predict(pts, br, cl, tb)
            dR = sixd_to_R((pred * st["std"] + st["mean"])[:, :6]).cpu().numpy()[:, None]        # (B,1,3,3)
        else:
            cond = model.cond(pts, br, cl, tb, b["latent"].to(dev) if arm == "diff_oracle" else None)
            x0 = ddim_sample(model, cond.repeat_interleave(N_SAMP, 0), 1, acp, steps=25, device=dev).squeeze(1)
            dR = sixd_to_R((x0 * st["std"] + st["mean"])[:, :6]).cpu().numpy().reshape(B, N_SAMP, 3, 3)
        for i in range(B):
            o = objs[i]
            if o not in Rc: Rc[o] = stables(o)
            Rs = Rc[o]; K = len(Rs); bins = np.zeros(K)
            for s in range(dR.shape[1]):
                Rrest = dR[i, s] @ Rrel[i]                    # dR @ R_rel = absolute resting orientation
                q = R.from_matrix(Rrest).as_quat()            # xyzw
                bins[basin_of(np.r_[q[3], q[:3]], Rs)] += 1
            p = (bins + 0.5) / (bins.sum() + 0.5 * K)          # Laplace-smoothed P(basin)
            t = int(truebasin[i]); oh = np.zeros(K); oh[t] = 1
            nll.append(-np.log(p[t])); top1.append(int(p.argmax() == t))
            cover.append(int(bins[t] > 0)); ent.append(-(p * np.log(p)).sum()); brier.append(((p - oh) ** 2).sum())
    return {k: np.array(v) for k, v in dict(nll=nll, top1=top1, coverage=cover, entropy=ent, brier=brier).items()}


def freq_baseline(te):
    """strong baseline (Codex): per-object TRAIN marginal P(basin). Diffusion must beat this, not just the point regressor."""
    tr = GateBDS("train"); Rc = {}; cnt = {}
    for o in set(tr.obj):
        if o not in Rc: Rc[o] = len(stables(o))
        K = Rc[o]; c = np.zeros(K)
        for bb in tr.basin[tr.obj == o]:
            if 0 <= bb < K: c[bb] += 1
        cnt[o] = (c + 0.5) / (c.sum() + 0.5 * K)
    nll, top1, cover, ent, brier = [], [], [], [], []
    for i in range(len(te.obj)):
        o = te.obj[i]; p = cnt.get(o);
        if p is None: continue
        K = len(p); t = min(int(te.basin[i]), K - 1); oh = np.zeros(K); oh[t] = 1
        nll.append(-np.log(p[t])); top1.append(int(p.argmax() == t)); cover.append(1)
        ent.append(-(p * np.log(p)).sum()); brier.append(((p - oh) ** 2).sum())
    return {k: np.array(v) for k, v in dict(nll=nll, top1=top1, coverage=cover, entropy=ent, brier=brier).items()}


def _agg(m, mask=None):
    s = slice(None) if mask is None else mask
    d = {k: float(np.mean(v[s])) for k, v in m.items()}
    d["n"] = int(len(m["nll"]) if mask is None else int(mask.sum()))
    return d


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    te = GateBDS("test"); st = {k: torch.tensor(v, device=dev) for k, v in te.stats.items()}
    bnd = boundaryness(te); hi = bnd >= np.quantile(bnd, 0.75)     # boundary subset = top-quartile local basin entropy
    print(f"test {len(te)} eps, {len(set(te.obj))} held-out objs; near-boundary subset (top-quartile) n={int(hi.sum())}")
    res = {}
    for arm in ["point", "diff", "diff_oracle"]:
        if not os.path.exists(f"gateb_runs/{arm}/best.pt"): print(f"{arm}: no ckpt, skip"); continue
        m = eval_arm(arm, dev, te, st); res[arm] = m
        ov, bd = _agg(m), _agg(m, hi)
        print(f"  {arm:12s} ALL   nll {ov['nll']:.3f} brier {ov['brier']:.3f} top1 {ov['top1']:.2f} cov {ov['coverage']:.2f} ent {ov['entropy']:.2f}")
        print(f"  {'':12s} BNDRY nll {bd['nll']:.3f} brier {bd['brier']:.3f} top1 {bd['top1']:.2f} cov {bd['coverage']:.2f} ent {bd['entropy']:.2f}")
    mode0 = os.environ.get("CDWM_GATEB_SPLIT", "object")
    if mode0 != "object" and "diff" in res:                   # basin-frequency baseline (shared objects only)
        res["freq"] = freq_baseline(te)
        fa = _agg(res["freq"]); print(f"  {'freq-base':12s} ALL   nll {fa['nll']:.3f} brier {fa['brier']:.3f} top1 {fa['top1']:.2f}")
    for lab, mask in [("ALL", None), ("BOUNDARY", hi)]:
        if "point" in res and "diff" in res:
            p, d = _agg(res["point"], mask), _agg(res["diff"], mask)
            print(f"\n[{lab}] C3 dist>point: NLL {d['nll'] < p['nll']} ({d['nll']:.3f}v{p['nll']:.3f})  "
                  f"Brier {d['brier'] < p['brier']} ({d['brier']:.3f}v{p['brier']:.3f})")
            if "freq" in res:
                f = _agg(res["freq"], mask)
                print(f"[{lab}] C3b dist>freq-baseline: NLL {d['nll'] < f['nll']} ({d['nll']:.3f}v{f['nll']:.3f})  "
                      f"Brier {d['brier'] < f['brier']} ({d['brier']:.3f}v{f['brier']:.3f})")
            if "diff_oracle" in res:
                o = _agg(res["diff_oracle"], mask)
                print(f"[{lab}] C2 oracle>no-latent: NLL {o['nll'] < d['nll']} ({o['nll']:.3f}v{d['nll']:.3f})  "
                      f"Brier {o['brier'] < d['brier']} ({o['brier']:.3f}v{d['brier']:.3f})")
    mode = os.environ.get("CDWM_GATEB_SPLIT", "object")
    json.dump({a: {"all": _agg(m), "boundary": _agg(m, hi)} for a, m in res.items()},
              open(f"gateb_runs/eval_{mode}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
