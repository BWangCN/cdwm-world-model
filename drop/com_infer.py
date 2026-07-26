"""Item 2 — CoM estimation from observations (drop the oracle). ANALYTIC Bayesian posterior over the hidden CoM
using the FROZEN grounded Gate B WM as the likelihood P(basin | CoM, release). No training: the WM already learned
P(basin | CoM, release); Bayesian updating over k observed drops of the SAME object instance recovers the CoM and
sharpens the next-drop prediction, approaching the oracle. (Amortized DeepSets encoder = a faster follow-up.)

Instances (no regen): existing hull Gate B episodes GROUP by (object, CoM-key = axis, round(delta,3mm)) into
fixed-CoM instances (~100 drops each at varied releases). Test = the 19 object-disjoint held-out objects.

Readout: predictive NLL of held-out query drops vs k (# observed drops), bounded by no_latent (k=0 prior) and
grounded_oracle (true CoM). Codex: posterior over CoM (not a point); check entropy drops with k; object-bootstrap.

    CDWM_GATEB_LATENT=grounded python com_infer.py            # (env set for the grounded test dataset)
"""
import os, json, argparse, collections
import numpy as np, torch
from drop.my_dataset_gateb import GateBDS, _load_all, object_split, build_feats_drop
from drop.drop_diffusion import GateBDiT
from common.dit import ddim_sample, cosine_acp
from drop.drop_net import sixd_to_R
from common.utils import R_to_6d
from drop.gateb.generate_obj import stable_orientations, basin_of
from drop.density.v2 import drop_sweep as DS
from common.paths import PCDIR
from scipy.spatial.transform import Rotation as R

N_SAMP = 40                                                    # WM basin samples per (CoM,release) likelihood eval
KS = [0, 1, 2, 4, 8]                                           # adaptation curve
N_QUERY = 4                                                    # held-out query drops per instance
MAX_INST = 8                                                   # instances per object (compute budget)
NDELTA = 11                                                    # CoM-grid resolution along each axis


def com_key(a, d):
    return (int(a), round(float(d) / 0.003) * 0.003)


@torch.no_grad()
def wm_pbasin(model, ds, o, releases, drop_hs, coms, dev, st, use_latent_feat):
    """P(basin | CoM, release) from the frozen WM for every (release, CoM) pair. releases:(R,3,3) mats,
    drop_hs:(R,) heights, coms:(C,3) object-frame CoM offsets (or None for no_latent). Returns bins (R,C,K)."""
    rc = ds._cl(o); mu_full = rc["mu"]; Sig, op, ic = rc["Sig_obj"], rc["opacity"], rc["ic"]
    idx = np.random.default_rng(0).choice(len(mu_full), ds.n, replace=len(mu_full) < ds.n)
    mu, Sig, op, ic = mu_full[idx], Sig[idx], op[idx], ic[idx]
    Rs = _stables(o); K = len(Rs)
    C = 1 if coms is None else len(coms)
    bins = np.zeros((len(releases), C, K))
    for ri, Rf in enumerate(releases):
        feat, w = build_feats_drop(mu, Sig, op, ic, Rf.astype(np.float32), "release")
        feat = (feat - ds.fmean) / ds.fstd
        c0 = mu.mean(0); mu_r = (mu - c0) @ Rf.T
        base_rel = np.concatenate([[drop_hs[ri]], R_to_6d(Rf[None])[0], [0, 0]]).astype(np.float32)   # actual drop_h + 6D release
        pts_list = []
        for ci in range(C):
            parts = [feat]
            if use_latent_feat:
                com_r = (coms[ci] - c0) @ Rf.T
                vec = com_r[None, :] - mu_r; dist = np.linalg.norm(vec, axis=1, keepdims=True)
                parts.append(np.concatenate([vec, dist], 1).astype(np.float32))
            pts_list.append(np.concatenate(parts + [w[:, None]], 1).astype(np.float32))
        pts = torch.from_numpy(np.stack(pts_list)).to(dev)                         # (C,N,F)
        br = torch.from_numpy(base_rel)[None].repeat(C, 1).to(dev)
        cl = torch.zeros(C, 1, device=dev); tb = torch.zeros(C, 1, device=dev)
        cond = model.cond(pts, br, cl, tb, None)
        x0 = ddim_sample(model, cond.repeat_interleave(N_SAMP, 0), 1, acp_g, steps=25, device=dev).squeeze(1)
        dR = sixd_to_R((x0 * st["std"] + st["mean"])[:, :6]).cpu().numpy().reshape(C, N_SAMP, 3, 3)
        for ci in range(C):
            for s in range(N_SAMP):
                q = R.from_matrix(dR[ci, s] @ Rf).as_quat()
                bins[ri, ci, basin_of(np.r_[q[3], q[:3]], Rs)] += 1
    return bins, K


_MS = {}
def _stables(o):
    if o not in _MS: _MS[o], _ = stable_orientations(DS.hull(o))
    return _MS[o]


def com_grid(o, ds):
    hc, axes = ds._hull(o)
    from scipy.spatial import ConvexHull
    mu = np.load(f"{PCDIR}/{o}.npz")["mu"].astype(np.float64)
    if len(mu) > 4000: mu = mu[np.random.default_rng(0).choice(len(mu), 4000, replace=False)]
    V = mu[ConvexHull(mu).vertices]; half = np.abs((V - hc) @ axes.T).max(0)
    coms, keys = [], []
    for a in range(3):
        for d in np.linspace(-0.35, 0.35, NDELTA) * half[a]:
            coms.append(hc + d * axes[a]); keys.append(com_key(a, d))
    return np.array(coms, np.float32), keys, (hc, axes, half)


def main():
    global acp_g, N_SAMP, KS, N_QUERY, MAX_INST, NDELTA
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    if a.smoke:
        N_SAMP, KS, N_QUERY, MAX_INST, NDELTA = 24, [0, 1, 2, 4], 3, 4, 9
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    acp_g = cosine_acp(1000, device=dev)
    os.environ["CDWM_GATEB_LATENT"] = "grounded"
    ds = GateBDS("test")                                       # hull src; grounded features; test = 19 held-out objects
    st = {k: torch.tensor(v, device=dev) for k, v in ds.stats.items()}
    gm = GateBDiT(n_feat=17, use_latent=False).to(dev)
    gm.load_state_dict(torch.load("gateb_runs/grounded_oracle/best.pt", map_location=dev)); gm.eval()
    nm = GateBDiT(n_feat=13, use_latent=False).to(dev)
    nm.load_state_dict(torch.load("gateb_runs/no_latent/best.pt", map_location=dev)); nm.eval()

    OBJ, RELQ, DH, COM, RESTQ, BAS, _ = _load_all("hull")
    sp = object_split(OBJ, seed=0); test_objs = [o for o in sorted(set(OBJ)) if sp[o] == "test"]
    if a.smoke: test_objs = test_objs[:5]
    rng = np.random.default_rng(0)
    curve = {k: [] for k in KS}; oracle = []; nolat = []; per_obj = collections.defaultdict(lambda: {k: [] for k in KS})
    ent_by_k = {k: [] for k in KS}
    map_curve = {k: [] for k in KS}; postmass = {k: [] for k in KS}          # MAP plug-in + posterior mass on true CoM
    ece = {a: [] for a in ("no_latent", "k0", "kmax", "map_kmax", "oracle")}  # (confidence, correct) per arm
    n_excl = [0, 0]                                                          # [excluded query drops, total]
    kmax = max(KS)

    def geodeg(Ra, Rb):
        return float(np.degrees(np.arccos(np.clip((np.trace(Ra @ Rb.T) - 1) / 2, -1, 1))))

    for o in test_objs:
        idx = np.where(OBJ == o)[0]
        relmat = R.from_quat(RELQ[idx][:, [1, 2, 3, 0]]).as_matrix()          # (m,3,3)
        groups = collections.defaultdict(list)
        for j, (a, d) in enumerate(COM[idx]): groups[com_key(a, d)].append(j)
        insts = [g for g in groups.values() if len(g) >= max(KS) + N_QUERY]
        rng.shuffle(insts); insts = insts[:MAX_INST]
        if not insts: continue
        coms, keys, (hc, axes, half) = com_grid(o, ds)
        base_C = len(coms)                                                     # posterior grid = [:base_C]
        # append each instance's EXACT stored CoM (what the WM trained on) so the oracle is exact, not grid-snapped
        for g in insts:
            a0, d0 = COM[idx][list(g)[0]]; coms = np.concatenate([coms, (hc + float(d0) * axes[int(a0)])[None]], 0)
        coms = coms.astype(np.float32)
        need = sorted({j for g in insts for j in list(g)[:max(KS) + N_QUERY]})
        pos = {j: i for i, j in enumerate(need)}; dh_need = DH[idx][need]
        binsG, K = wm_pbasin(gm, ds, o, relmat[need], dh_need, coms, dev, st, True)   # (R,C,K) grounded likelihood
        binsN, _ = wm_pbasin(nm, ds, o, relmat[need], dh_need, None, dev, st, False)  # (R,1,K) no-latent marginal
        pG = (binsG + 0.5) / (binsG.sum(-1, keepdims=True) + 0.5 * K)           # P(basin|c,rel)  (R,C,K)
        pN = (binsN + 0.5) / (binsN.sum(-1, keepdims=True) + 0.5 * K)           # (R,1,K)
        truebasin = BAS[idx]
        for gi, g in enumerate(insts):
            g = list(g); support = g[:max(KS)]; query0 = g[max(KS):max(KS) + N_QUERY]
            ci_true = base_C + gi                                              # exact stored CoM index
            # release-duplicate guard: drop query drops within 5 deg of ANY support release (Codex)
            query = []
            for jq in query0:
                n_excl[1] += 1
                if min(geodeg(relmat[jq], relmat[js]) for js in support) < 5.0: n_excl[0] += 1; continue
                query.append(jq)
            if not query: continue
            ci_near = int(np.argmin(np.linalg.norm(coms[:base_C] - coms[ci_true], axis=1)))   # true CoM's grid cell
            for k in KS:
                lp = np.zeros(base_C)                                          # uniform prior over the CLEAN grid only
                for j in support[:k]:
                    lp = lp + np.log(pG[pos[j], :base_C, int(truebasin[j])] + 1e-9)
                w = np.exp(lp - lp.max()); w /= w.sum()                         # posterior over CoM grid
                if k > 0: ent_by_k[k].append(float(-(w * np.log(w + 1e-12)).sum()))
                postmass[k].append(float(w[ci_near])); cmap = int(w.argmax())   # posterior mass on true CoM + MAP cell
                for jq in query:
                    pq = (w[:, None] * pG[pos[jq], :base_C]).sum(0); pq /= pq.sum()   # marginal predictive
                    pm = pG[pos[jq], cmap] / pG[pos[jq], cmap].sum()                  # MAP plug-in predictive
                    t = int(truebasin[jq])
                    curve[k].append(-np.log(pq[t] + 1e-9)); per_obj[o][k].append(curve[k][-1])
                    map_curve[k].append(-np.log(pm[t] + 1e-9))
                    if k == 0: ece["k0"].append((float(pq.max()), int(pq.argmax() == t)))
                    if k == kmax:
                        ece["kmax"].append((float(pq.max()), int(pq.argmax() == t)))
                        ece["map_kmax"].append((float(pm.max()), int(pm.argmax() == t)))
            for jq in query:                                                    # oracle bound (exact true CoM) + no_latent
                t = int(truebasin[jq]); po = pG[pos[jq], ci_true] / pG[pos[jq], ci_true].sum(); pn = pN[pos[jq], 0]
                oracle.append(-np.log(po[t] + 1e-9)); ece["oracle"].append((float(po.max()), int(po.argmax() == t)))
                nolat.append(-np.log(pn[t] + 1e-9)); ece["no_latent"].append((float(pn.max()), int(pn.argmax() == t)))
        print(f"[{o[:28]:28s}] {len(insts)} instances", flush=True)

    def obj_boot(per, ks, nb=3000):
        objs = list(per.keys())
        base = {k: np.mean([np.mean(per[o][k]) for o in objs if per[o][k]]) for k in ks}
        r = np.random.default_rng(0); out = {}
        for k in ks:
            vals = np.array([np.mean(per[o][k]) for o in objs if per[o][k]])
            b = np.array([r.choice(vals, len(vals), replace=True).mean() for _ in range(nb)])
            out[k] = (float(base[k]), *np.percentile(b, [2.5, 97.5]).tolist())
        return out
    # PAIRED per-object adaptation gain k=0 -> k=max (the significance test; marginal per-k CIs are object-variance dominated)
    kmax = max(KS); objs_ok = [o for o in per_obj if per_obj[o][0] and per_obj[o][kmax]]
    pair = np.array([np.mean(per_obj[o][0]) - np.mean(per_obj[o][kmax]) for o in objs_ok])
    rb = np.random.default_rng(0); pboot = np.array([rb.choice(pair, len(pair), replace=True).mean() for _ in range(5000)])
    plo, phi = np.percentile(pboot, [2.5, 97.5])
    def ece_of(pairs, nb=10):
        if not pairs: return float("nan")
        c = np.array([p[0] for p in pairs]); ok = np.array([p[1] for p in pairs]); e = 0.0
        for i in range(nb):
            m = (c >= i / nb) & (c < (i + 1) / nb)
            if m.sum(): e += m.mean() * abs(c[m].mean() - ok[m].mean())
        return float(e)
    res = {"nll_vs_k": {k: float(np.mean(curve[k])) for k in KS},
           "map_nll_vs_k": {k: float(np.mean(map_curve[k])) for k in KS},          # MAP plug-in (vs marginal)
           "oracle_nll": float(np.mean(oracle)), "no_latent_nll": float(np.mean(nolat)),
           "post_entropy_vs_k": {k: float(np.mean(ent_by_k[k])) for k in KS if ent_by_k[k]},
           "post_mass_on_trueCoM_vs_k": {k: float(np.mean(postmass[k])) for k in KS},   # should rise with k
           "ece": {a: ece_of(v) for a, v in ece.items()},
           "query_excluded_dup_release": {"n": n_excl[0], "total": n_excl[1]},
           "obj_boot": {str(k): v for k, v in obj_boot(per_obj, KS).items()},
           "paired_gain_k0_to_kmax": {"mean": float(pair.mean()), "lo": float(plo), "hi": float(phi),
                                      "n_pos": int((pair > 0).sum()), "n_obj": len(objs_ok), "kmax": kmax},
           "per_obj_nll": {o: {int(k): float(np.mean(per_obj[o][k])) for k in KS if per_obj[o][k]} for o in objs_ok}}
    print(f"\n[PAIRED] adaptation gain k=0->k={kmax}: {pair.mean():+.3f} [{plo:+.3f},{phi:+.3f}] "
          f"({int((pair>0).sum())}/{len(objs_ok)} obj) SIG={plo>0}")
    print("\n=== CoM-from-observation adaptation curve (predictive NLL) ===")
    print(f"  no_latent arm (marginal, no obs)  NLL {res['no_latent_nll']:.3f}   ECE {res['ece']['no_latent']:.3f}")
    for k in KS: print(f"  k={k:2d}  marginal NLL {res['nll_vs_k'][k]:.3f}   MAP-plugin NLL {res['map_nll_vs_k'][k]:.3f}"
                       + ("   (uniform-CoM prior)" if k == 0 else ""))
    print(f"  oracle (true CoM, point)  NLL {res['oracle_nll']:.3f}   ECE {res['ece']['oracle']:.3f}")
    print(f"  ECE: k0 {res['ece']['k0']:.3f} | k{kmax} marginal {res['ece']['kmax']:.3f} | k{kmax} MAP {res['ece']['map_kmax']:.3f}")
    print("  posterior entropy vs k:", {k: round(v, 3) for k, v in res["post_entropy_vs_k"].items()})
    print("  posterior mass on true CoM vs k:", {k: round(v, 3) for k, v in res["post_mass_on_trueCoM_vs_k"].items()})
    print(f"  query drops excluded (release dup <5deg): {n_excl[0]}/{n_excl[1]}")
    json.dump(res, open("gateb_runs/com_infer.json", "w"), indent=1)
    print("\nwrote gateb_runs/com_infer.json")


if __name__ == "__main__":
    main()
