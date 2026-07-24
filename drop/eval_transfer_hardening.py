"""Transfer-result hardening (Codex parity pass): (#1) per-object breakdown of the grounded transfer gain across the
19 held-out objects — rule out few-object dominance; (#2) calibration/ECE on the transfer split (same `ece` as
eval_gateb.py, so comparable to the Gate B ECE 0.015). Reuses the 4 trained transfer checkpoints; seeded noise
(deterministic). Object-disjoint split.

    python eval_transfer_hardening.py            # full GPU eval + analysis
    python eval_transfer_hardening.py --smoke    # analysis-only self-check on synthetic data (no GPU)
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))

N = 40; SEED = 1234
ARMS = {"no_latent": ("none", False, 13), "abstract_oracle": ("abstract", True, 13),
        "grounded_oracle": ("grounded", False, 17), "shuffled_oracle": ("shuffled", False, 17)}


def ece(pt, ok, nb=10):                                        # identical to eval_gateb.py's ece
    e = 0.0
    for i in range(nb):
        m = (pt >= i / nb) & (pt < (i + 1) / nb)
        if m.sum(): e += m.mean() * abs(pt[m].mean() - ok[m].mean())
    return float(e)


def per_object_gain(res, base, obj):                          # mean NLL gain (base - grounded) per object; >0 = grounded better
    g = res["grounded_oracle"]["nll"]; b = res[base]["nll"]; uo = np.unique(obj)
    return uo, np.array([b[obj == o].mean() - g[obj == o].mean() for o in uo])


def analyze(res):
    print(f"{'arm':16s} {'NLL':>6s} {'Brier':>6s} {'ECE':>6s} {'conf':>6s} {'acc':>6s}")
    for a in res:
        r = res[a]; print(f"{a:16s} {r['nll'].mean():6.3f} {r['brier'].mean():6.3f} "
                          f"{ece(r['ptop'], r['top1']):6.3f} {r['ptop'].mean():6.3f} {r['top1'].mean():6.3f}")
    obj = res["grounded_oracle"]["obj"]
    print(f"\n[PER-OBJECT] grounded transfer gain across held-out objects (base_NLL - grounded_NLL; >0 = grounded better):")
    for base in ["no_latent", "abstract_oracle"]:
        if base not in res: continue
        uo, gains = per_object_gain(res, base, obj)
        npos = int((gains > 0).sum())
        print(f"  vs {base:16s}: {npos}/{len(uo)} objects positive | median {np.median(gains):+.3f} | "
              f"min {gains.min():+.3f} max {gains.max():+.3f}")
        print("     sorted per-object gains: " + " ".join(f"{x:+.2f}" for x in np.sort(gains)))
    print("\n[CALIBRATION] ECE (lower better) matches eval_gateb method; compare grounded to the Gate B ECE 0.015.")


def gpu_eval():
    import torch
    from torch.utils.data import DataLoader
    from drop.my_dataset_gateb import GateBDS
    from drop.drop_diffusion import GateBDiT, cosine_acp
    from common.dit import ddim_sample
    from drop.drop_net import sixd_to_R
    from drop.density.v2 import drop_sweep as DS
    from drop.gateb.generate_obj import stable_orientations, basin_of
    from scipy.spatial.transform import Rotation as R
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    @torch.no_grad()
    def eval_arm(lm, use_lat, nf, ckpt):
        os.environ["CDWM_GATEB_LATENT"] = lm
        te = GateBDS("test"); st = {k: torch.tensor(v, device=dev) for k, v in te.stats.items()}
        model = GateBDiT(n_feat=nf, use_latent=use_lat).to(dev)
        model.load_state_dict(torch.load(ckpt, map_location=dev)); model.eval()
        acp = cosine_acp(1000, device=dev); g = torch.Generator().manual_seed(SEED); Rc = {}
        nll, brier, ptop, top1, OBJ = [], [], [], [], []
        for b in DataLoader(te, 128, num_workers=6):
            pts, br, cl, tb = [b[k].to(dev) for k in ("pts", "base_rel", "closing", "table")]
            B = pts.shape[0]; Rrel = sixd_to_R(br[:, 1:7]).cpu().numpy()
            cond = model.cond(pts, br, cl, tb, b["latent"].to(dev) if use_lat else None)
            xT = torch.randn(B * N, 1, 9, generator=g)
            x0 = ddim_sample(model, cond.repeat_interleave(N, 0), 1, acp, steps=25, device=dev, x_T=xT).squeeze(1)
            dR = sixd_to_R((x0 * st["std"] + st["mean"])[:, :6]).cpu().numpy().reshape(B, N, 3, 3)
            objs = b["object"]; tb_ = b["basin"].numpy()
            for i in range(B):
                o = objs[i]
                if o not in Rc: Rc[o] = stable_orientations(DS.hull(o))[0]
                Rs = Rc[o]; K = len(Rs); bins = np.zeros(K)
                for s in range(N):
                    q = R.from_matrix(dR[i, s] @ Rrel[i]).as_quat(); bins[basin_of(np.r_[q[3], q[:3]], Rs)] += 1
                p = (bins + 0.5) / (bins.sum() + 0.5 * K); t = min(int(tb_[i]), K - 1); oh = np.zeros(K); oh[t] = 1
                nll.append(-np.log(p[t])); brier.append(((p - oh) ** 2).sum())
                ptop.append(float(p.max())); top1.append(int(p.argmax() == t)); OBJ.append(o)
        return dict(nll=np.array(nll), brier=np.array(brier), ptop=np.array(ptop),
                    top1=np.array(top1), obj=np.array(OBJ))

    res = {}
    for a, (lm, ul, nf) in ARMS.items():
        ck = f"gateb_runs/{a}/best.pt"
        if not os.path.exists(ck): print(f"{a}: no ckpt, skip"); continue
        res[a] = eval_arm(lm, ul, nf, ck); print(f"  done {a} (n={len(res[a]['nll'])})")
    np.savez("gateb_runs/transfer_hardening.npz", **{f"{a}__{k}": res[a][k] for a in res for k in res[a]})
    return res


def main():
    if "--smoke" in sys.argv:                                 # analysis self-check on synthetic data, no GPU
        rng = np.random.default_rng(0); objs = np.repeat([f"o{i}" for i in range(19)], 100)
        def fake(mu): return dict(nll=rng.normal(mu, 0.3, len(objs)).clip(1e-3), brier=rng.random(len(objs)),
                                  ptop=rng.random(len(objs)), top1=rng.integers(0, 2, len(objs)), obj=objs)
        res = {"no_latent": fake(0.68), "abstract_oracle": fake(0.68),
               "grounded_oracle": fake(0.50), "shuffled_oracle": fake(0.65)}
        analyze(res)
        uo, gains = per_object_gain(res, "no_latent", objs)
        assert len(uo) == 19 and gains.shape == (19,), "per-object shape"
        assert 0.0 <= ece(res["grounded_oracle"]["ptop"], res["grounded_oracle"]["top1"]) <= 1.0, "ece range"
        print("\nsmoke OK: per-object=19, ece in [0,1], grounded gain broad on synthetic")
        return
    res = gpu_eval(); analyze(res)
    json.dump({a: {"nll": float(res[a]["nll"].mean()), "brier": float(res[a]["brier"].mean()),
                   "ece": ece(res[a]["ptop"], res[a]["top1"])} for a in res},
              open("gateb_runs/transfer_hardening.json", "w"), indent=1)


if __name__ == "__main__":
    main()
