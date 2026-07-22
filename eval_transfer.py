"""Cross-object latent transfer eval (object-disjoint). Compares 4 arms — no_latent / abstract_oracle /
grounded_oracle / shuffled_oracle — on HELD-OUT objects. Validated iff grounded-oracle beats abstract-oracle AND
shuffled-oracle (i.e. grounding the CoM in geometry is what lets the latent transfer). Paired object-bootstrap CIs.

    python eval_transfer.py
"""
import os, sys, json
import numpy as np, torch
from torch.utils.data import DataLoader
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "gateb")); sys.path.insert(0, os.path.join(HERE, "density", "v2"))
from my_dataset_gateb import GateBDS
from wm.drop_diffusion import GateBDiT, ddim_sample, cosine_acp
from wm.drop_net import sixd_to_R
import drop_sweep as DS
from generate_obj import stable_orientations, basin_of
from scipy.spatial.transform import Rotation as R

N = 40
ARMS = {"no_latent": ("none", False, 13), "abstract_oracle": ("abstract", True, 13),
        "grounded_oracle": ("grounded", False, 17), "shuffled_oracle": ("shuffled", False, 17)}


def stables(o):
    Rs, _ = stable_orientations(DS.hull(o)); return Rs


@torch.no_grad()
def eval_arm(lm, use_lat, nf, ckpt, dev):
    os.environ["CDWM_GATEB_LATENT"] = lm                      # set BEFORE GateBDS builds its features
    te = GateBDS("test"); st = {k: torch.tensor(v, device=dev) for k, v in te.stats.items()}
    model = GateBDiT(n_feat=nf, use_latent=use_lat).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev)); model.eval()
    acp = cosine_acp(1000, device=dev); Rc = {}; nll, brier, OBJ = [], [], []
    for b in DataLoader(te, 128, num_workers=6):
        pts, br, cl, tb = [b[k].to(dev) for k in ("pts", "base_rel", "closing", "table")]
        B = pts.shape[0]; Rrel = sixd_to_R(br[:, 1:7]).cpu().numpy()
        cond = model.cond(pts, br, cl, tb, b["latent"].to(dev) if use_lat else None)
        x0 = ddim_sample(model, cond.repeat_interleave(N, 0), 1, acp, steps=25, device=dev).squeeze(1)
        dR = sixd_to_R((x0 * st["std"] + st["mean"])[:, :6]).cpu().numpy().reshape(B, N, 3, 3)
        objs = b["object"]; tb_ = b["basin"].numpy()
        for i in range(B):
            o = objs[i]
            if o not in Rc: Rc[o] = stables(o)
            Rs = Rc[o]; K = len(Rs); bins = np.zeros(K)
            for s in range(N):
                q = R.from_matrix(dR[i, s] @ Rrel[i]).as_quat(); bins[basin_of(np.r_[q[3], q[:3]], Rs)] += 1
            p = (bins + 0.5) / (bins.sum() + 0.5 * K); t = min(int(tb_[i]), K - 1); oh = np.zeros(K); oh[t] = 1
            nll.append(-np.log(p[t])); brier.append(((p - oh) ** 2).sum()); OBJ.append(o)
    return np.array(nll), np.array(brier), np.array(OBJ)


def oboot(delta, obj, rng):
    uo = np.unique(obj); idx = {o: np.where(obj == o)[0] for o in uo}
    bs = [np.mean(delta[np.concatenate([idx[o] for o in rng.choice(uo, len(uo), True)])]) for _ in range(3000)]
    return float(np.mean(delta)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"; rng = np.random.default_rng(0); res = {}
    for a, (lm, ul, nf) in ARMS.items():
        ck = f"gateb_runs/{a}/best.pt"                        # object mode -> no prefix
        if not os.path.exists(ck): print(f"{a}: no ckpt, skip"); continue
        nll, brier, obj = eval_arm(lm, ul, nf, ck, dev); res[a] = (nll, brier, obj)
        print(f"  {a:16s} NLL {nll.mean():.3f}  Brier {brier.mean():.3f}  (n={len(nll)})")
    if "no_latent" not in res: return
    obj = res["no_latent"][2]
    print("\ntransfer gain over no-latent (paired object-bootstrap; >0 & CI excludes 0 = the latent HELPS on unseen objects):")
    for a in ["abstract_oracle", "grounded_oracle", "shuffled_oracle"]:
        if a in res:
            pt, lo, hi = oboot(res["no_latent"][0] - res[a][0], obj, rng)
            print(f"  {a:16s} {pt:+.3f} [{lo:+.3f},{hi:+.3f}] {'SIG' if lo > 0 else 'ns'}")
    print("\n★ key comparisons (grounding vs abstract vs shuffled):")
    for a, b in [("grounded_oracle", "abstract_oracle"), ("grounded_oracle", "shuffled_oracle")]:
        if a in res and b in res:
            pt, lo, hi = oboot(res[b][0] - res[a][0], obj, rng)
            print(f"  {a} < {b}: {pt:+.3f} [{lo:+.3f},{hi:+.3f}] {'SIG' if lo > 0 else 'ns'}")
    json.dump({a: {"nll": float(res[a][0].mean()), "brier": float(res[a][1].mean())} for a in res},
              open("gateb_runs/eval_transfer.json", "w"), indent=1)


if __name__ == "__main__":
    main()
