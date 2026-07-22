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


@torch.no_grad()
def eval_arm(arm, dev, te, st):
    model = (GateBPoint() if arm == "point" else GateBDiT(use_latent=(arm == "diff_oracle"))).to(dev)
    model.load_state_dict(torch.load(f"gateb_runs/{arm}/best.pt", map_location=dev)); model.eval()
    acp = cosine_acp(1000, device=dev); Rc = {}
    nll, top1, cover, ent = [], [], [], []
    for b in DataLoader(te, 128, num_workers=6):
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
            t = int(truebasin[i])
            nll.append(-np.log(p[t])); top1.append(int(p.argmax() == t))
            cover.append(int(bins[t] > 0)); ent.append(-(p * np.log(p)).sum())
    return dict(arm=arm, n=len(nll), nll=float(np.mean(nll)), top1=float(np.mean(top1)),
                coverage=float(np.mean(cover)), entropy=float(np.mean(ent)))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    te = GateBDS("test"); st = {k: torch.tensor(v, device=dev) for k, v in te.stats.items()}
    print(f"test: {len(te)} episodes, {len(set(te.obj))} held-out objects")
    rows = []
    for arm in ["point", "diff", "diff_oracle"]:
        if not os.path.exists(f"gateb_runs/{arm}/best.pt"): print(f"{arm}: no ckpt, skip"); continue
        r = eval_arm(arm, dev, te, st); rows.append(r)
        print(f"  {r['arm']:12s} NLL {r['nll']:.3f}  top1 {r['top1']:.2f}  coverage {r['coverage']:.2f}  entropy {r['entropy']:.2f}")
    json.dump(rows, open("gateb_runs/eval.json", "w"), indent=1)
    d = {r["arm"]: r for r in rows}
    if "point" in d and "diff" in d:
        print(f"\ncriterion 3 (distribution > point on NLL): {d['diff']['nll'] < d['point']['nll']}  "
              f"({d['diff']['nll']:.3f} vs {d['point']['nll']:.3f})")
    if "diff" in d and "diff_oracle" in d:
        print(f"criterion 2 (oracle > no-latent): NLL {d['diff_oracle']['nll']:.3f} < {d['diff']['nll']:.3f} = "
              f"{d['diff_oracle']['nll'] < d['diff']['nll']}; entropy {d['diff_oracle']['entropy']:.2f} < "
              f"{d['diff']['entropy']:.2f} (oracle sharper) = {d['diff_oracle']['entropy'] < d['diff']['entropy']}")


if __name__ == "__main__":
    main()
