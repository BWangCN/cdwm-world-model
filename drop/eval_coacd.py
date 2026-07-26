"""Item 1 eval — Gate B on CoACD END-TO-END geometry (CoACD physics + real-mesh cloud). Three arms: point,
no_latent (distribution), grounded (hidden-CoM per-point latent). Object-disjoint test over the 29 mesh objects.
Basin frame = the REAL MESH stable poses (exactly as used at generation). Reports NLL / basin-acc / coverage /
ECE and object-bootstrap CIs for the two ABSOLUTE claims: dist>point and grounded>no_latent survive on more
faithful geometry (Codex: N=29 small -> bootstrap over OBJECTS, not episodes).

    CDWM_GATEB_SRC=coacd python eval_coacd.py
"""
import os, json
import numpy as np, torch
from torch.utils.data import DataLoader
from drop.my_dataset_gateb import GateBDS
from drop.drop_diffusion import GateBDiT, GateBPoint
from common.dit import ddim_sample, cosine_acp
from drop.drop_net import sixd_to_R
from drop.gateb.generate_obj import basin_of
from drop.gateb.generate_coacd import align_mesh, mesh_stable_orientations
from drop.eval_gateb import boundaryness
from scipy.spatial.transform import Rotation as R

N_SAMP = 60
ARMS = {"point": ("none", None, "coacd_point", False),
        "no_latent": ("none", 13, "coacd_nolatent", False),
        "grounded": ("grounded", 17, "coacd_grounded", False),
        "abstract": ("abstract", 13, "coacd_abstract", True),      # specificity: abstract CoM (should NOT transfer)
        "shuffled": ("shuffled", 17, "coacd_shuffled", False)}     # specificity: wrong-CoM control
_MS = {}


def stables_mesh(o):                                          # basin frame = real-mesh stable poses (as at generation)
    if o not in _MS: _MS[o] = mesh_stable_orientations(align_mesh(o))[0]
    return _MS[o]


@torch.no_grad()
def eval_arm(arm, dev):
    lm, nf, tag, use_lat = ARMS[arm]
    os.environ["CDWM_GATEB_LATENT"] = lm
    te = GateBDS("test")                                      # src=coacd from env; latent_mode from lm (13 vs 17 pts)
    st = {k: torch.tensor(v, device=dev) for k, v in te.stats.items()}
    model = (GateBPoint() if arm == "point" else GateBDiT(n_feat=nf, use_latent=use_lat)).to(dev)
    model.load_state_dict(torch.load(f"gateb_runs/{tag}/best.pt", map_location=dev)); model.eval()
    acp = cosine_acp(1000, device=dev)
    nll, top1, brier, ptop, cover = [], [], [], [], []
    for b in DataLoader(te, 128, num_workers=6):
        pts, br, cl, tb = [b[k].to(dev) for k in ("pts", "base_rel", "closing", "table")]
        B = pts.shape[0]; Rrel = sixd_to_R(br[:, 1:7]).cpu().numpy(); objs = b["object"]; truebasin = b["basin"].numpy()
        if arm == "point":
            pred = model.predict(pts, br, cl, tb)
            dR = sixd_to_R((pred * st["std"] + st["mean"])[:, :6]).cpu().numpy()[:, None]
        else:
            cond = model.cond(pts, br, cl, tb, b["latent"].to(dev) if use_lat else None)   # abstract feeds the latent vec
            x0 = ddim_sample(model, cond.repeat_interleave(N_SAMP, 0), 1, acp, steps=25, device=dev).squeeze(1)
            dR = sixd_to_R((x0 * st["std"] + st["mean"])[:, :6]).cpu().numpy().reshape(B, N_SAMP, 3, 3)
        for i in range(B):
            Rs = stables_mesh(objs[i]); K = len(Rs); bins = np.zeros(K)
            for s in range(dR.shape[1]):
                Rrest = dR[i, s] @ Rrel[i]; q = R.from_matrix(Rrest).as_quat()   # xyzw
                bins[basin_of(np.r_[q[3], q[:3]], Rs)] += 1
            p = (bins + 0.5) / (bins.sum() + 0.5 * K); t = int(truebasin[i]); oh = np.zeros(K); oh[t] = 1
            nll.append(-np.log(p[t])); top1.append(int(p.argmax() == t)); cover.append(int(bins[t] > 0))
            brier.append(((p - oh) ** 2).sum()); ptop.append(float(p.max()))
    return dict(obj=np.array(te.obj), basin=np.array(te.basin), boundary=boundaryness(te),
                nll=np.array(nll), top1=np.array(top1), brier=np.array(brier), ptop=np.array(ptop), cover=np.array(cover))


def ece(pt, ok, nb=10):
    e = 0.0
    for i in range(nb):
        m = (pt >= i / nb) & (pt < (i + 1) / nb)
        if m.sum(): e += m.mean() * abs(pt[m].mean() - ok[m].mean())
    return float(e)


def obj_bootstrap(gain, obj, nb=3000, seed=0):
    """object-level bootstrap of a paired per-episode gain: resample OBJECTS, take the mean of per-object mean gains."""
    uo = np.array(sorted(set(obj.tolist()))); per = np.array([gain[obj == o].mean() for o in uo])
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(per, len(per), replace=True).mean() for _ in range(nb)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(mean=float(per.mean()), lo=float(lo), hi=float(hi), n_pos=int((per > 0).sum()), n_obj=len(uo))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = {a: eval_arm(a, dev) for a in ARMS if os.path.exists(f"gateb_runs/{ARMS[a][2]}/best.pt")}
    obj = res["point"]["obj"]; hi_b = res["point"]["boundary"] > np.median(res["point"]["boundary"])
    out = {}
    for a, m in res.items():
        agg = lambda mask=None: dict(nll=float(m["nll"][mask].mean() if mask is not None else m["nll"].mean()),
                                     top1=float(m["top1"][mask].mean() if mask is not None else m["top1"].mean()),
                                     brier=float(m["brier"][mask].mean() if mask is not None else m["brier"].mean()),
                                     cover=float(m["cover"][mask].mean() if mask is not None else m["cover"].mean()),
                                     ece=ece(m["ptop"] if mask is None else m["ptop"][mask],
                                             m["top1"] if mask is None else m["top1"][mask]),
                                     n=int(len(m["nll"]) if mask is None else int(mask.sum())))
        out[a] = {"all": agg(), "boundary": agg(hi_b)}
        print(f"  {a:11s} ALL   nll {out[a]['all']['nll']:.3f} top1 {out[a]['all']['top1']:.2f} "
              f"cov {out[a]['all']['cover']:.2f} brier {out[a]['all']['brier']:.3f} ece {out[a]['all']['ece']:.3f}")
        print(f"  {'':11s} BNDRY nll {out[a]['boundary']['nll']:.3f} top1 {out[a]['boundary']['top1']:.2f}")
    # ABSOLUTE claims, object-bootstrap CIs over the 29
    dp = obj_bootstrap(res["point"]["nll"] - res["no_latent"]["nll"], obj)      # dist > point
    gc = obj_bootstrap(res["no_latent"]["nll"] - res["grounded"]["nll"], obj)   # grounded (hidden CoM) > no_latent
    print(f"\n[CoACD e2e] dist>point  NLL gain {dp['mean']:+.3f} [{dp['lo']:+.3f},{dp['hi']:+.3f}] "
          f"({dp['n_pos']}/{dp['n_obj']} obj) SIG={dp['lo'] > 0}")
    print(f"[CoACD e2e] grounded>no_latent (CoM causal) NLL gain {gc['mean']:+.3f} [{gc['lo']:+.3f},{gc['hi']:+.3f}] "
          f"({gc['n_pos']}/{gc['n_obj']} obj) SIG={gc['lo'] > 0}")
    out["_claims"] = {"dist_gt_point": dp, "grounded_gt_nolatent": gc}
    # SPECIFICITY (if the arms are trained): grounded should beat abstract (transfers) AND shuffled (correct-CoM control)
    for ctrl in ("abstract", "shuffled"):
        if ctrl in res:
            gg = obj_bootstrap(res[ctrl]["nll"] - res["grounded"]["nll"], obj)
            print(f"[CoACD e2e] grounded>{ctrl} NLL gain {gg['mean']:+.3f} [{gg['lo']:+.3f},{gg['hi']:+.3f}] "
                  f"({gg['n_pos']}/{gg['n_obj']} obj) SIG={gg['lo'] > 0}")
            out["_claims"][f"grounded_gt_{ctrl}"] = gg
    json.dump(out, open("gateb_runs/eval_coacd.json", "w"), indent=1)
    np.savez("gateb_runs/eval_coacd_pred.npz", obj=obj, boundary=hi_b,
             **{f"{a}_nll": res[a]["nll"] for a in res}, **{f"{a}_top1": res[a]["top1"] for a in res})
    print("\nwrote gateb_runs/eval_coacd.json + eval_coacd_pred.npz")


if __name__ == "__main__":
    main()
