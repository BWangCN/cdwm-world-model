"""Object-level bootstrap CIs for a trained drop-WM arm (rigor: the basin_transition metrics ride on only ~44
held-out positives). Loads best.pt, dumps per-episode (geo err, no-motion err, bt label/prob, object), and
reports object-bootstrap 95% CIs. Reuses bootstrap_ci.oboot.

    python eval_drop.py --tag full_release --frame release   # CDWM_MODALITY from env
"""
import os, sys, json, argparse
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from my_dataset_drop import DropDS
from wm.drop_net import DropNet, sixd_to_R
from train_drop import geodesic, auroc

RNG = np.random.default_rng(0); B = 1000


def oboot(vals, obj, reduce=np.median):                    # object-level bootstrap (bootstrap_ci pattern)
    uo = np.unique(obj); idx = {o: np.where(obj == o)[0] for o in uo}
    pt = reduce(vals)
    bs = [reduce(vals[np.concatenate([idx[o] for o in RNG.choice(uo, len(uo), replace=True)])]) for _ in range(B)]
    return float(pt), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def oboot_auroc(prob, y, obj):                             # object-bootstrap AUROC (wide with few positives -> honest)
    uo = np.unique(obj); idx = {o: np.where(obj == o)[0] for o in uo}
    bs = []
    for _ in range(B):
        take = np.concatenate([idx[o] for o in RNG.choice(uo, len(uo), replace=True)])
        a = auroc(prob[take], y[take])
        if not np.isnan(a): bs.append(a)
    return float(auroc(prob, y)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(bs)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="full_release")
    ap.add_argument("--frame", default="release"); a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mod = os.environ.get("CDWM_MODALITY", "full")
    te = DropDS("test", modality=mod, frame=a.frame)
    model = DropNet().to(dev); model.load_state_dict(torch.load(f"drop_runs/{a.tag}/best.pt", map_location=dev)); model.eval()
    GEO, GEO0, PR, BT, OBJ = [], [], [], [], []
    for b in DataLoader(te, 512, num_workers=6):
        lbt, rot = model(b["pts"].to(dev), b["base_rel"].to(dev), b["closing"].to(dev), b["table"].to(dev))
        Rp = sixd_to_R(rot); Rg = sixd_to_R(b["rest6d"].to(dev)); I = torch.eye(3, device=dev).expand_as(Rg)
        GEO.append(geodesic(Rp, Rg).cpu().numpy()); GEO0.append(geodesic(I, Rg).cpu().numpy())
        PR.append(torch.softmax(lbt, 1)[:, 1].cpu().numpy()); BT.append(b["y_bt"].numpy()); OBJ += list(b["object"])
    geo = np.concatenate(GEO); geo0 = np.concatenate(GEO0); pr = np.concatenate(PR); bt = np.concatenate(BT); obj = np.array(OBJ)
    np.savez(f"drop_runs/{a.tag}/test_pred.npz", geo=geo, geo0=geo0, pr=pr, bt=bt, obj=obj)
    gm = oboot(geo, obj); g0 = oboot(geo0, obj); ar = oboot_auroc(pr, bt, obj)
    print(f"[{a.tag}] test n={len(bt)} objects={len(np.unique(obj))} bt_pos={int(bt.sum())}")
    print(f"  geo_med   model {gm[0]:.2f}° [{gm[1]:.2f},{gm[2]:.2f}]   no-motion {g0[0]:.2f}° [{g0[1]:.2f},{g0[2]:.2f}]")
    print(f"  bt AUROC  {ar[0]:.3f} [{ar[1]:.3f},{ar[2]:.3f}]  (object-bootstrap, {ar[3]}/{B} resamples had both classes)")
    json.dump(dict(tag=a.tag, geo_med=gm, nomotion_med=g0, auroc=ar[:3], n=len(bt), bt_pos=int(bt.sum())),
              open(f"drop_runs/{a.tag}/ci.json", "w"), indent=1)


if __name__ == "__main__":
    main()
