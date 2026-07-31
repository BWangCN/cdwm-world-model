"""Expanded eval suite for the clean MTL grasp WM (gated_runs/grasp_mtl/best.pt) on the TEST split.

Supersedes eval_grasp_mtl.py's rotation-only endpoint metric with the community-standard set agreed with the
user + Codex (2026-07-31), biased toward pose usefulness + mode coverage + invalid-rollout rejection (this WM
feeds gs-native-datagen -> Diffusion Policy, so eval quality gates DATA quality):

  A. POINT ACCURACY (N=1 + best-of-K), full trajectory not just endpoint:
     - rotation geodesic (deg): ADE (mean over H) + FDE (endpoint)
     - translation error (cm):  ADE + FDE
     - ADD / ADD-S (cm): mesh-vertex pose error, ADD-S = symmetry-aware nearest-neighbour (cans/bottles)
  C. DISTRIBUTION (single-GT-per-condition => best-of-K family, not P/R):
     - Miss Rate at an object-relative tolerance (rot & ADD)
     - coverage = 1 - MissRate (does the K-sample set contain a good outcome)
     - sample spread (rot deg, trans cm) + spread-vs-error correlation (calibration proxy)
  D. PHYSICAL PLAUSIBILITY (GT-free; the per-rollout gate):
     - 6D orthonormality residual of the raw rot output
     - OOD: final settle magnitude (rot deg, trans cm) vs the TRAIN p99  (mirrors gs-native-datagen's guard)
     - residual motion over the last frames (a settled pose should have ~0 velocity)
  gate: AUROC / acc / slip-recall / rigid-kept + ECE (calibration of the slip head)

Tolerances are OBJECT-RELATIVE + SYMMETRY-AWARE (agreed): translation/ADD tol = fraction of the object's bbox
diagonal; rotation tol is a plain angle (tilt) — a yaw-only symmetry fold is applied to ADD-S via NN, and can be
extended to the rotation metric per-object if a symmetry table is supplied.

    python grasp_eval_suite.py [--S 8] [--ver 256] [--rot_tol_deg 10] [--pos_tol_frac 0.10]
Runs on GPU (DDIM sampling); submit via slurm/grasp_eval_suite.sbatch.
"""
import os, sys, json, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from my_dataset_grasp_mtl import GraspMTLDS
from my_dataset_outcomes import _raw_cloud
from wm.grasp_mtl import GraspMTL
from wm.dit import cosine_acp, ddim_sample
from wm.common import sixd_to_R, geodesic_deg, quat_to_R
from torch.utils.data import DataLoader


def auroc(y, s):
    o = np.argsort(-s); y = y[o]; P = y.sum(); N = len(y) - P
    return float(np.trapz(np.cumsum(y) / P, np.cumsum(1 - y) / N)) if P and N else float("nan")


def ece(y, p, bins=10):                                             # expected calibration error of the slip prob
    e = 0.0; n = len(y)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (p >= lo) & (p < hi if b < bins - 1 else p <= hi)
        if m.sum() == 0: continue
        e += m.sum() / n * abs(p[m].mean() - y[m].mean())
    return float(e)


_MCACHE = {}
def obj_cloud(oid, ver):                                           # object-frame gaussian-cloud points (m) + bbox diagonal (m)
    if oid in _MCACHE: return _MCACHE[oid]
    V = np.asarray(_raw_cloud(oid)["mu"], np.float64)              # the SAME cloud the model conditions on (object frame)
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))              # object-relative tolerance scale
    if len(V) > ver:
        idx = np.linspace(0, len(V) - 1, ver).astype(int); V = V[idx]
    _MCACHE[oid] = (V, diag); return _MCACHE[oid]


def cloud_in_gripper(V, t0):                                       # object cloud (obj frame) -> gripper frame at t=0 (settle-target frame)
    R0, p0, Rg, bp = quat_to_R(t0[:4]), t0[4:7], quat_to_R(t0[7:11]), t0[11:14]
    Vw = V @ R0.T + p0                                             # object -> world
    return (Vw - bp) @ Rg                                          # world -> gripper (Rg^T (x-bp))


def add_pair(V, Rp, tp, Rg, tg, symmetric):                        # ADD (mean) / ADD-S (NN) between pred & GT posed verts (m)
    Pp = V @ Rp.T + tp; Pg = V @ Rg.T + tg
    if not symmetric:
        return float(np.linalg.norm(Pp - Pg, axis=1).mean())
    d = np.linalg.norm(Pp[:, None, :] - Pg[None, :, :], axis=2)     # NN (symmetry-aware)
    return float(d.min(1).mean())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=8)                    # samples per condition (best-of-K)
    ap.add_argument("--ver", type=int, default=256)               # mesh verts for ADD
    ap.add_argument("--rot_tol_deg", type=float, default=10.0)    # "good enough" tilt tolerance
    ap.add_argument("--pos_tol_frac", type=float, default=0.10)   # pos/ADD tol = frac of bbox diagonal (object-relative)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0)
    ck = torch.load("gated_runs/grasp_mtl/best.pt", map_location=dev)
    m = GraspMTL(n_feat=13, H=32, D=256, depth=4).to(dev); m.load_state_dict(ck["model"]); m.eval()
    stats = ck["stats"]; sd, mu = stats["std"].astype(np.float64), stats["mean"].astype(np.float64)
    acp = cosine_acp(1000, device=dev)

    # ---- TRAIN p99 of settle magnitude (OOD reference) from GT rigid trajectories ----
    tr = GraspMTLDS("train", stats=stats)
    Tg = tr.target[tr.is_rigid.astype(bool)]                       # (n,H,9) de-normalized (dataset stores raw target)
    fin = Tg[:, -1, :]
    rot_mag = geodesic_deg(sixd_to_R(fin[:, 3:9].astype(np.float32)), np.tile(np.eye(3), (len(fin), 1, 1)).astype(np.float32))
    trans_mag = np.linalg.norm(fin[:, :3], axis=1) * 100.0
    p99 = dict(rot=float(np.percentile(rot_mag, 99)), trans_cm=float(np.percentile(trans_mag, 99)))

    te = GraspMTLDS("test", stats=stats); dl = DataLoader(te, 128, num_workers=6)
    # accumulators
    A = {k: [] for k in ["rot_ade1", "rot_fde1", "rot_adeK", "rot_fdeK",
                          "trn_ade1", "trn_fde1", "trn_adeK", "trn_fdeK",
                          "add1", "addK", "adds1", "addsK",
                          "spread_rot", "spread_trn", "orth", "ood", "resid"]}
    PS, YS = [], []
    for b in dl:
        cond = m.encode(*[b[k].to(dev) for k in ("pts", "base_rel", "closing", "table")])
        PS.append(torch.softmax(m.classify(cond), 1)[:, 1].cpu().numpy()); YS.append(b["y_slip"].numpy())
        rig = b["is_rigid"].bool()
        if not rig.any(): continue
        cr = cond[rig]; objs = [o for o, r in zip(b["object"], rig.numpy()) if r]
        t0b = b["t0"].numpy()[rig.numpy()]                        # (nr,14) for the ADD gripper-frame transform
        gt = b["target"].numpy()[rig.numpy()] * sd + mu            # (nr,H,9) de-normalized GT trajectory
        nr, H = gt.shape[0], gt.shape[1]
        Rg = np.stack([sixd_to_R(gt[i, :, 3:9].astype(np.float32)) for i in range(nr)])  # (nr,H,3,3)
        tg = gt[:, :, :3]                                          # (nr,H,3) m
        # draw S samples (full trajectory)
        xs = []
        for s in range(a.S):
            x0 = ddim_sample(m, cr, H, acp, steps=50, device=dev).cpu().numpy() * sd + mu  # (nr,H,9)
            xs.append(x0)
        xs = np.stack(xs)                                          # (S,nr,H,9)
        Rp = np.stack([[sixd_to_R(xs[s, i, :, 3:9].astype(np.float32)) for i in range(nr)] for s in range(a.S)])  # (S,nr,H,3,3)
        tp = xs[:, :, :, :3]                                       # (S,nr,H,3) m
        for i in range(nr):
            V_obj, diag = obj_cloud(objs[i], a.ver); V = cloud_in_gripper(V_obj, t0b[i]); postol = a.pos_tol_frac * diag
            # per-sample errors
            rot_ade_s = np.array([geodesic_deg(Rp[s, i], Rg[i]).mean() for s in range(a.S)])
            rot_fde_s = np.array([geodesic_deg(Rp[s, i, -1:], Rg[i, -1:])[0] for s in range(a.S)])
            trn_ade_s = np.array([np.linalg.norm(tp[s, i] - tg[i], axis=1).mean() * 100 for s in range(a.S)])
            trn_fde_s = np.linalg.norm(tp[:, i, -1] - tg[i, -1], axis=1) * 100
            add_s  = np.array([add_pair(V, Rp[s, i, -1], tp[s, i, -1], Rg[i, -1], tg[i, -1], False) * 100 for s in range(a.S)])
            adds_s = np.array([add_pair(V, Rp[s, i, -1], tp[s, i, -1], Rg[i, -1], tg[i, -1], True) * 100 for s in range(a.S)])
            # A: N=1 (first sample) + best-of-K (by ADD, the pose-level criterion)
            k = int(np.argmin(add_s))
            A["rot_ade1"].append(rot_ade_s[0]); A["rot_fde1"].append(rot_fde_s[0])
            A["rot_adeK"].append(rot_ade_s[k]); A["rot_fdeK"].append(rot_fde_s[k])
            A["trn_ade1"].append(trn_ade_s[0]); A["trn_fde1"].append(trn_fde_s[0])
            A["trn_adeK"].append(trn_ade_s[k]); A["trn_fdeK"].append(trn_fde_s[k])
            A["add1"].append(add_s[0]); A["addK"].append(add_s[k])
            A["adds1"].append(adds_s[0]); A["addsK"].append(adds_s[k])
            # C: sample spread (endpoint) + store per-cond best rot for miss-rate
            ep_t = tp[:, i, -1]; A["spread_trn"].append(float(np.linalg.norm(ep_t - ep_t.mean(0), axis=1).mean() * 100))
            rot_pair = np.array([geodesic_deg(Rp[s, i, -1:], Rp[0, i, -1:])[0] for s in range(a.S)])
            A["spread_rot"].append(float(rot_pair.mean()))
            # miss flags stored alongside (rot uses FDE best-of-K, pos uses ADD best-of-K)
            A.setdefault("miss_rot", []).append(int(rot_fde_s[k] > a.rot_tol_deg))
            A.setdefault("miss_add", []).append(int(add_s[k] > postol * 100))
            A.setdefault("err_add_bestK", []).append(add_s[k]); A.setdefault("postol_cm", []).append(postol * 100)
            # D: physical plausibility (on sample 0)
            r6 = xs[0, i, -1, 3:9]; b1 = r6[:3] / (np.linalg.norm(r6[:3]) + 1e-9); b2 = r6[3:6]
            A["orth"].append(float(abs(np.dot(b1, b2 / (np.linalg.norm(b2) + 1e-9)))))         # |cos| of the two basis vecs (0 = orthonormal)
            fr = geodesic_deg(Rp[0, i, -1:], np.eye(3)[None].astype(np.float32))[0]
            ft = np.linalg.norm(tp[0, i, -1]) * 100
            A["ood"].append(int(fr > p99["rot"] or ft > p99["trans_cm"]))
            A["resid"].append(float(np.linalg.norm(tp[0, i, -1] - tp[0, i, -4], axis=0).mean() * 100))  # last-3-frame drift cm

    P = np.concatenate(PS); Y = np.concatenate(YS); pred = (P > 0.5).astype(int)
    med = lambda k: round(float(np.median(A[k])), 3)
    mean = lambda k: round(float(np.mean(A[k])), 4)
    out = {
        "n_rigid": len(A["add1"]), "n_test": len(Y), "S": a.S,
        "A_point_accuracy": {
            "rot_deg": {"ade_n1": med("rot_ade1"), "fde_n1": med("rot_fde1"), "ade_bestK": med("rot_adeK"), "fde_bestK": med("rot_fdeK")},
            "trans_cm": {"ade_n1": med("trn_ade1"), "fde_n1": med("trn_fde1"), "ade_bestK": med("trn_adeK"), "fde_bestK": med("trn_fdeK")},
            "ADD_cm":  {"n1": med("add1"), "bestK": med("addK")},
            "ADDS_cm": {"n1": med("adds1"), "bestK": med("addsK")},
        },
        "C_distribution": {
            "miss_rate_rot@%.0fdeg" % a.rot_tol_deg: mean("miss_rot"),
            "miss_rate_ADD@%.0f%%diag" % (a.pos_tol_frac * 100): mean("miss_add"),
            "coverage_ADD": round(1 - mean("miss_add"), 4),
            "spread_rot_deg_med": med("spread_rot"), "spread_trans_cm_med": med("spread_trn"),
            "calib_spread_err_corr": round(float(np.corrcoef(A["spread_rot"], A["err_add_bestK"])[0, 1]), 3),
        },
        "D_physical": {
            "orthonormality_resid_med": med("orth"), "ood_rate_vs_train_p99": mean("ood"),
            "residual_motion_cm_med": med("resid"), "train_p99": {k: round(v, 3) for k, v in p99.items()},
        },
        "gate": {"auroc": round(auroc(Y, P), 4), "acc": round(float((pred == Y).mean()), 4),
                 "slip_recall": round(float(pred[Y == 1].mean()), 4), "rigid_kept": round(float((pred[Y == 0] == 0).mean()), 4),
                 "ece": round(ece(Y, P), 4)},
        "tolerances": {"rot_tol_deg": a.rot_tol_deg, "pos_tol_frac_of_bbox_diag": a.pos_tol_frac},
    }
    print(json.dumps(out, indent=1)); json.dump(out, open("gated_runs/grasp_mtl/eval_suite.json", "w"), indent=1)
    print("\nwrote gated_runs/grasp_mtl/eval_suite.json", flush=True)


if __name__ == "__main__":
    main()
