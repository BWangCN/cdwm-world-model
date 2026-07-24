"""(#1 push) MDN eval: does geometry -> a calibrated DISTRIBUTION over contact outcomes? Compares MDN vs single-mean
direct vs labeled mode-conditioned mixture vs per-mode median on: (1) mixture NLL, (2) top-K oracle settle-rot error (does
a component cover the true outcome?), (3) mixture point-estimate error, (4) MDN-severity grasp ranking vs classifier P(fail),
(5) emergent-component vs labeled-mode alignment. Pure analysis of saved test_pred (aligned by test order).
    python eval_mdn.py
"""
import sys, math, collections, numpy as np
from common.utils import sixd_to_R, geodesic_deg
from grasp.eval_traj import errors, train_medians, mixture, MODES
L = lambda p: dict(np.load(p, allow_pickle=True))


def mdn_nll(w, mu, sc, y):                                   # physical-space diagonal-Gaussian mixture NLL (per-sample)
    sc = np.maximum(sc, 1e-6)
    logN = (-0.5 * ((y[:, None, :] - mu) / sc) ** 2 - np.log(sc) - 0.5 * math.log(2 * math.pi)).sum(-1)   # (N,K)
    m = (np.log(w + 1e-9) + logN).max(1)
    return -(m + np.log(np.sum(np.exp(np.log(w + 1e-9) + logN - m[:, None]), 1)))


def riskcov_auc(sev, fail):
    o = np.argsort(-sev); return round(float(np.mean([fail[o[int(r * len(o)):]].mean() for r in np.linspace(0, 0.5, 11)])), 3)


def main():
    mdn = L("traj_runs/mdn_full/test_pred.npz"); mdp = L("traj_runs/mdn_pose/test_pred.npz")
    tf = L("traj_runs/traj_full/test_pred.npz"); cls = L("cls_runs/full/test_pred.npz")
    try: dpl = L("traj_runs/direct_plain_full/test_pred.npz")
    except Exception: dpl = None
    y = mdn["y5"]; gt = mdn["summ_raw"]; med = train_medians(); fail = (y >= 2).astype(float)
    w, mu, sc = mdn["weight"], mdn["means"], mdn["scales"]                 # (N,K) (N,K,S) (N,K,S)
    mix_pt = np.einsum("nk,nks->ns", w, mu)                                # MDN mixture mean (point estimate)
    print("=== NLL (physical, lower=better) ===")
    print(f"  MDN_full {np.mean(mdn_nll(w, mu, sc, gt)):.2f}  MDN_pose {np.mean(mdn_nll(mdp['weight'], mdp['means'], mdp['scales'], gt)):.2f}")
    # top-K oracle settle-rot: best component vs GT (does the mixture COVER the true outcome?)
    Rg = sixd_to_R(gt[:, :6]); best = np.full(len(y), 1e9)
    for k in range(w.shape[1]): best = np.minimum(best, geodesic_deg(sixd_to_R(mu[:, k, :6]), Rg))
    print("\n=== settle-rot° (median) ===")
    print(f"  MDN top-K-oracle {np.median(best):.2f} | MDN mixture-mean {np.median(errors(mix_pt, gt)['settle_rot_deg']):.2f} "
          f"| labeled-mixture {np.median(errors(mixture(tf), gt)['settle_rot_deg']):.2f} | median {np.median(errors(med[y], gt)['settle_rot_deg']):.2f}"
          + (f" | single-mean(plain) {np.median(errors(dpl['pred_mean'], gt)['settle_rot_deg']):.2f}" if dpl else ""))
    # grasp ranking: MDN severity (expected max-drift + P(high-drift component)) vs classifier P(fail)
    sev_mdn = mix_pt[:, 9]; sev_var = np.einsum("nk,nks->ns", w, (mu - mix_pt[:, None]) ** 2)[:, 9]         # + predictive variance
    print("\n=== grasp-ranking risk-coverage AUC (lower=better; base fail {:.3f}) ===".format(fail.mean()))
    print(f"  classifier P(fail) {riskcov_auc(cls['p2'][:, 1], fail)} | MDN E[maxdrift] {riskcov_auc(sev_mdn, fail)} "
          f"| MDN E[maxdrift]+var {riskcov_auc(sev_mdn + np.sqrt(sev_var), fail)} | random {riskcov_auc(np.random.default_rng(0).random(len(y)), fail)}")
    # emergent components vs labeled modes
    comp = w.argmax(1); print("\n=== emergent MDN component (argmax weight) vs labeled mode — dominant mode per component ===")
    for k in sorted(set(comp.tolist())):
        c = collections.Counter(MODES[m] for m in y[comp == k]); tot = sum(c.values())
        top = c.most_common(2); print(f"  comp{k}: n{tot:5d}  " + "  ".join(f"{m} {100*v//tot}%" for m, v in top))


if __name__ == "__main__":
    main()
