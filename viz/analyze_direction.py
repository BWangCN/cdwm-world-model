"""STAGE 3 (env: gaussianobject). Is the WM's per-chunk axis error SYSTEMATIC or RANDOM? On a LARGER sample of NORMAL
near-success grasps (bands 5-15/15-30, establishment<60deg, mag_err<=8), compute the GT vs WM tilt-direction (gripper-frame
up-vector azimuth) and test: (a) does the WM predict a GENERIC direction (WM-azimuth spread << GT-azimuth spread)?
(b) does WM track the per-grasp GT direction (circular correlation)? (c) is the signed error biased (systematic) or zero-mean
(random)? Also compares the WM's default direction to the gripper CLOSING axis. Writes direction_analysis.json + a polar PNG."""
import os, sys, csv, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from make_videos import load_model, rollout
from wm.common import sixd_to_R, quat_to_R, geodesic_deg
import wm.metrics as MET

CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "example_data", "chunk_index", "chunks_corrected.csv")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "viz_out")
UP = np.array([0.0, 0.0, 1.0])


def circ_std(a):                                     # a: azimuths (rad); returns circular std (deg)
    R = np.hypot(np.mean(np.cos(a)), np.mean(np.sin(a)))
    return float(np.degrees(np.sqrt(-2*np.log(max(R, 1e-9)))))


def main():
    rows = [r for r in csv.DictReader(open(CSV)) if r["band"] in ("5-15", "15-30")]
    rng = np.random.default_rng(7); idx = rng.permutation(len(rows))[:280]
    cand = []
    for i in idx:
        r = rows[i]
        try:
            z = np.load(r["episode"], allow_pickle=True)
            est = geodesic_deg(quat_to_R(z["obj_quat"][0].astype(float)), np.eye(3))
            if est < 60:
                r = dict(r); r["_clv"] = z["closing"].astype(float); r["_bq"] = z["base_quat"][0].astype(float)
                cand.append(r)
        except Exception:
            pass
        if len(cand) >= 200: break
    m, acp, sd, mu, H = load_model(0)
    GT, PR, _ = rollout(m, acp, sd, mu, H, cand)
    Rg = sixd_to_R(GT[:, -1, 3:]); Rp = sixd_to_R(PR[:, -1, 3:])
    geo, mag, der, _ = MET.decompose(Rp, Rg)
    ok = np.isfinite(der) & (mag <= 8.0)             # near-success: magnitude right, axis defined
    G, P = Rg[ok], Rp[ok]; de = der[ok]; N = int(ok.sum())
    # tilt direction = gripper-frame up-vector azimuth (the "fall direction")
    gdir = (G @ UP)[:, :2]; pdir = (P @ UP)[:, :2]
    gaz = np.arctan2(gdir[:, 1], gdir[:, 0]); paz = np.arctan2(pdir[:, 1], pdir[:, 0])
    # WM default vs gripper CLOSING axis (does WM tilt along a grasp-fixed axis?): closing in gripper frame
    Rgb = np.array([quat_to_R(r["_bq"]) for r, o in zip(cand, ok) if o])
    clv = np.array([r["_clv"] for r, o in zip(cand, ok) if o])
    clg = np.einsum("nij,nj->ni", np.transpose(Rgb, (0, 2, 1)), clv)[:, :2]     # closing axis in gripper frame (2D)
    claz = np.arctan2(clg[:, 1], clg[:, 0])
    d_wm_cl = np.degrees(np.abs(np.angle(np.exp(1j*(paz - claz)))))             # WM-dir vs closing axis
    d_gt_cl = np.degrees(np.abs(np.angle(np.exp(1j*(gaz - claz)))))
    # circular correlation (does WM track GT per-grasp direction?)
    gb, pb = gaz - np.angle(np.mean(np.exp(1j*gaz))), paz - np.angle(np.mean(np.exp(1j*paz)))
    ccorr = float(np.sum(np.sin(gb)*np.sin(pb)) / np.sqrt(np.sum(np.sin(gb)**2)*np.sum(np.sin(pb)**2) + 1e-12))
    err = np.angle(np.exp(1j*(paz - gaz)))           # signed azimuth error (rad)
    res = dict(N=N, dir_err_median=round(float(np.median(de)), 1),
               GT_dir_spread_deg=round(circ_std(gaz), 1), WM_dir_spread_deg=round(circ_std(paz), 1),
               circular_corr_GT_WM=round(ccorr, 3),
               signed_err_mean_deg=round(float(np.degrees(np.angle(np.mean(np.exp(1j*err))))), 1),
               signed_err_absmedian_deg=round(float(np.degrees(np.median(np.abs(err)))), 1),
               WM_resultant_len=round(float(np.hypot(np.mean(np.cos(paz)), np.mean(np.sin(paz)))), 3),
               GT_resultant_len=round(float(np.hypot(np.mean(np.cos(gaz)), np.mean(np.sin(gaz)))), 3),
               WM_dir_vs_closingaxis_median_deg=round(float(np.median(d_wm_cl)), 1),
               GT_dir_vs_closingaxis_median_deg=round(float(np.median(d_gt_cl)), 1))
    json.dump(res, open(f"{OUT}/direction_analysis.json", "w"), indent=1)
    # figure: (1) GT vs WM azimuth scatter, (2) signed-error hist
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].scatter(np.degrees(gaz), np.degrees(paz), s=12, alpha=0.5, c="#c0392b")
    ax[0].plot([-180, 180], [-180, 180], "k--", lw=1, label="WM=GT (tracks grasp)")
    ax[0].set_xlabel("GT tilt direction (gripper-frame azimuth °)"); ax[0].set_ylabel("WM tilt direction °")
    ax[0].set_title(f"WM vs GT fall-direction  (circ corr={res['circular_corr_GT_WM']})"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].hist(np.degrees(err), bins=30, color="#f46d43"); ax[1].axvline(0, color="k", lw=1)
    ax[1].axvline(res["signed_err_mean_deg"], color="b", lw=1.5, label=f"mean {res['signed_err_mean_deg']}°")
    ax[1].set_xlabel("signed WM−GT direction error (°)"); ax[1].set_title("systematic (biased) vs random (zero-mean)?"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/direction_analysis.png", dpi=130); plt.close(fig)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
