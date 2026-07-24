"""STANDARD WM evaluation suite (reusable; pure numpy). Every WM eval should call metric_suite()/stratified().

THE THREE GAPS THIS CLOSES (vs the old r>15 / r_all / geo15 headline):
  (1) DIRECTION-BLINDNESS — the old correlation r(pred_tilt, gt_tilt) is between SCALAR magnitudes, so "right
      amount, wrong way" scores high. Here ALL rotation error is the DIRECTION-AWARE relative-rotation geodesic
      angle(R_pred^T @ R_gt) (captures magnitude AND axis), and we add an explicit sign-aware direction error.
  (2) TREND-ONLY — correlation is blind to systematic scale/offset and to absolute accuracy. Here the PRIMARY
      view is ABSOLUTE geodesic error (mean+median), not a correlation. Correlations are kept but SUPPLEMENTARY.
  (3) >15-deg FILTER DROPPED SMALL ROTATIONS — here we stratify by TRUE magnitude over ALL bins incl. [0-2),[2-5),
      [5-15) that the old headline never scored.

INPUTS are rotation MATRICES already in the consistent eval frame (the caller builds R_pred,R_gt via sixd_to_R on
the SE(3) target's 6-D rotation, in the same gripper frame used everywhere). Translation is OPTIONAL and secondary.

Identity used: angle(R_pred^T @ R_gt) == angle(R_pred @ R_gt^T) (trace is the Frobenius inner product, symmetric),
so this matches the existing geodesic_deg(R_pred,R_gt) — our past geo/geo15 numbers were ALREADY direction-aware;
it is the CORRELATION that was direction/scale blind.
"""
import numpy as np
from common.utils import rot_angle_axis  # (R)->(angle_deg, unit_axis) right-handed

# PRIMARY magnitude bins (deg). Deliberately include the small-rotation bins the >15 filter dropped.
STD_BANDS = [(0, 2), (2, 5), (5, 15), (15, 30), (30, 1e9)]
STD_BLAB = ["0-2", "2-5", "5-15", "15-30", "30+"]
AXIS_MIN_DEG = 2.0   # below this the rotation axis is ill-defined -> direction error masked out


def _stat(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return dict(mean=(round(float(v.mean()), 2) if len(v) else None),
                median=(round(float(np.median(v)), 2) if len(v) else None), n=int(len(v)))


def relrot_geodesic_deg(Rp, Rg):
    """DIRECTION-AWARE primary error: geodesic angle of the relative rotation R_pred^T @ R_gt (deg)."""
    Rr = np.matmul(np.swapaxes(Rp, -1, -2), Rg)
    tr = np.clip((Rr[..., 0, 0] + Rr[..., 1, 1] + Rr[..., 2, 2] - 1) / 2, -1, 1)
    return np.degrees(np.arccos(tr))


def net_angle_deg(R):
    """Scalar magnitude of a net rotation (deg) = how MUCH it rotates (direction-agnostic, for the decomposition)."""
    return rot_angle_axis(R)[0]


def decompose(Rp, Rg):
    """Split the error into AMOUNT vs WHICH-WAY:
       mag_err = | |Rp| - |Rg| |               (wrong-amount, deg)
       dir_err = angle between rotation AXES, SIGN-AWARE (NO abs: tip-left vs tip-right -> ~180), deg
                 masked (nan) where either rotation is below AXIS_MIN_DEG (axis ill-defined).
    Returns (geo, mag_err, dir_err, mag_gt) with geo = relrot_geodesic_deg (the combined primary)."""
    ap, axp = rot_angle_axis(Rp); ag, axg = rot_angle_axis(Rg)
    geo = relrot_geodesic_deg(Rp, Rg)
    mag_err = np.abs(ap - ag)
    cos = np.clip((axp * axg).sum(-1), -1, 1)          # NO abs -> sign-aware direction
    dir_err = np.degrees(np.arccos(cos))
    dir_err = np.where((ap > AXIS_MIN_DEG) & (ag > AXIS_MIN_DEG), dir_err, np.nan)
    return geo, mag_err, dir_err, ag


def _band_idx(t):
    for i, (lo, hi) in enumerate(STD_BANDS):
        if lo <= t < hi:
            return i
    return len(STD_BANDS) - 1


def metric_suite(Rp, Rg, t_pred=None, t_gt=None, pred_mag=None):
    """Full suite on a flat set of paired rotations (already best-of-N selected by the caller, if applicable).
       t_pred/t_gt optional translations (..,3) in METERS -> reported separately in cm.
       pred_mag optional explicit predicted magnitudes for the TREND corr (else derived from Rp)."""
    geo, mag_err, dir_err, mag_gt = decompose(Rp, Rg)
    pm = net_angle_deg(Rp) if pred_mag is None else np.asarray(pred_mag, float)
    bidx = np.array([_band_idx(t) for t in mag_gt])

    trans = None
    if t_pred is not None and t_gt is not None:
        terr = np.linalg.norm(np.asarray(t_pred) - np.asarray(t_gt), axis=-1) * 100.0  # m->cm

    bands = []
    for i, lab in enumerate(STD_BLAB):
        m = bidx == i
        if not m.sum():
            continue
        row = dict(band=lab, n=int(m.sum()),
                   geo=_stat(geo[m]),            # DIRECTION-AWARE absolute error (PRIMARY)
                   mag_err=_stat(mag_err[m]),    # wrong-amount
                   dir_err=_stat(dir_err[m]),    # wrong-way (sign-aware axis), nan-masked small rots
                   gt_mag=_stat(mag_gt[m]), pred_mag=_stat(pm[m]))
        if t_pred is not None and t_gt is not None:
            row["trans_cm"] = _stat(terr[m])
        bands.append(row)

    out = dict(
        primary_stratified=bands,                                  # << PRIMARY: abs geo per magnitude band
        overall_geo=_stat(geo),                                    # abs geo over ALL samples
        geo_gt15=_stat(geo[mag_gt > 15]),                          # continuity w/ old headline (now mean+median)
        decomposition_overall=dict(mag_err=_stat(mag_err), dir_err=_stat(dir_err[np.isfinite(dir_err)])),
        trend_ONLY=dict(                                           # SUPPLEMENTARY — blind to scale/offset/direction
            r_all=(round(float(np.corrcoef(pm, mag_gt)[0, 1]), 3) if len(pm) > 2 else None),
            r_gt15=(round(float(np.corrcoef(pm[mag_gt > 15], mag_gt[mag_gt > 15])[0, 1]), 3)
                    if (mag_gt > 15).sum() > 2 else None),
            note="correlation of SCALAR magnitudes — TREND/ranking only; blind to scale, offset, and direction"),
    )
    if t_pred is not None and t_gt is not None:
        out["trans_cm_overall"] = _stat(terr)
    return out


def stratified(Rp, Rg, t_pred=None, t_gt=None, pred_mag=None, strata=None):
    """metric_suite over 'all' plus, for each named stratum, over each of its values.
       strata = {'split': array, 'mechanism': array, 'class': array, ...} (any label arrays, len == n samples).
       Future cross-object runs pass class=GOOD/fail and mechanism=box/bottle/...; bowl passes variant/kind/split."""
    def sub(mask):
        kw = {}
        if t_pred is not None:
            kw = dict(t_pred=np.asarray(t_pred)[mask], t_gt=np.asarray(t_gt)[mask])
        pm = None if pred_mag is None else np.asarray(pred_mag)[mask]
        return metric_suite(Rp[mask], Rg[mask], pred_mag=pm, **kw)

    res = {"all": sub(np.ones(len(Rp), bool))}
    for name, lab in (strata or {}).items():
        lab = np.asarray(lab); res[name] = {}
        for v in sorted(set(lab.tolist())):
            res[name][str(v)] = sub(lab == v)
    return res


def fmt_primary(suite, title=""):
    """One-line-per-band text table of the PRIMARY stratified view (+ overall + trend footnote)."""
    L = [f"  {title}".rstrip(),
         f"  {'band':6s} {'n':>5s} | {'geo mean':>8s} {'geo med':>7s} | {'magErr md':>9s} {'dirErr md':>9s} | {'gtMag md':>8s} {'predMag md':>10s}"]
    for r in suite["primary_stratified"]:
        g, me, de, gm, pmg = r["geo"], r["mag_err"], r["dir_err"], r["gt_mag"], r["pred_mag"]
        de_md = "   -   " if de["median"] is None else f"{de['median']:7.1f}"
        L.append(f"  {r['band']:6s} {r['n']:5d} | {g['mean']:8.2f} {g['median']:7.2f} | "
                 f"{me['median']:9.1f} {de_md:>9s} | {gm['median']:8.1f} {pmg['median']:10.1f}")
    o = suite["overall_geo"]; t = suite["trend_ONLY"]
    L.append(f"  OVERALL geo mean {o['mean']} med {o['median']} (n{o['n']}) | "
             f">15 geo mean {suite['geo_gt15']['mean']} med {suite['geo_gt15']['median']} (n{suite['geo_gt15']['n']})")
    L.append(f"  [trend-only] r_all {t['r_all']}  r>15 {t['r_gt15']}  (magnitude correlation; not primary)")
    return "\n".join(L)
