"""Compare arms (baseline vs +frame [vs +channels]) against the matched floor and the PRE-REGISTERED success criteria:
(1) circ_corr 0.50 -> above (primary), (2) held dir_err 5-15/15-30 drops >1° (noise floor), (3) mag not regressed,
(4) concave high-tilt improves, (5) train+held both improve (modeling, not overfit). Writes compare_summary.json."""
import json, numpy as np, os
D = os.path.dirname(os.path.abspath(__file__))
FLOOR = {b: json.load(open(os.path.join(D, "floor_summary.json")))["per_band"][b]["floor_held"] for b in ["2-5", "5-15", "15-30", "30+"]}
BANDS = ["5-15", "15-30", "30+"]


def load(tag): return json.load(open(f"{D}/eval_{tag}.json")) if os.path.exists(f"{D}/eval_{tag}.json") else None
def m(x): return x[0] if x else None
def s(x): return x[1] if x else None


def main():
    base = load("base700"); frame = load("frame"); chan = load("channels")
    arms = [("baseline", base), ("+frame", frame)] + ([("+frame+chan", chan)] if chan else [])
    out = {"floor": FLOOR, "criteria": {}}
    print("="*94)
    print("PER-BAND held-out dir_err (mean±std, 3 seeds, dir_err-selected @700ep) + circ_corr")
    hdr = f"{'band':6s} {'floor':>6s} | " + " | ".join(f"{n:>16s}" for n, _ in arms)
    print(hdr)
    for b in BANDS:
        row = f"{b:6s} {FLOOR[b]:6.1f} | " + " | ".join(f"{m(a['held_dir'][b]):5.1f}±{s(a['held_dir'][b]):<3.1f}" if a else f"{'-':>16s}" for _, a in arms)
        print(row)
    print(f"\n{'circ_corr':6s}        | " + " | ".join(f"{m(a['mech']['circ_corr'])!s:>16s}" if a else "-" for _, a in arms) + "   (GT-tracking; baseline stuck 0.50)")
    print(f"{'wm_vs_cl':6s}        | " + " | ".join(f"{m(a['mech']['wm_vs_closing'])!s:>16s}" if a else "-" for _, a in arms) + f"   (GT {m(base['mech']['gt_vs_closing']) if base else '?'})")
    # criteria vs baseline (for +frame; +chan if present)
    for name, arm in arms[1:]:
        if not arm: continue
        c = {}
        c["1_mechanism_circ_corr"] = dict(baseline=m(base["mech"]["circ_corr"]), arm=m(arm["mech"]["circ_corr"]),
                                          pass_=bool(m(arm["mech"]["circ_corr"]) is not None and m(arm["mech"]["circ_corr"]) > m(base["mech"]["circ_corr"]) + 0.03))
        c["2_dir_err"] = {b: dict(baseline=m(base["held_dir"][b]), arm=m(arm["held_dir"][b]), delta=round(m(arm["held_dir"][b])-m(base["held_dir"][b]), 1),
                                  pass_=bool(m(base["held_dir"][b])-m(arm["held_dir"][b]) > 1.0)) for b in ["5-15", "15-30"]}
        c["3_mag_not_regressed"] = {b: dict(baseline=m(base["held_mag"][b]), arm=m(arm["held_mag"][b]),
                                            pass_=bool(m(arm["held_mag"][b]) <= m(base["held_mag"][b]) + 1.0)) for b in BANDS}
        c["4_concave_hitilt"] = {b: dict(baseline=m(base["held_dir_shape"]["concave"][b]), arm=m(arm["held_dir_shape"]["concave"][b])) for b in ["15-30", "30+"]}
        c["5_train_and_held"] = {b: dict(held_delta=round(m(arm["held_dir"][b])-m(base["held_dir"][b]), 1),
                                         train_delta=round(m(arm["train_dir"][b])-m(base["train_dir"][b]), 1)) for b in ["5-15", "15-30"]}
        c["gap_vs_floor"] = {b: dict(baseline_gap=round(m(base["held_dir"][b])-FLOOR[b], 1), arm_gap=round(m(arm["held_dir"][b])-FLOOR[b], 1),
                                     pct_closed=round(100*(m(base["held_dir"][b])-m(arm["held_dir"][b]))/(m(base["held_dir"][b])-FLOOR[b]), 0) if m(base["held_dir"][b])-FLOOR[b] > 0 else None) for b in ["5-15", "15-30"]}
        out["criteria"][name] = c
        print(f"\n=== {name} vs baseline — PRE-REGISTERED CRITERIA ===")
        print(f"  (1) circ_corr {c['1_mechanism_circ_corr']['baseline']} -> {c['1_mechanism_circ_corr']['arm']}   PASS={c['1_mechanism_circ_corr']['pass_']}   <<< PRIMARY")
        for b in ["5-15", "15-30"]:
            print(f"  (2) dir_err {b}: {c['2_dir_err'][b]['baseline']} -> {c['2_dir_err'][b]['arm']} ({c['2_dir_err'][b]['delta']:+}) PASS={c['2_dir_err'][b]['pass_']} | gap {c['gap_vs_floor'][b]['baseline_gap']}->{c['gap_vs_floor'][b]['arm_gap']} ({c['gap_vs_floor'][b]['pct_closed']}%)")
        print(f"  (3) mag not regressed: " + " ".join(f"{b}:{'OK' if c['3_mag_not_regressed'][b]['pass_'] else 'REGRESSED'}" for b in BANDS))
        print(f"  (4) concave hi-tilt: " + " ".join(f"{b} {c['4_concave_hitilt'][b]['baseline']}->{c['4_concave_hitilt'][b]['arm']}" for b in ["15-30", "30+"]))
        print(f"  (5) train vs held: " + " ".join(f"{b} held{c['5_train_and_held'][b]['held_delta']:+}/train{c['5_train_and_held'][b]['train_delta']:+}" for b in ["5-15", "15-30"]))
    json.dump(out, open(f"{D}/compare_summary.json", "w"), indent=1)
    print("\nwrote compare_summary.json")


if __name__ == "__main__":
    main()
