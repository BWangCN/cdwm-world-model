"""Paired GPU A/B analysis for the `diff` config (real V100 ckpt vs TITAN retrain). All runs use GateBDS("test")
shuffle=False + identical CPU-seeded noise -> per-episode aligned. Object-bootstrap CI on each mean delta.

  A = v100ckpt @ TITAN   B = titanckpt @ TITAN   C = v100ckpt @ V100 (optional)
  TRAINING-GPU effect = A-B (both eval'd on TITAN, only the ckpt's training GPU differs)
  INFERENCE-GPU effect = A-C (same v100 ckpt, only eval GPU differs; expect ~0)

    python gpu_ab_compare.py gpu_ab/diff_v100_ttgpu.npz gpu_ab/diff_tt_ttgpu.npz [gpu_ab/diff_v100_v100gpu.npz]
"""
import os, sys
import numpy as np


def load(p):
    d = np.load(p, allow_pickle=True); return d["nll"], d["brier"], d["obj"]


def oboot(delta, obj, seed=0):
    rng = np.random.default_rng(seed); uo = np.unique(obj); idx = {o: np.where(obj == o)[0] for o in uo}
    bs = [np.mean(delta[np.concatenate([idx[o] for o in rng.choice(uo, len(uo), True)])]) for _ in range(3000)]
    return float(np.mean(delta)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    A_p, B_p = sys.argv[1], sys.argv[2]
    C_p = sys.argv[3] if len(sys.argv) > 3 and os.path.exists(sys.argv[3]) else None
    An, Ab, Ao = load(A_p); Bn, Bb, Bo = load(B_p)
    assert (Ao == Bo).all(), "episode order mismatch A vs B"
    print(f"n={len(Ao)} episodes, {len(np.unique(Ao))} held-out objects\n")
    print(f"{'run':22s} {'meanNLL':>8s} {'meanBrier':>9s}")
    print(f"{'A v100ckpt @ TITAN':22s} {An.mean():8.4f} {Ab.mean():9.4f}")
    print(f"{'B titanckpt @ TITAN':22s} {Bn.mean():8.4f} {Bb.mean():9.4f}")
    print("\nTRAINING-GPU effect (v100- vs titan-trained `diff`, both eval'd @TITAN, seeded noise):")
    pt, lo, hi = oboot(An - Bn, Ao); print(f"  NLL  A-B: {pt:+.5f} [{lo:+.5f},{hi:+.5f}]")
    pt, lo, hi = oboot(Ab - Bb, Ao); print(f"  Brier A-B: {pt:+.5f} [{lo:+.5f},{hi:+.5f}]")
    if C_p:
        Cn, Cb, Co = load(C_p); assert (Ao == Co).all(), "episode order mismatch A vs C"
        print(f"\n{'C v100ckpt @ V100':22s} {Cn.mean():8.4f} {Cb.mean():9.4f}")
        print("\nINFERENCE-GPU effect (same v100 ckpt, TITAN vs V100, seeded noise; expect ~0):")
        pt, lo, hi = oboot(An - Cn, Ao); print(f"  NLL  A-C: {pt:+.5f} [{lo:+.5f},{hi:+.5f}]  |maxabs|={np.abs(An-Cn).max():.4g}")
    else:
        print("\n(no V100-eval file -> inference-GPU test skipped)")


if __name__ == "__main__":
    main()
