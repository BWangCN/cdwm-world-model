"""Precompute K=8 settle-transient anchor orientations per corpus episode, for the trajectory-supervision arm
(does supervising the impact->settle transient improve the endpoint?). Design per Codex convergence:
- anchors span CONTACT ONSET -> SETTLE only (during-fall frames carry zero signal: zero-velocity kinematic release
  means orientation is constant and height is closed-form ballistics until contact);
- contact onset k0 = first frame whose orientation deviates >0.5 deg from release (orientation cannot change before
  contact), fallback = the ballistic impact estimate ceil(sqrt(2h/g)*fps); anchors = K evenly spaced k0..N-1;
- per-sample duration (anchor fractions of the observed transient), K=8 = an implementation convention (half the
  gateb rollout K=16; corpus transients are short, n_frames 13-52).
Iterates the tars in EXACTLY precompute_drop's order so rows align with derived/drop_index.csv; stores the release
quat per row so the loader can assert alignment against drop_poses.npz.
    python -m drop.precompute_droptraj
"""
import os, io, glob, tarfile
import numpy as np

from common.paths import DROP_CORPUS as DROP
DERIVED = os.path.join(DROP, "derived")
K, FPS, G = 8, 31.25, 9.81


def geod_deg(q, q0):                                        # (N,4) wxyz vs (4,) wxyz -> degrees
    d = np.abs(q @ q0)
    return np.degrees(2 * np.arccos(np.clip(d, -1, 1)))


def anchors(oq, drop_h):
    n = len(oq)
    dev = geod_deg(np.asarray(oq, np.float64), np.asarray(oq[0], np.float64))
    moved = np.nonzero(dev > 0.5)[0]
    k0 = int(moved[0]) if len(moved) else int(np.ceil(np.sqrt(2 * max(drop_h, 1e-4) / G) * FPS))
    k0 = min(max(k0, 1), n - 1)
    ks = np.linspace(k0, n - 1, K).round().astype(int)
    return np.asarray(oq, np.float32)[ks], k0


def main():
    TQ, K0, RQ = [], [], []
    for tp_ in sorted(glob.glob(f"{DROP}/train_corpus/*.tar")):
        with tarfile.open(tp_) as tf:
            for m in tf.getmembers():
                if not m.name.endswith(".npz"): continue
                z = np.load(io.BytesIO(tf.extractfile(m).read()), allow_pickle=True)
                tq, k0 = anchors(z["obj_quat"], float(z["drop_h"]))
                TQ.append(tq); K0.append(k0); RQ.append(np.asarray(z["obj_quat"][0], np.float32))
        print(f"  {os.path.basename(tp_)}: {len(TQ)} episodes so far", flush=True)
    TQ = np.stack(TQ); K0 = np.array(K0, np.int32); RQ = np.stack(RQ)
    np.savez(f"{DERIVED}/drop_traj{K}.npz", traj_quat=TQ, k0=K0, release_quat=RQ)
    print(f"wrote {DERIVED}/drop_traj{K}.npz  traj_quat {TQ.shape}  k0 med {int(np.median(K0))} max {K0.max()}")


if __name__ == "__main__":
    main()
