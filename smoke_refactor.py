"""Refactor smoke gate (Codex #3): import EVERY module in common/grasp/drop (catches import/path/circular breakage),
check path resolution, and run one tiny CPU forward for the shared encoder + both drop heads with fake tensors.

    cd <repo> && PYTHONPATH=. python smoke_refactor.py
"""
import importlib, pkgutil
import torch
import common, grasp, drop


def import_all():
    struct, data = [], []                                    # structure breakage (blocks) vs data-not-staged (warn)
    for pkg in (common, grasp, drop):
        for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
            try:
                importlib.import_module(m.name)
            except (FileNotFoundError, OSError) as e:
                data.append((m.name, repr(e)[:120]))
            except Exception as e:
                struct.append((m.name, repr(e)[:240]))
    return struct, data


def main():
    struct, data = import_all()
    print(f"[imports] structure-failures={len(struct)}  data-missing(non-blocking)={len(data)}")
    for n, e in struct:
        print("   STRUCT-FAIL", n, "::", e)
    for n, e in data:
        print("   data-missing", n)

    from common.paths import resolve_report
    print("[paths]")
    for k, (v, ok) in resolve_report().items():
        print(f"   {'OK  ' if ok else 'MISS'} {k} = {v}")

    from common.dit_local import DiTWMLocal
    from drop.drop_diffusion import GateBDiT, GateBPoint
    B, N = 2, 64
    pts, br, tb = torch.randn(B, N, 14), torch.randn(B, 9), torch.zeros(B, 1)
    fwd_ok = True
    with torch.no_grad():
        for name, fn in [("encoder(grasp shared, closing=H32)", lambda: DiTWMLocal(n_feat=13).encode(pts, br, torch.zeros(B, 32), tb)),
                         ("drop GateBDiT.cond", lambda: GateBDiT(n_feat=13, use_latent=False).cond(pts, br, torch.zeros(B, 1), tb, None)),
                         ("drop GateBPoint.predict", lambda: GateBPoint(n_feat=13).predict(pts, br, torch.zeros(B, 1), tb))]:
            try:
                o = fn(); print(f"   OK   {name} -> {tuple(o.shape)}")
            except Exception as e:
                fwd_ok = False; print(f"   FWD-FAIL {name} :: {repr(e)[:200]}")

    print("SMOKE", "PASS" if (not struct and fwd_ok) else "FAIL", f"({len(data)} data-missing modules load stats at import; fine once data present)")


if __name__ == "__main__":
    main()
