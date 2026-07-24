"""GPU A/B control: eval ONE checkpoint with CPU-seeded initial noise, so the ONLY variable across runs is
(checkpoint, GPU). CPU generator -> byte-identical x_T on V100 and TITAN RTX -> pure GPU/checkpoint delta.
Dumps per-episode nll/brier/obj for a paired comparison. Reuses the no_latent config (none, use_lat=False, 13 feat).

    AB_CKPT=gateb_runs/no_latent_v100/best.pt AB_OUT=gpu_ab/v100ckpt_ttgpu.npz python gpu_ab.py
"""
import os, sys
import numpy as np, torch
from torch.utils.data import DataLoader
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("CDWM_GATEB_LATENT", "none")
from drop.my_dataset_gateb import GateBDS
from drop.drop_diffusion import GateBDiT, cosine_acp
from common.dit import ddim_sample
from drop.drop_net import sixd_to_R
from drop.density.v2 import drop_sweep as DS
from drop.gateb.generate_obj import stable_orientations, basin_of
from scipy.spatial.transform import Rotation as R

N = 40; SEED = 1234


@torch.no_grad()
def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = os.environ["AB_CKPT"]; out = os.environ["AB_OUT"]
    te = GateBDS("test"); st = {k: torch.tensor(v, device=dev) for k, v in te.stats.items()}
    model = GateBDiT(n_feat=13, use_latent=False).to(dev)
    model.load_state_dict(torch.load(ck, map_location=dev)); model.eval()
    acp = cosine_acp(1000, device=dev); g = torch.Generator().manual_seed(SEED)   # CPU gen -> device-independent noise
    Rc = {}; nll, brier, OBJ = [], [], []
    for b in DataLoader(te, 128, num_workers=6):                                    # shuffle=False -> deterministic order
        pts, br, cl, tb = [b[k].to(dev) for k in ("pts", "base_rel", "closing", "table")]
        B = pts.shape[0]; Rrel = sixd_to_R(br[:, 1:7]).cpu().numpy()
        cond = model.cond(pts, br, cl, tb, None)
        xT = torch.randn(B * N, 1, 9, generator=g)                                 # seeded on CPU, same across GPUs/runs
        x0 = ddim_sample(model, cond.repeat_interleave(N, 0), 1, acp, steps=25, device=dev, x_T=xT).squeeze(1)
        dR = sixd_to_R((x0 * st["std"] + st["mean"])[:, :6]).cpu().numpy().reshape(B, N, 3, 3)
        objs = b["object"]; tb_ = b["basin"].numpy()
        for i in range(B):
            o = objs[i]
            if o not in Rc: Rc[o] = stable_orientations(DS.hull(o))[0]
            Rs = Rc[o]; K = len(Rs); bins = np.zeros(K)
            for s in range(N):
                q = R.from_matrix(dR[i, s] @ Rrel[i]).as_quat(); bins[basin_of(np.r_[q[3], q[:3]], Rs)] += 1
            p = (bins + 0.5) / (bins.sum() + 0.5 * K); t = min(int(tb_[i]), K - 1); oh = np.zeros(K); oh[t] = 1
            nll.append(-np.log(p[t])); brier.append(((p - oh) ** 2).sum()); OBJ.append(o)
    nll = np.array(nll); brier = np.array(brier)
    print(f"CKPT {ck} | GPU {torch.cuda.get_device_name(0)} | NLL {nll.mean():.4f} Brier {brier.mean():.4f} n={len(nll)}")
    os.makedirs(os.path.dirname(out), exist_ok=True); np.savez(out, nll=nll, brier=brier, obj=np.array(OBJ))


if __name__ == "__main__":
    main()
