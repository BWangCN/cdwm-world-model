"""Render each demo object's stable resting poses (the 'basins') as labeled images, so basin 0/1/2 is interpretable.
A basin = one stable resting pose from trimesh.compute_stable_poses; the index is that pose's probability rank
(basin 0 = most likely under a uniform-random drop). Also reports each pose's probability + which principal axis
points up. Writes notes/demo_videos/<obj>_basins.png (basins in a row).

    MUJOCO_GL=egl python render_basins.py
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, mujoco, imageio
from scipy.spatial.transform import Rotation as R
HERE = os.path.dirname(os.path.abspath(__file__))
from drop.density.v2 import drop_sweep as DS
from drop.gateb.generate_obj import stable_orientations
from drop.render_drop_demo import build_render, camera, FONT, W, H_, OUT
from PIL import Image, ImageDraw

OBJS = ["011_banana", "050_medium_clamp", "Razer_Taipan_Black_Ambidextrous_Gaming_Mouse"]
AXNAME = {0: "long", 1: "mid", 2: "short"}


def up_axis(H, Rmat):                                         # which principal axis points up when resting in this pose
    up_body = Rmat.T @ np.array([0.0, 0, 1])                  # world-up expressed in body frame
    return AXNAME[int(np.argmax(np.abs(H["axes"] @ up_body)))]


def render_pose(H, Rmat, rnd, cam):
    m = build_render(H, 0.0, H["axes"][0]); d = mujoco.MjData(m)
    quat = R.from_matrix(Rmat).as_quat(); q = np.array([quat[3], quat[0], quat[1], quat[2]])
    pz = -R.from_matrix(Rmat).apply(H["V"])[:, 2].min() + 0.002
    d.qpos[:3] = [0, 0, pz]; d.qpos[3:7] = q; d.qvel[:] = 0; mujoco.mj_forward(m, d)
    rnd.update_scene(d, camera=cam); return rnd.render().copy()


def main():
    for obj in OBJS:
        H = DS.hull(obj); Rs, P = stable_orientations(H)
        rnd = mujoco.Renderer(build_render(H, 0, H["axes"][0]), height=H_, width=W); cam = camera(H)
        tiles = []
        desc = []
        for i, (Rmat, p) in enumerate(zip(Rs, P)):
            ua = up_axis(H, Rmat); desc.append(f"basin {i}: P={p:.0%}, {ua}-axis up")
            im = Image.fromarray(render_pose(H, Rmat, rnd, cam)); dr = ImageDraw.Draw(im)
            dr.rectangle([0, 0, W, 38], fill=(0, 0, 0))
            dr.text((10, 6), f"basin {i}   P={p:.0%}   ({ua}-axis up)", fill=(255, 255, 255), font=FONT)
            tiles.append(np.asarray(im))
        strip = np.concatenate(tiles, axis=1)
        outp = os.path.join(OUT, f"{obj}_basins.png"); imageio.imwrite(outp, strip)
        print(f"[{obj}] {len(Rs)} basins | {'; '.join(desc)} -> {outp}")


if __name__ == "__main__":
    main()
