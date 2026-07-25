"""Test: can MuJoCo render the full TEXTURED mesh (not the convex hull) cleanly + aligned to the point-cloud frame?
Renders one object's mesh.obj + material_0.png at a resting orientation, on the floor. If good -> use for the demos.
    MUJOCO_GL=egl python mesh_render_test.py
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, mujoco, imageio, trimesh
from scipy.spatial.transform import Rotation as R
from drop.density.v2 import drop_sweep as DS
from common.paths import OBJECTS as OBJDIR
W, Hh = 800, 600


def mesh_model(obj):
    d = f"{OBJDIR}/{obj}"
    mu = np.load(f"{DS.PCDIR}/{obj}.npz")["mu"].astype(np.float64)          # the WM's point cloud (scale reference)
    m = trimesh.load(f"{d}/mesh.obj", force="mesh")
    scale = float((mu.max(0) - mu.min(0)).max() / (m.vertices.max(0) - m.vertices.min(0)).max())
    off = m.vertices.mean(0) * scale                                        # center the (scaled) mesh at origin, like the hull
    xml = f"""<mujoco>
      <compiler meshdir="{d}" texturedir="{d}"/>
      <visual><headlight ambient="0.45 0.45 0.45" diffuse="0.7 0.7 0.7"/><global offwidth="{W}" offheight="{Hh}"/></visual>
      <asset>
        <mesh name="m" file="mesh.obj" scale="{scale} {scale} {scale}" refpos="{off[0]} {off[1]} {off[2]}"/>
        <texture name="tex" type="2d" file="material_0.png"/>
        <material name="mat" texture="tex" specular="0.1" shininess="0.3"/>
        <texture name="grid" type="2d" builtin="checker" rgb1=".18 .26 .34" rgb2=".10 .15 .20" width="512" height="512"/>
        <material name="grid" texture="grid" texrepeat="8 8" reflectance=".1"/>
      </asset>
      <worldbody>
        <light pos="0.4 0.4 1.4" dir="-0.25 -0.25 -1" diffuse="0.8 0.8 0.8"/>
        <geom name="floor" type="plane" size="3 3 0.1" material="grid"/>
        <body name="obj" pos="0 0 0.15"><freejoint/><geom type="mesh" mesh="m" material="mat"/></body>
      </worldbody></mujoco>"""
    return mujoco.MjModel.from_xml_string(xml), scale, m, off


def main():
    obj = "Schleich_Spinosaurus_Action_Figure"
    mdl, scale, m, off = mesh_model(obj); d = mujoco.MjData(mdl)
    print(f"[{obj}] scale={scale:.4f} mesh_verts={len(m.vertices)} tris={len(m.faces)}")
    rnd = mujoco.Renderer(mdl, height=Hh, width=W)
    cam = mujoco.MjvCamera(); cam.lookat[:] = [0, 0, 0.08]; cam.distance = 0.55; cam.elevation = -18; cam.azimuth = 90
    # settle the object flat-ish on the floor so it looks natural
    d.qpos[:3] = [0, 0, 0.12]; d.qpos[3:7] = [1, 0, 0, 0]; d.qvel[:] = 0
    for _ in range(1500): mujoco.mj_step(mdl, d)
    mujoco.mj_forward(mdl, d); rnd.update_scene(d, camera=cam)
    imageio.imwrite(f"{HERE}/notes/demo_videos/_meshtest.png", rnd.render())
    print("wrote _meshtest.png (settled quat:", np.round(d.qpos[3:7], 2), ")")


if __name__ == "__main__":
    main()
