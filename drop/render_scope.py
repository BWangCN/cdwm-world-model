"""Scope-boundary evidence (Codex): ONE concave counterexample showing WHY concave objects are out of scope.
Two panels for the bull: LEFT = the real textured mesh at rest (the object we recognize); RIGHT = the actual
CONVEX HULL geometry the simulator and world model reason about (a legless blob). The hull has far fewer stable
resting modes than the mesh, so its resting predictions cannot capture how the real object settles.
    MUJOCO_GL=egl python -m drop.render_scope
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, mujoco, imageio, trimesh
from scipy.spatial.transform import Rotation as R
from PIL import Image, ImageDraw
import drop.render_grid as G                              # reuse render_seq / mesh_cam / band / wxyz / N
from drop.mesh_render_test import mesh_model, W, Hh
from drop.density.v2 import drop_sweep as DS
from drop.gateb.generate_obj import stable_orientations

OBJ = "Schleich_Hereford_Bull"
THR = 0.02


def build_hull_model(V):                                  # the convex hull the sim/WM actually collide, as a grey geom
    hull = trimesh.Trimesh(vertices=np.asarray(V)).convex_hull
    Vh = np.asarray(hull.vertices) - np.asarray(hull.vertices).mean(0)
    verts = " ".join(f"{x:.5f}" for x in Vh.flatten())
    xml = f"""<mujoco>
      <visual><headlight ambient="0.45 0.45 0.45" diffuse="0.7 0.7 0.7"/><global offwidth="{W}" offheight="{Hh}"/></visual>
      <asset>
        <mesh name="hull" vertex="{verts}"/>
        <material name="hull" rgba="0.63 0.66 0.70 1" specular="0.2" shininess="0.3"/>
        <texture name="grid" type="2d" builtin="checker" rgb1=".18 .26 .34" rgb2=".10 .15 .20" width="512" height="512"/>
        <material name="grid" texture="grid" texrepeat="8 8" reflectance=".1"/>
      </asset>
      <worldbody>
        <light pos="0.4 0.4 1.4" dir="-0.25 -0.25 -1" diffuse="0.8 0.8 0.8"/>
        <geom name="floor" type="plane" size="3 3 0.1" material="grid"/>
        <body name="obj" pos="0 0 0.15"><freejoint/><geom type="mesh" mesh="hull" material="hull"/></body>
      </worldbody></mujoco>"""
    return mujoco.MjModel.from_xml_string(xml), Vh


def main():
    mdl_m, sc, mt, off = mesh_model(OBJ); meshV = (np.asarray(mt.vertices) - mt.vertices.mean(0)) * sc
    H = DS.hull(OBJ); Rs, _ = stable_orientations(H)
    mdl_h, hullV = build_hull_model(H["V"])
    rnd_m = mujoco.Renderer(mdl_m, height=Hh, width=W); rnd_h = mujoco.Renderer(mdl_h, height=Hh, width=W)
    cam = G.mesh_cam(meshV)
    Tf, pr = mt.compute_stable_poses(n_samples=1, threshold=0.0)
    nm, nh = int((np.asarray(pr) >= THR).sum()), len(Rs)
    q_mesh = G.wxyz(R.from_matrix(Tf[0][:3, :3]).as_quat())      # most probable real-mesh resting pose
    q_hull = G.wxyz(R.from_matrix(Rs[0]).as_quat())              # most probable hull resting pose
    left = G.band(G.render_seq(mdl_m, rnd_m, cam, meshV, [q_mesh, q_mesh]),  f"real mesh: {nm} stable resting modes")
    right = G.band(G.render_seq(mdl_h, rnd_h, cam, hullV, [q_hull, q_hull]), f"convex hull the model rests: {nh} modes (legs gone)")
    frames = [np.concatenate([left[i], right[i]], axis=1) for i in range(len(left))]
    im = Image.new("RGB", (frames[-1].shape[1], frames[-1].shape[0] + 40), (15, 18, 24))
    im.paste(Image.fromarray(frames[-1]), (0, 40)); dr = ImageDraw.Draw(im)
    dr.text((12, 8), "Scope boundary: the sim and world model reason about the convex hull, not the mesh - concave objects lose resting modes", fill=(255, 210, 120), font=G.FONT)
    p = f"{os.path.dirname(os.path.abspath(__file__))}/figures/scope_boundary.png"
    imageio.imwrite(p, np.asarray(im))
    print(f"[{OBJ}] mesh modes={nm} hull modes={nh} | wrote {p} {np.asarray(im).shape}")


if __name__ == "__main__":
    main()
