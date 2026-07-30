"""Grasp-pose sampler (VLA pipeline step 5). Draws WM-NATIVE grasps from the grasp dataset (per-object
`grasps.npz`: grasp_point + approach + closing axes + net_tilt_deg stability), maps each into the FR3 2F-85
ee frame (approach -> ee local +z, closing -> ee local +y — the locked frame convention), gates every
candidate through the step-2 feasibility filter, and ranks the survivors by grasp stability. This is the
faithful grasp source for in-dataset objects (the grasp WM was trained on exactly these). A learned open
detector (GSNet -> AnyGrasp) is the novel-object generalization upgrade (deferred; heavy custom-CUDA build).

    python grasp_sampler.py
"""
import numpy as np, sapien
import cdwm_scene, gymnasium as gym, mani_skill
from feasibility import Feasibility, OBJ_Z, TCP_OFFSET

DSET = "/misc/kcgscratch1/MengyeGroup/yy5259/datasets/cdwm-grasp-dataset"


def R_to_quat(R):
    w = np.sqrt(max(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w < 1e-6: return np.array([1.0, 0, 0, 0])
    return np.array([w, (R[2, 1]-R[1, 2])/(4*w), (R[0, 2]-R[2, 0])/(4*w), (R[1, 0]-R[0, 1])/(4*w)])


class GraspSampler:
    def __init__(self, obj_id):
        self.obj_id = obj_id
        self.g = np.load(f"{DSET}/objects/{obj_id}/grasps.npz", allow_pickle=True)

    def ee_pose(self, i, R_obj, t_obj):
        """FR3 ee(2f85_base) world pose for grasp i, given the object's world rotation R_obj + position t_obj.
        approach -> ee local +z (into the object); closing -> ee local +y (jaw separation)."""
        gp = R_obj @ self.g["grasp_point"][i] + t_obj                 # TCP (grasp point) in world
        z = R_obj @ self.g["approach"][i]; z = z / (np.linalg.norm(z) + 1e-9)
        c = R_obj @ self.g["closing"][i]
        y = c - (c @ z) * z; y = y / (np.linalg.norm(y) + 1e-9)       # orthonormalize against approach
        x = np.cross(y, z)
        R_ee = np.stack([x, y, z], axis=1)                           # columns = world images of ee local x,y,z
        ee_pos = gp - TCP_OFFSET * z                                 # 2f85_base sits behind the TCP along -approach
        return sapien.Pose(p=ee_pos, q=R_to_quat(R_ee))

    def sample(self, F, obj_pos, obj_quat=(1, 0, 0, 0)):
        """Return feasible grasps for the object placed at (obj_pos, obj_quat), ranked by stability (net_tilt)."""
        R_obj = F_qR(obj_quat); t_obj = np.asarray(obj_pos, float)
        obj_center = t_obj + R_obj @ [0, 0, OBJ_Z]                    # AABB center for the intrusion test
        out = []
        for i in range(len(self.g["grasp_point"])):
            ee = self.ee_pose(i, R_obj, t_obj)
            ok, rep = F.check_grasp(ee, obj_center)
            appr_vert = float(np.degrees(np.arccos(np.clip(-(R_obj @ self.g["approach"][i])[2], -1, 1))))  # 0=top-down
            out.append({"i": i, "ok": ok, "net_tilt": float(self.g["net_tilt_deg"][i]),
                        "appr_from_vertical": appr_vert, "ee": ee, "rep": rep})
        feas = [o for o in out if o["ok"]]
        feas.sort(key=lambda o: o["net_tilt"])                       # most stable first
        return feas, out


def F_qR(q):
    w, x, y, z = np.asarray(q, float) / (np.linalg.norm(q) + 1e-12)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


if __name__ == "__main__":
    env = gym.make("CDWMScene-v0", obj_id="006_mustard_bottle", num_envs=1, render_mode="rgb_array",
                   sim_backend="physx_cpu", control_mode="pd_joint_pos")
    env.reset(seed=0)
    F = Feasibility(env)
    S = GraspSampler("006_mustard_bottle")
    feas, allg = S.sample(F, obj_pos=[0.0, 0.0, 0.0])                # object base at the origin, upright
    print(f"=== grasp sampler: 006_mustard_bottle, {len(allg)} dataset grasps ===", flush=True)
    print(f"FEASIBLE: {len(feas)}/{len(allg)} pass the feasibility filter\n", flush=True)
    print("top-8 feasible (ranked by stability / net_tilt):", flush=True)
    for o in feas[:8]:
        print(f"  grasp {o['i']:3d}: net_tilt={o['net_tilt']:5.1f}deg  approach {o['appr_from_vertical']:5.1f}deg from vertical", flush=True)
    # a few rejected + why
    rej = [o for o in allg if not o["ok"]][:4]
    print("\nsample rejected grasps (reasons):", flush=True)
    for o in rej:
        bad = [f"{k}:{v['reasons']}" for k, v in o["rep"].items() if v["reasons"]]
        print(f"  grasp {o['i']:3d}: approach {o['appr_from_vertical']:5.1f}deg from vertical -> {bad[:2]}", flush=True)
    env.close()
    print("\n=== done ===", flush=True)
