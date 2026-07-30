"""Feasibility filter (VLA pipeline step 2). Rejects synthesized grasp trajectories that are kinematically
infeasible or geometrically implausible BEFORE they become corpus data:
  - IK does not converge to the target EEF pose (unreachable),
  - arm joint-limit violation,
  - a link plows through the table (z < table top),
  - a NON-gripper link intrudes into the object AABB (the gripper itself is allowed to touch it).
Tolerances derive from the robot model + scene geometry (joint limits, table height, object half-extents),
not hand-tuned thresholds. v1 uses link-origin geometry (coarse); SAPIEN mesh-contact self-collision is a
v2 refinement. Reusable via Feasibility.check_grasp(...); standalone runs a good grasp + deliberately-bad ones.

    python feasibility.py
"""
import numpy as np, sapien, torch
import cdwm_scene  # registers CDWMScene-v0
import gymnasium as gym, mani_skill

OBJ_Z = cdwm_scene.CDWMScene.OBJ_HALF[2]
Q_DOWN = np.array([0.0, 1, 0, 0])                    # top-down gripper (approach along -world_z)
TCP_OFFSET = 0.112                                  # ee(2f85_base) -> TCP along approach (from pad geometry)


class Feasibility:
    def __init__(self, env):
        u = env.unwrapped; self.u = u; self.robot = u.agent.robot
        self.pino = self.robot.create_pinocchio_model()
        self.links = self.robot.get_links()
        self.ee_idx = next(i for i, l in enumerate(self.links) if l.name == "2f85_base")
        self.nq = int(self.robot.dof.item() if hasattr(self.robot.dof, "item") else self.robot.dof)
        self.arm_mask = np.zeros(self.nq); self.arm_mask[:7] = 1
        _bp = self.robot.pose
        self.base = sapien.Pose(_bp.p.cpu().numpy().reshape(-1), _bp.q.cpu().numpy().reshape(-1))
        self.base_inv = self.base.inv()
        ql = self.robot.get_qlimits().cpu().numpy().reshape(self.nq, 2)
        self.qmin, self.qmax = ql[:, 0], ql[:, 1]
        self.grip_links = {i for i, l in enumerate(self.links) if l.name.startswith("2f85_")}
        self.base_links = {i for i, l in enumerate(self.links) if l.name in ("dummy_root_0", "fr3_link0")}
        self.obj_half = np.array(cdwm_scene.CDWMScene.OBJ_HALF)

    def _ik(self, target_world, q0):
        t = self.base_inv * target_world
        res, ok, err = self.pino.compute_inverse_kinematics(self.ee_idx, t, initial_qpos=q0,
                                                            active_qmask=self.arm_mask, max_iterations=200)
        q = np.asarray(res).reshape(-1)
        self.pino.compute_forward_kinematics(q)
        ach = self.base * self.pino.get_link_pose(self.ee_idx)
        perr = float(np.linalg.norm(np.asarray(ach.p) - np.asarray(target_world.p)))
        return q, perr

    def _linkpos(self, qarm, grip=0.0):
        full = qarm.copy(); full[7:9] = grip
        self.robot.set_qpos(torch.tensor(full[None], dtype=torch.float32, device=self.u.device))
        return [self.links[i].pose.p.cpu().numpy().reshape(-1) for i in range(len(self.links))]

    def check_waypoint(self, target_world, q0, obj_center=None, grip=0.0, ptol=0.005, zmargin=0.01, omargin=0.02):
        q, perr = self._ik(target_world, q0); reasons = []
        if perr > ptol: reasons.append(f"ik_unconverged({perr:.3f}m)")
        bad = (q[:7] < self.qmin[:7] - 1e-3) | (q[:7] > self.qmax[:7] + 1e-3)
        if bad.any(): reasons.append(f"joint_limit(j{np.where(bad)[0].tolist()})")
        pos = self._linkpos(q, grip)
        for i, p in enumerate(pos):
            if i in self.base_links: continue
            if p[2] < -zmargin: reasons.append(f"below_table({self.links[i].name} z={p[2]:.3f})"); break
        if obj_center is not None:
            half = self.obj_half + omargin
            for i, p in enumerate(pos):
                if i in self.grip_links or i in self.base_links: continue
                if np.all(np.abs(p - np.asarray(obj_center)) < half):
                    reasons.append(f"arm_in_object({self.links[i].name})"); break
        return q, (len(reasons) == 0), reasons, perr

    def check_grasp(self, grasp_world, obj_center, approach=0.20, lift=0.22, n_seeds=10):
        """Feasible iff the pre-grasp, grasp, and lift EEF poses all pass for SOME IK seed. The FR3 is 7-DOF
        redundant, so a single seed can land in a joint-limit-violating branch while another branch is valid;
        we try n_seeds initial configs (rest + samples within joint limits) and accept the first full chain."""
        gp = np.asarray(grasp_world.p)
        wps = {"pre": sapien.Pose(p=gp + [0, 0, approach], q=grasp_world.q),
               "grasp": grasp_world,
               "lift": sapien.Pose(p=gp + [0, 0, lift], q=grasp_world.q)}
        rng = np.random.default_rng(0)
        q_rest = self.robot.get_qpos().cpu().numpy().reshape(-1)
        seeds = [q_rest] + [self._seed(q_rest, rng) for _ in range(n_seeds - 1)]
        best = None
        for seed in seeds:
            rep = {}; q0 = seed.copy(); ok_all = True
            for name in ("pre", "grasp", "lift"):
                oc = obj_center if name in ("pre", "grasp") else None   # object is lifted away after grasp
                q0, ok, reasons, perr = self.check_waypoint(wps[name], q0, obj_center=oc)
                rep[name] = {"ok": ok, "perr": round(perr, 4), "reasons": reasons}; ok_all &= ok
                if not ok: break
            if ok_all: return True, rep
            if best is None: best = rep
        return False, best

    def _seed(self, q0, rng):
        s = q0.copy(); s[:7] = self.qmin[:7] + rng.random(7) * (self.qmax[:7] - self.qmin[:7]); return s


def grasp_ee(xy, z_tcp):
    """EEF(2f85_base) pose for a top-down grasp whose TCP sits at world (xy, z_tcp)."""
    return sapien.Pose(p=[xy[0], xy[1], z_tcp + TCP_OFFSET], q=Q_DOWN)


if __name__ == "__main__":
    env = gym.make("CDWMScene-v0", obj_id="006_mustard_bottle", num_envs=1, render_mode="rgb_array",
                   sim_backend="physx_cpu", control_mode="pd_joint_pos")
    env.reset(seed=0)
    F = Feasibility(env)
    obj_c = [0.0, 0.0, OBJ_Z]                        # object rests at the origin
    cases = {
        "GOOD top-grasp @origin":       (grasp_ee([0.0, 0.0], OBJ_Z), obj_c),
        "BAD unreachable far (x=1.4)":  (grasp_ee([1.4, 0.0], OBJ_Z), [1.4, 0.0, OBJ_Z]),
        "BAD grasp below table":        (sapien.Pose(p=[0.0, 0.0, -0.05], q=Q_DOWN), obj_c),
        "BAD reach behind base (x=-0.9)": (grasp_ee([-0.9, 0.0], OBJ_Z), [-0.9, 0.0, OBJ_Z]),
    }
    print("=== feasibility filter ===", flush=True)
    for name, (gpose, oc) in cases.items():
        ok, rep = F.check_grasp(gpose, oc)
        print(f"\n[{'PASS' if ok else 'REJECT'}] {name}", flush=True)
        for wp, r in rep.items():
            print(f"    {wp:6s} ok={r['ok']} perr={r['perr']} {r['reasons']}", flush=True)
    env.close()
    print("\n=== done ===", flush=True)
