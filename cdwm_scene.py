"""ManiSkill scene for CDWM VLA demos: table + a CDWM object (textured mesh from the grasp dataset) + the FR3 + 2F-85
agent + a framed render camera. Registered as 'CDWMScene-v0'. Object pose is set kinematically (WM-driven replay comes
next); this file just stands up a well-framed scene to render."""
import os, numpy as np, sapien
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.types import SimConfig, SceneConfig
import fr3_robotiq_agent  # noqa: registers uid 'fr3_robotiq'

HF = "/misc/kcgscratch1/MengyeGroup/yy5259/datasets/cdwm-grasp-dataset"


@register_env("CDWMScene-v0", max_episode_steps=200)
class CDWMScene(BaseEnv):
    SUPPORTED_ROBOTS = ["fr3_robotiq"]

    def __init__(self, *args, obj_id="006_mustard_bottle", **kwargs):
        self.obj_id = obj_id
        super().__init__(*args, robot_uids="fr3_robotiq", **kwargs)

    @property
    def _default_sim_config(self):
        # fine timestep (dt=1/500) + extra solver iters so the stiff arm/gripper PD stays numerically stable
        # under env.step control (default sim_freq=100 explodes the light 2F-85 joints).
        return SimConfig(sim_freq=500, control_freq=50,
                         scene_config=SceneConfig(solver_position_iterations=20, solver_velocity_iterations=2))

    @property
    def _default_sensor_configs(self):
        return [CameraConfig("base_camera", sapien_utils.look_at([0.4, 0.4, 0.4], [0.0, 0.0, 0.1]), 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        # 3/4 front-right view, lower angle so the standing bottle reads as tall: FR3 (base at x=-0.5) arching to the object.
        # shader_pack="minimal" = plain rasterization (the default/rt shader drops the textured env materials to gray headless).
        return CameraConfig("render_camera", sapien_utils.look_at([0.95, -0.85, 0.5], [-0.1, 0.0, 0.12]), 1024, 1024, 1.0, 0.01, 100,
                            shader_pack="minimal")

    def _load_agent(self, options):
        super()._load_agent(options, sapien.Pose(p=[-0.5, 0.0, 0.0]))   # FR3 base behind the table edge

    # object AABB half-extents (proxy renders reliably; SAPIEN file-visual is buggy). Class default = mustard;
    # per-object value is computed from the mesh AABB in _load_scene and exposed on the instance as self.obj_half.
    OBJ_HALF = [0.048, 0.029, 0.096]

    def _obj_aabb_half(self):
        """AABB half-extents of the object mesh (trimesh). Box proxy is centered at the actor origin (= object
        center), so the settle/rest machinery treats the actor origin as the object center."""
        try:
            import trimesh
            m = trimesh.load(f"{HF}/objects/{self.obj_id}/mesh.obj", force="mesh", process=False)
            lo, hi = m.bounds
            return (0.5 * (hi - lo)).astype(float).tolist()
        except Exception as e:
            print("AABB fallback (mustard):", e, flush=True)
            return list(self.OBJ_HALF)

    def _load_scene(self, options):
        self.table = TableSceneBuilder(self); self.table.build()
        # reliable PRIMITIVE tabletop over the glb table (the textured glb intermittently drops its texture ->
        # renders flat gray during motion; a primitive with a RenderMaterial always renders). Top at z~0.
        tb = self.scene.create_actor_builder()
        wood = sapien.render.RenderMaterial(base_color=[0.62, 0.42, 0.24, 1.0], roughness=0.85, metallic=0.0)
        tb.add_box_visual(pose=sapien.Pose(p=[0.0, 0.0, -0.02]), half_size=[0.75, 0.6, 0.02], material=wood)
        self.tabletop = tb.build_kinematic(name="tabletop_proxy")
        self.obj_half = self._obj_aabb_half()
        b = self.scene.create_actor_builder()
        mat = sapien.render.RenderMaterial(base_color=[0.92, 0.78, 0.12, 1.0], roughness=0.6)   # yellow proxy
        b.add_box_visual(half_size=self.obj_half, material=mat)     # centered at the actor origin
        self.obj = b.build_kinematic(name="cdwm_object")            # kinematic: WM drives the pose (not sim physics)

    def _initialize_episode(self, env_idx, options):
        self.table.initialize(env_idx)
        self.obj.set_pose(sapien.Pose(p=[0.0, 0.0, self.obj_half[2]]))   # object center at half-height -> base on table
        self.agent.reset(self.agent.keyframes["rest"].qpos)
