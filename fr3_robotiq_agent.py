"""ManiSkill agent for the composed FR3 + Robotiq 2F-85 (assets/fr3_2f85.xml). Arm = 7 FR3 joints (PD joint-pos);
gripper = 2F-85 driven by the two driver joints via a mimic controller, with the coupler/spring/follower joints passive
(the 2F-85 linkage) -- mirrors ManiSkill's floating_robotiq_2f_85 agent, adapted to the Menagerie joint names.
Registered as uid 'fr3_robotiq'. Import this module before gym.make to register it.
"""
import os, numpy as np, sapien
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.registration import register_agent
from mani_skill.agents.controllers import (
    PDJointPosControllerConfig, PDJointPosMimicControllerConfig, PassiveControllerConfig,
)
from mani_skill.utils import common
from mani_skill.utils.structs.pose import Pose

MJCF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "fr3_2f85.xml")


@register_agent()
class FR3Robotiq(BaseAgent):
    uid = "fr3_robotiq"
    mjcf_path = MJCF
    arm_joint_names = [f"fr3_joint{i}" for i in range(1, 8)]
    gripper_driver_joints = ["2f85_right_driver_joint", "2f85_left_driver_joint"]
    gripper_passive_joints = [
        "2f85_right_coupler_joint", "2f85_right_spring_link_joint", "2f85_right_follower_joint",
        "2f85_left_coupler_joint", "2f85_left_spring_link_joint", "2f85_left_follower_joint",
    ]
    ee_link_name = "2f85_base"
    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853] + [0] * 8, dtype=np.float32),
            pose=sapien.Pose(),
        )
    )

    @property
    def _controller_configs(self):
        arm = PDJointPosControllerConfig(
            self.arm_joint_names, lower=None, upper=None, stiffness=1e3, damping=1e2,
            force_limit=100, normalize_action=False,
        )
        arm_delta = PDJointPosControllerConfig(
            self.arm_joint_names, lower=-0.1, upper=0.1, stiffness=1e3, damping=1e2,
            force_limit=100, use_delta=True,
        )
        # 2F-85 fingers have tiny inertia -> stiffness 1e3 is numerically UNSTABLE at the sim dt (explicit PD
        # explodes -> whole articulation NaN when the gripper closes). Soft, well-damped gains keep it stable.
        finger = PDJointPosMimicControllerConfig(
            self.gripper_driver_joints, lower=0.0, upper=0.8, stiffness=100, damping=20, force_limit=20,
        )
        # the passive four-bar linkage (coupler/spring/follower) needs real damping+friction to not oscillate.
        passive = PassiveControllerConfig(self.gripper_passive_joints, damping=5.0, friction=0.5)
        return dict(
            pd_joint_pos=dict(arm=arm, finger=finger, passive=passive),
            pd_joint_delta_pos=dict(arm=arm_delta, finger=finger, passive=passive),
        )
