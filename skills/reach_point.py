"""Skill: translate the EEF toward a point in task_state, holding orientation."""

import numpy as np
from scipy.spatial.transform import Rotation as R

from .base_skill import BaseSkill


class ReachPoint(BaseSkill):
    """Drive the EEF to a task_state position while holding its orientation.

    The held orientation is anchored on the first frame after activation, so
    the orientation loop keeps a restoring term. With grasp_radius set, the
    gripper switches to grasp_action inside that radius on frames where
    grasp_required holds; None keeps gripper_action throughout.
    """

    def __init__(
        self,
        target_key="mouth_pos",
        gain=1.0, max_linear_speed=0.3, max_angular_speed=0.3,
        gripper_action=0.0,
        grasp_radius=None, grasp_action=-1.0,
        **kwargs):

        super().__init__()

        self.target_key = target_key
        self.gain = gain
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.gripper_action = float(gripper_action)
        self.grasp_radius = None if grasp_radius is None else float(grasp_radius)
        self.grasp_action = float(grasp_action)

        self.hold_rot = None

    def reset(self):
        super().reset()
        self.hold_rot = None

    def get_action(self, task_state):
        eef_pos = task_state['eef_pos']
        eef_rot = R.from_quat(task_state['eef_quat'])
        target_pos = np.asarray(task_state[self.target_key], dtype=float)

        # Anchor the held orientation on the first frame after activation
        if self.hold_rot is None:
            self.hold_rot = eef_rot

        twist = self._compute_twist(eef_pos, eef_rot, target_pos, self.hold_rot)

        # Grasp on approach: flip the gripper inside grasp_radius when required
        gripper = self.gripper_action
        if (self.grasp_radius is not None
                and task_state.get('grasp_required', True)
                and float(np.linalg.norm(
                    np.asarray(eef_pos, dtype=float) - target_pos)) < self.grasp_radius):
            gripper = self.grasp_action

        return np.concatenate([twist, [gripper]])

    def get_candidates(self, task_state):
        return {type(self).__name__: self.get_action(task_state)}
