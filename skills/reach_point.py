"""Skill: translate the EEF toward a point in task_state, holding orientation."""

import numpy as np
from scipy.spatial.transform import Rotation as R

from .base_skill import BaseSkill


class ReachPoint(BaseSkill):
    """Drive the EEF to a task_state position with a fixed gripper command.

    Orientation is held, so the emitted twist is translation-only.

    Params: target_key, gain, max_linear_speed, gripper_action
    """

    def __init__(
        self,
        target_key="mouth_pos",
        gain=1.0, max_linear_speed=0.3, gripper_action=0.0,
        **kwargs):

        super().__init__()

        self.target_key = target_key
        self.gain = gain
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = 0.0
        self.gripper_action = float(gripper_action)

    def get_action(self, task_state):
        eef_pos = task_state['eef_pos']
        eef_rot = R.from_quat(task_state['eef_quat'])
        target_pos = np.asarray(task_state[self.target_key], dtype=float)

        twist = self._compute_twist(eef_pos, eef_rot, target_pos, eef_rot)

        return np.concatenate([twist, [self.gripper_action]])

    def get_candidates(self, task_state):
        return {type(self).__name__: self.get_action(task_state)}
