"""Skill: fly the EEF grasp point straight onto a target pose."""

import numpy as np
from scipy.spatial.transform import Rotation as R

from .base_skill import BaseSkill


class DirectGrasp(BaseSkill):
    """Converge the grasp midpoint on a task_state pose with a fixed gripper
    command. No hover, descend, or place waypoints - one straight unified
    twist toward the target.

    target_source "object" resolves the pose from obj_pos/obj_quat[tgt_id];
    "drop" resolves obj_pos/obj_quat["drop"]. tip_offset (EEF frame) shifts
    the convergence point from the EEF origin to the fingertip midpoint.

    Params: target_source, tip_offset, gain, max_linear_speed,
            max_angular_speed, gripper_action
    """

    def __init__(self, target_source="object", tip_offset=(0.0, 0.0, 0.0),
                 gain=1.0, max_linear_speed=0.3, max_angular_speed=1.0,
                 gripper_action=0.0, **kwargs):
        super().__init__()
        if target_source not in ("object", "drop"):
            raise ValueError(
                f"Unknown target_source '{target_source}' (object|drop)")
        self.target_source = target_source
        self.tip_offset = np.asarray(tip_offset, dtype=float)
        self.gain = gain
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.gripper_action = float(gripper_action)

    def get_action(self, task_state):
        eef_pos = np.asarray(task_state["eef_pos"], dtype=float)
        eef_rot = R.from_quat(task_state["eef_quat"])

        key = "drop" if self.target_source == "drop" else task_state["tgt_id"]
        target_pos = np.asarray(task_state["obj_pos"][key], dtype=float)
        target_rot = R.from_quat(
            np.asarray(task_state["obj_quat"][key], dtype=float))

        target_pos = target_pos - target_rot.apply(self.tip_offset)
        twist = self._compute_twist(eef_pos, eef_rot, target_pos, target_rot)
        return np.concatenate([twist, [self.gripper_action]])

    def get_candidates(self, task_state):
        return {type(self).__name__: self.get_action(task_state)}
