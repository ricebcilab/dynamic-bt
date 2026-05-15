"""Skill: descend to the nearest object and close gripper on arrival.

On activation, identifies the nearest object to the EEF and caches it.
Then drives downward to its centre while continuing to align to the
grasp orientation, and commands the gripper to close once within tolerance.
"""

import json
import numpy as np
from scipy.spatial.transform import Rotation as R

from .base_skill import BaseSkill


class DescendAndGrasp(BaseSkill):
    """Descend to nearest object, align orientation, close gripper.

    Params: gain, max_linear_speed, max_angular_speed, grasp_tolerance
    """

    def __init__(
        self,
        gain=1.0, max_linear_speed=0.2, max_angular_speed=0.5,
        grasp_tolerance=0.1, obj_cfg_path=None,
        **kwargs):

        super().__init__()

        self.gain = gain
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.grasp_tolerance = grasp_tolerance

        if obj_cfg_path:
            with open(obj_cfg_path, 'r') as f:
                self.food_json = json.load(f)
        else:
            self.food_json = None

        self.tgt_id = None

    def reset(self):
        super().reset()
        self.tgt_id = None

    def get_action(self, task_state):
        eef_pos = task_state['eef_pos']
        eef_rot = R.from_quat(task_state['eef_quat'])

        # Select nearest object on first frame
        if self.tgt_id is None:
            self.tgt_id = min(
                task_state['obj_pos'],
                key=lambda oid: np.linalg.norm(
                    task_state['obj_pos'][oid] - eef_pos),
                default=None)
        if self.tgt_id is None:
            return np.zeros(7)

        obj_pos = task_state['obj_pos'][self.tgt_id]
        obj_quat = task_state['obj_quat'][self.tgt_id]
        obj_bbox = task_state['obj_bbox'][self.tgt_id]

        # Continue aligning to grasp orientation while descending
        target_rot = self._grasp_orientation(eef_rot, obj_quat, self.tgt_id)
        twist = self._compute_twist(eef_pos, eef_rot, obj_pos, target_rot)

        # Close gripper when within tolerance
        bbox_size = np.array([
            obj_bbox[1] - obj_bbox[0],
            obj_bbox[3] - obj_bbox[2],
            obj_bbox[5] - obj_bbox[4],
        ])
        close = np.all(np.abs(obj_pos - eef_pos) <= self.grasp_tolerance * bbox_size)

        return np.concatenate([twist, [-1.0 if close else 0.0]])

    def get_candidates(self, task_state):
        action = self.get_action(task_state)
        if self.tgt_id is None:
            return None
        key = self.received_message.get("tgt_id", self.tgt_id)
        return {key: action}

    def _grasp_orientation(self, eef_rot, obj_quat, tgt_id):
        align = 0
        approach_angle = np.deg2rad(-45.0)
        if self.food_json and tgt_id in self.food_json:
            align = self.food_json[tgt_id].get('align', 0)
            approach_angle = np.deg2rad(
                self.food_json[tgt_id].get('approach_angle', -45.0))

        z_down = np.array([0.0, 0.0, -1.0])

        if align:
            obj_rot = R.from_quat(obj_quat)
            ref = np.array([1, 0, 0]) if align == 1 else np.array([0, 1, 0])
            long_axis = self._get_rotated_axis_in_xy(obj_rot, ref)
            candidates = [self._make_frame(d, z_down, approach_angle)
                          for d in (long_axis, -long_axis)]
            return min(candidates,
                       key=lambda r: (r * eef_rot.inv()).magnitude())
        else:
            y_dir = self._get_rotated_axis_in_xy(eef_rot, np.array([0, 1, 0]))
            x_dir = np.cross(y_dir, z_down)
            return self._make_frame(x_dir, z_down, approach_angle)

    @staticmethod
    def _make_frame(x, z, pitch_angle):
        y = np.cross(z, x)
        frame = R.from_matrix(np.column_stack((x, y, z)))
        return frame * R.from_rotvec(pitch_angle * np.array([0, 1, 0]))
