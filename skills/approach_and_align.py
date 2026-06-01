"""Skill: move and orient EEF to a pre-grasp hover above the target object.

Computes a unified twist from the configuration error between the current
EEF pose and a target pose hovering above the object's top face in a
grasp-ready orientation. APF collision avoidance is integrated.
"""

import json
import numpy as np
from scipy.spatial.transform import Rotation as R

from .base_skill import BaseSkill


class ApproachAndAlign(BaseSkill):
    """Move + orient EEF to pre-grasp hover above target, with APF.

    Params: gain, max_linear_speed, max_angular_speed, approach_offset,
            safe_dist, repulsive_gain
    """

    def __init__(
        self, obj_cfg_path=None, 
        gain=1.0, max_linear_speed=0.3, max_angular_speed=1.0, 
        approach_offset=0.03, safe_dist=0.10, repulsive_gain=0.5, 
        **kwargs):
        
        super().__init__()

        self.gain = gain
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.approach_offset = approach_offset
        self.safe_dist = safe_dist
        self.repulsive_gain = repulsive_gain

        if obj_cfg_path:
            with open(obj_cfg_path, 'r') as f:
                self.food_json = json.load(f)
        else:
            self.food_json = None

    def get_action(self, task_state):
        eef_pos = task_state['eef_pos']
        eef_rot = R.from_quat(task_state['eef_quat'])
        tgt_id = task_state['tgt_id']
        if not self._is_compatible(tgt_id, task_state):
            return np.zeros(7)

        obj_pos = task_state['obj_pos'][tgt_id]
        obj_quat = task_state['obj_quat'][tgt_id]
        obj_bbox = task_state['obj_bbox'][tgt_id]

        target_pos = obj_pos.copy()
        target_pos[2] = obj_bbox[5] + self.approach_offset
        target_rot = self._grasp_orientation(eef_rot, obj_quat, tgt_id)
        twist = self._compute_twist(eef_pos, eef_rot, target_pos, target_rot)

        # APF collision avoidance (exclude target object)
        obstacles = [bbox for oid, bbox in task_state['obj_bbox'].items()
                     if oid != tgt_id]
        apf = self._compute_apf(
            eef_pos, obstacles, self.safe_dist, self.repulsive_gain)

        # Vertical escape when APF strongly opposes the twist
        if (np.dot(twist[:3], apf)
                < -np.linalg.norm(twist[:3]) * np.linalg.norm(apf) * 0.8):
            twist[:3] += np.array(
                [0.0, 0.0, abs(np.dot(twist[:3], apf))])
        twist[:3] += apf

        return np.concatenate([twist, [1.0]])  # gripper open

    def is_complete(self, task_state):
        filtered = task_state.copy()
        filtered['obj_bbox'] = {
            oid: bbox for oid, bbox in task_state['obj_bbox'].items()
            if self._is_compatible(oid, task_state)
        }
        return super().is_complete(filtered)

    def get_candidates(self, task_state):
        """One candidate per compatible object, keyed by object ID."""
        candidates = {}
        for oid in task_state['obj_pos'].keys():
            if not self._is_compatible(oid, task_state):
                continue
            assumed = task_state.copy()
            assumed['tgt_id'] = oid
            candidates[str(oid)] = self.get_action(assumed)
        return candidates if candidates else None

    def _is_compatible(self, oid, task_state):
        if not self.food_json:
            return True

        cfg = self.food_json.get(str(oid))
        if cfg is None:
            return True

        compatible_eefs = cfg.get('compatible_eefs')
        if compatible_eefs is None:
            return True

        return task_state.get('eef', 'gripper') in compatible_eefs

    # ------------------------------------------------------------------
    # Grasp orientation
    # ------------------------------------------------------------------

    def _grasp_orientation(self, eef_rot, obj_quat, tgt_id):
        """Target grasp rotation: z down, x aligned in XY, pitched."""
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
        """Build [x, z×x, z] frame, then pitch around local y."""
        y = np.cross(z, x)
        frame = R.from_matrix(np.column_stack((x, y, z)))
        return frame * R.from_rotvec(pitch_angle * np.array([0, 1, 0]))
