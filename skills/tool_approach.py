"""Skill: move an installed utensil's tip to a pre-acquisition hover.

Tool-only counterpart of the bare-gripper ApproachAndAlign: aims the tracked
tool tip (not the wrist) at a hover point above the target's top face while
keeping the utensil orientation, with APF collision avoidance.
"""

import json
import numpy as np
from scipy.spatial.transform import Rotation as R

from .base_skill import BaseSkill


class ToolApproach(BaseSkill):
    """Move the tool tip to a hover above the target, with APF.

    Params: gain, max_linear_speed, max_angular_speed, approach_offset,
            safe_dist, repulsive_gain, tool_scoop_command,
            hold_orientation (regulate toward the orientation captured at
            skill start instead of re-anchoring to the drifting pose)
    """

    def __init__(
        self, obj_cfg_path=None,
        gain=1.0, max_linear_speed=0.3, max_angular_speed=1.0,
        approach_offset=0.03, safe_dist=0.10, repulsive_gain=0.5,
        tool_scoop_command=1.0,
        hold_orientation=False,
        **kwargs):

        super().__init__()

        self.gain = gain
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.approach_offset = approach_offset
        self.safe_dist = safe_dist
        self.repulsive_gain = repulsive_gain
        self.tool_scoop_command = float(tool_scoop_command)
        self.hold_orientation = bool(hold_orientation)
        self._hold_rot = None

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
            return np.zeros(9, dtype=np.float32)

        obj_pos = task_state['obj_pos'][tgt_id]
        obj_bbox = task_state['obj_bbox'][tgt_id]

        # Aim the tracked tool tip (not the wrist) at the hover point by
        # offsetting the EEF target with the live EEF-to-tip vector
        target_pos = obj_pos.copy()
        target_pos[2] = obj_bbox[5] + self.approach_offset
        target_pos = target_pos - (
            np.asarray(task_state["tip_pos"], dtype=np.float64)
            - np.asarray(eef_pos, dtype=np.float64))
        if self.hold_orientation:
            if self._hold_rot is None:
                self._hold_rot = eef_rot
            target_rot = self._hold_rot
        else:
            target_rot = eef_rot
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

        action = np.zeros(9, dtype=np.float32)
        action[:6] = twist
        action[8] = self.tool_scoop_command
        return action

    def reset(self):
        super().reset()
        self._hold_rot = None

    def is_complete(self, task_state):
        # Judge completion on the tool tip itself so the criteria fire when
        # the tip (not the wrist) is over the food
        filtered = task_state.copy()
        filtered['obj_bbox'] = {
            oid: bbox for oid, bbox in task_state['obj_bbox'].items()
            if self._is_compatible(oid, task_state)
        }
        filtered['eef_pos'] = self._tool_tip_pos(task_state)
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

    def _tool_tip_pos(self, task_state):
        return np.asarray(task_state["tip_pos"], dtype=np.float64)
