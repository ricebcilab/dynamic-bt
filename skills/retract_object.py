"""Skill: transport an acquired object or tool bite to the mouth position.

Bare-gripper mode preserves the original grasped-object behavior. Tool mode
for fork/spoon moves the estimated utensil target point to the mouth.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

from .base_skill import BaseSkill, EPS


class RetractObject(BaseSkill):
    """Transport the acquired item to mouth with APF.

    Tool geometry comes from the environment as task_state["tip_pos"] and
    task_state["tip_clearance"].

    Params: gain, max_linear_speed, max_angular_speed, safe_dist,
            repulsive_gain, orient_mode ("mouth" or "world_y"), tool_eefs,
            transport_clearance, hard_clearance, lift_z_speed
    """

    def __init__(
        self,
        gain=1.0, max_linear_speed=0.3, max_angular_speed=1.0,
        safe_dist=0.03, repulsive_gain=0.5, orient_mode="world_y",
        tool_eefs=("fork", "spoon"), tool_scoop_command=-1.0,
        transport_clearance=0.05, hard_clearance=0.01,
        lift_z_speed=0.05, tool_mouth_tolerance=0.05,
        **kwargs):

        super().__init__()

        self.gain = gain
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.safe_dist = safe_dist
        self.repulsive_gain = repulsive_gain
        self.orient_mode = orient_mode
        if isinstance(tool_eefs, str):
            tool_eefs = (tool_eefs,)
        self.tool_eefs = {str(eef) for eef in tool_eefs}
        self.tool_scoop_command = float(tool_scoop_command)
        self.hard_clearance = float(hard_clearance)
        self.transport_clearance = max(
            float(transport_clearance), self.hard_clearance)
        self.lift_z_speed = abs(float(lift_z_speed))
        self.tool_mouth_tolerance = float(tool_mouth_tolerance)

    def get_action(self, task_state):
        if self._is_tool_mode(task_state):
            return self._get_tool_action(task_state)
        return self._get_gripper_action(task_state)

    def get_candidates(self, task_state):
        action = self.get_action(task_state)
        default_key = task_state.get("acquired_id")
        if default_key is None:
            default_key = task_state.get("tgt_id", type(self).__name__)
        key = self.received_message.get("tgt_id", default_key)
        return {str(key): action}

    def is_complete(self, task_state):
        if not self._is_tool_mode(task_state):
            return super().is_complete(task_state)

        mouth_pos = np.asarray(task_state["mouth_pos"], dtype=np.float64)
        tip_pos = np.asarray(task_state["tip_pos"], dtype=np.float64)
        return (
            np.linalg.norm(tip_pos - mouth_pos)
            <= self.tool_mouth_tolerance
        )

    def _is_tool_mode(self, task_state):
        return task_state.get("eef", "gripper") in self.tool_eefs

    def _get_gripper_action(self, task_state):
        eef_pos = task_state['eef_pos']
        eef_rot = R.from_quat(task_state['eef_quat'])
        mouth_pos = task_state['mouth_pos']
        self.tgt_id = task_state.get('acquired_id')

        if self.orient_mode == "mouth":
            target_rot = self._yaw_toward(eef_rot, mouth_pos - eef_pos)
        else:
            target_rot = self._yaw_toward(eef_rot, self.WORLD_Y)

        twist = self._compute_twist(eef_pos, eef_rot, mouth_pos, target_rot)

        # APF: grasped-object bbox vs obstacle bboxes
        obj_bbox = task_state['obj_bbox'].get(self.tgt_id)
        obstacles = [bbox for oid, bbox in task_state['obj_bbox'].items()
                     if oid != self.tgt_id]
        apf = self._compute_apf_with_object(
            obj_bbox, obstacles, self.safe_dist, self.repulsive_gain)

        # Vertical escape when APF strongly opposes the twist
        if (np.dot(twist[:3], apf)
                < -np.linalg.norm(twist[:3]) * np.linalg.norm(apf) * 0.8):
            twist[:3] += np.array(
                [0.0, 0.0, abs(np.dot(twist[:3], apf))])
        twist[:3] += apf

        return np.concatenate([twist, [-1.0]])  # keep gripper closed

    def _get_tool_action(self, task_state):
        eef_rot = R.from_quat(task_state['eef_quat'])
        mouth_pos = task_state['mouth_pos']
        tip_pos = np.asarray(task_state['tip_pos'], dtype=np.float64)
        clearance = float(task_state['tip_clearance'])

        if clearance < self.transport_clearance:
            action = np.zeros(9, dtype=np.float32)
            action[2] = self.lift_z_speed
            action[8] = self.tool_scoop_command
            return action

        if self.orient_mode == "mouth":
            target_rot = self._yaw_toward(eef_rot, mouth_pos - tip_pos)
        else:
            target_rot = self._yaw_toward(eef_rot, self.WORLD_Y)

        twist = self._compute_twist(tip_pos, eef_rot, mouth_pos, target_rot)

        acquired_id = task_state.get("acquired_id")
        obstacles = [
            bbox for oid, bbox in task_state['obj_bbox'].items()
            if str(oid) != str(acquired_id)
        ]
        apf = self._compute_apf(
            tip_pos, obstacles, self.safe_dist, self.repulsive_gain)
        twist[:3] += apf
        if clearance <= self.hard_clearance and twist[2] < 0.0:
            twist[2] = 0.0

        action = np.zeros(9, dtype=np.float32)
        action[:6] = twist
        action[8] = self.tool_scoop_command
        return action

    def _yaw_toward(self, eef_rot, direction):
        """Yaw EEF around world Z so its z-axis projects toward direction in XY."""
        eef_fwd_xy, norm_fwd = self._project_to_xy(
            eef_rot.apply(self.WORLD_Z))
        dir_xy, norm_dir = self._project_to_xy(direction)

        if norm_fwd < EPS or norm_dir < EPS:
            return eef_rot

        yaw_target = np.arctan2(dir_xy[1], dir_xy[0])
        yaw_current = np.arctan2(eef_fwd_xy[1], eef_fwd_xy[0])
        delta_yaw = (yaw_target - yaw_current) % (2 * np.pi)

        return R.from_rotvec(delta_yaw * self.WORLD_Z) * eef_rot
