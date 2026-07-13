"""Skill: transport an acquired tool bite to the mouth position.

Tool-only counterpart of the bare-gripper RetractObject: moves the utensil
target point to the mouth with APF, holding table clearance.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..base_skill import BaseSkill, EPS
from .common import tool_tip_clearance, tool_tip_pos


class ToolPresent(BaseSkill):
    """Transport the acquired bite on the utensil to the mouth with APF.

    Params: gain, max_linear_speed, max_angular_speed, safe_dist,
            repulsive_gain, orient_mode ("mouth" or "world_y"),
            tool_scoop_command, translate_only (present with pure XYZ
            translation), transport_clearance, hard_clearance, lift_z_speed,
            tool_mouth_tolerance, plus the geometric tip constants used when
            no tracked tip is available
    """

    def __init__(
        self,
        gain=1.0, max_linear_speed=0.3, max_angular_speed=1.0,
        safe_dist=0.03, repulsive_gain=0.5, orient_mode="world_y",
        tool_scoop_command=-1.0,
        gripper_to_scoop_servo=0.13, scoop_servo_to_tip=0.167,
        tool_tip_gripper_to_scoop_servo=0.115,
        scoop_to_twirl_servo=0.055, twirl_to_tool_tip=0.190,
        translate_only=False,
        table_z=0.0, z_calibration_offset=0.0,
        default_scoop_angle_deg=270.0,
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
        self.translate_only = bool(translate_only)
        self.tool_scoop_command = float(tool_scoop_command)
        self.gripper_to_scoop_servo = float(gripper_to_scoop_servo)
        self.scoop_servo_to_tip = float(scoop_servo_to_tip)
        self.tool_tip_gripper_to_scoop_servo = float(
            tool_tip_gripper_to_scoop_servo)
        self.scoop_to_twirl_servo = float(scoop_to_twirl_servo)
        self.twirl_to_tool_tip = float(twirl_to_tool_tip)
        self.table_z = float(table_z)
        self.z_calibration_offset = float(z_calibration_offset)
        self.default_scoop_angle_deg = float(default_scoop_angle_deg)
        self.hard_clearance = float(hard_clearance)
        self.transport_clearance = max(
            float(transport_clearance), self.hard_clearance)
        self.lift_z_speed = abs(float(lift_z_speed))
        self.tool_mouth_tolerance = float(tool_mouth_tolerance)

    def get_action(self, task_state):
        eef_rot = R.from_quat(task_state['eef_quat'])
        mouth_pos = task_state['mouth_pos']
        tip_pos = self._tool_tip_pos(task_state)
        clearance = self._tool_tip_clearance(task_state)

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
        if self.translate_only:
            # Fork/spoon: present with pure XYZ translation, no arm rotation (the
            # twirl/scoop servos handle utensil orientation). Zeroing the angular
            # twist stops the far tip from being swung around by wrist rotation.
            twist[3:6] = 0.0

        carried_bite_id = task_state.get("carried_bite_id")
        obstacles = [
            bbox for oid, bbox in task_state['obj_bbox'].items()
            if str(oid) != str(carried_bite_id)
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

    def get_candidates(self, task_state):
        default_key = task_state.get("carried_bite_id")
        if default_key is None:
            default_key = task_state.get("tgt_id", type(self).__name__)
        key = self.received_message.get("tgt_id", default_key)
        return {str(key): self.get_action(task_state)}

    def is_complete(self, task_state):
        mouth_pos = np.asarray(task_state["mouth_pos"], dtype=np.float64)
        tip_pos = self._tool_tip_pos(task_state)
        return (
            np.linalg.norm(tip_pos - mouth_pos)
            <= self.tool_mouth_tolerance
        )

    def _tool_tip_clearance(self, task_state):
        return tool_tip_clearance(
            task_state,
            gripper_to_scoop_servo=self.gripper_to_scoop_servo,
            scoop_servo_to_tip=self.scoop_servo_to_tip,
            table_z=self.table_z,
            z_calibration_offset=self.z_calibration_offset,
            default_scoop_angle_deg=self.default_scoop_angle_deg,
        )

    def _tool_tip_pos(self, task_state):
        # Prefer the env-provided tracked tip; fall back to the geometric
        # estimate when it is absent
        tip = task_state.get("tip_pos")
        if tip is not None:
            return np.asarray(tip, dtype=np.float64)
        return tool_tip_pos(
            task_state,
            gripper_to_scoop_servo=self.tool_tip_gripper_to_scoop_servo,
            scoop_to_twirl_servo=self.scoop_to_twirl_servo,
            twirl_to_tool_tip=self.twirl_to_tool_tip,
            default_scoop_angle_deg=self.default_scoop_angle_deg,
        )

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
