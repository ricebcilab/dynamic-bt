"""Skill: drive the installed utensil downward until tip clearance is safe."""

import numpy as np

from ..base_skill import BaseSkill
from .common import tool_tip_clearance


class Stab(BaseSkill):
    """Emit variable downward z velocity based on tool-tip table clearance.

    Params: gripper_to_scoop_servo, scoop_servo_to_tip, table_z,
            z_calibration_offset, fast_clearance, slow_clearance,
            fast_z_speed, slow_z_speed.
    """

    def __init__(
        self,
        gripper_to_scoop_servo=0.13,
        scoop_servo_to_tip=0.167,
        table_z=0.0,
        z_calibration_offset=0.0,
        default_scoop_angle_deg=270.0,
        fast_clearance=0.05,
        slow_clearance=0.01,
        fast_z_speed=-0.05,
        slow_z_speed=-0.02,
        **kwargs):

        super().__init__()

        self.gripper_to_scoop_servo = float(gripper_to_scoop_servo)
        self.scoop_servo_to_tip = float(scoop_servo_to_tip)
        self.table_z = float(table_z)
        self.z_calibration_offset = float(z_calibration_offset)
        self.default_scoop_angle_deg = float(default_scoop_angle_deg)
        self.fast_clearance = float(fast_clearance)
        self.slow_clearance = float(slow_clearance)
        self.fast_z_speed = float(fast_z_speed)
        self.slow_z_speed = float(slow_z_speed)

    def get_action(self, task_state):
        action = np.zeros(9, dtype=np.float32)
        clearance = self._tip_clearance(task_state)
        if clearance > self.fast_clearance:
            action[2] = self.fast_z_speed
        elif clearance > self.slow_clearance:
            action[2] = self.slow_z_speed
        return action

    def get_candidates(self, task_state):
        return {type(self).__name__: self.get_action(task_state)}

    def is_complete(self, task_state):
        return self._tip_clearance(task_state) <= self.slow_clearance

    def _tip_clearance(self, task_state):
        return tool_tip_clearance(
            task_state,
            gripper_to_scoop_servo=self.gripper_to_scoop_servo,
            scoop_servo_to_tip=self.scoop_servo_to_tip,
            table_z=self.table_z,
            z_calibration_offset=self.z_calibration_offset,
            default_scoop_angle_deg=self.default_scoop_angle_deg,
        )
