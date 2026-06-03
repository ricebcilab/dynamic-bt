"""Skill: dock the mounted EEF and return to bare gripper."""

import logging

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..base_skill import BaseSkill
from .common import (
    action9,
    at_pose,
    bracket_pose,
    clearance_z,
    get_bracket_config,
    load_eef_configs,
    pose_from_config,
    with_z,
)


class EEFUninstall(BaseSkill):
    """Line up with an EEF bracket, slide in, close gripper, and lift away."""

    def __init__(
        self,
        eef="auto",
        eef_config_path=None,
        gain=1.0,
        max_linear_speed=0.05,
        max_angular_speed=0.4,
        pos_tolerance=0.01,
        rot_tolerance=0.1,
        release_duration_s=0.5,
        close_duration_s=None,
        close_gripper_speed=-1.0,
        **kwargs):

        super().__init__()
        self.eef = eef
        self.eef_configs = load_eef_configs(eef_config_path)
        self.gain = gain
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.pos_tolerance = pos_tolerance
        self.rot_tolerance = rot_tolerance
        self.close_duration_s = (
            release_duration_s if close_duration_s is None
            else close_duration_s)
        self.close_gripper_speed = close_gripper_speed
        self.reset()

    def reset(self):
        super().reset()
        self._active_eef = None
        self._bracket = None
        self._phase = "raise"
        self._phase_start_ts = None
        self._done = False
        self._last_action_ts = None
        self._last_action = None
        self._last_advance_ts = None

    def get_action(self, task_state):
        ts = task_state.get("ts", 0)
        if self._last_action_ts == ts and self._last_action is not None:
            return self._last_action.copy()

        self._ensure_initialized(task_state)
        if self._done or self._bracket is None:
            action = np.zeros(9, dtype=np.float32)
            self._cache_action(ts, action)
            return action

        bracket_pos, bracket_rot, slide_axis, slide_distance = bracket_pose(
            self._bracket)
        lineup_pos = bracket_pos + slide_axis * slide_distance
        safe_z = clearance_z(self._bracket, [bracket_pos, lineup_pos])

        eef_pos = np.asarray(task_state["eef_pos"], dtype=np.float64)
        eef_rot = R.from_quat(task_state["eef_quat"])
        transit_z = max(float(eef_pos[2]), safe_z)
        raise_pos = with_z(eef_pos, transit_z)
        transit_pos = with_z(lineup_pos, transit_z)
        lineup_high_pos = with_z(lineup_pos, safe_z)
        bracket_high_pos = with_z(bracket_pos, safe_z)
        ready_pos, ready_rot = pose_from_config(
            self._bracket.get("uninstall_ready_pose")
            or self._bracket.get("uninstall_finish_pose")
            or self._bracket.get("install_ready_pose")
            or self._bracket.get("install_finish_pose"),
            bracket_high_pos,
            bracket_rot,
            default_frame=self._bracket.get("rot_frame", "internal"),
        )

        if self._phase == "raise":
            action = self._move_action(
                eef_pos, eef_rot, raise_pos, eef_rot, 0.0)
            if at_pose(eef_pos, eef_rot, raise_pos, eef_rot,
                       self.pos_tolerance, self.rot_tolerance):
                self._advance("transit", ts)

        elif self._phase == "transit":
            action = self._move_action(
                eef_pos, eef_rot, transit_pos, bracket_rot, 0.0)
            if at_pose(eef_pos, eef_rot, transit_pos, bracket_rot,
                       self.pos_tolerance, self.rot_tolerance):
                self._advance("descend_clearance", ts)

        elif self._phase == "descend_clearance":
            action = self._move_action(
                eef_pos, eef_rot, lineup_high_pos, bracket_rot, 0.0)
            if at_pose(eef_pos, eef_rot, lineup_high_pos, bracket_rot,
                       self.pos_tolerance, self.rot_tolerance):
                self._advance("descend", ts)

        elif self._phase == "descend":
            action = self._move_action(
                eef_pos, eef_rot, lineup_pos, bracket_rot, 0.0)
            if at_pose(eef_pos, eef_rot, lineup_pos, bracket_rot,
                       self.pos_tolerance, self.rot_tolerance):
                self._advance("slide_in", ts)

        elif self._phase == "slide_in":
            action = self._move_action(
                eef_pos, eef_rot, bracket_pos, bracket_rot, 0.0)
            if at_pose(eef_pos, eef_rot, bracket_pos, bracket_rot,
                       self.pos_tolerance, self.rot_tolerance):
                self._advance("close", ts)

        elif self._phase == "close":
            action = action9(gripper_speed=self.close_gripper_speed)
            if self._phase_start_ts is None:
                self._phase_start_ts = ts
            if (ts - self._phase_start_ts) >= self.close_duration_s * 1e9:
                self._mark_uninstalled(task_state)
                self._advance("lift", ts)

        elif self._phase == "lift":
            action = self._move_action(
                eef_pos, eef_rot, bracket_high_pos, bracket_rot, 0.0)
            if at_pose(eef_pos, eef_rot, bracket_high_pos, bracket_rot,
                       self.pos_tolerance, self.rot_tolerance):
                self._advance("ready", ts)

        elif self._phase == "ready":
            action = self._move_action(
                eef_pos, eef_rot, ready_pos, ready_rot, 0.0)
            if at_pose(eef_pos, eef_rot, ready_pos, ready_rot,
                       self.pos_tolerance, self.rot_tolerance):
                self._finish(task_state, ts)

        else:
            action = np.zeros(9, dtype=np.float32)

        self._cache_action(ts, action)
        return action

    def get_candidates(self, task_state):
        return {type(self).__name__: self.get_action(task_state)}

    def is_complete(self, task_state):
        return self._done

    def _ensure_initialized(self, task_state):
        if self._active_eef is not None:
            return

        self._active_eef = (
            task_state.get("eef", "gripper")
            if self.eef == "auto" else self.eef)
        if self._active_eef == "gripper":
            self._done = True
            return

        self._bracket = get_bracket_config(self.eef_configs, self._active_eef)
        if self._bracket is None:
            logging.warning("No EEF bracket config for '%s'", self._active_eef)
            self._done = True

    def _move_action(self, eef_pos, eef_rot, target_pos, target_rot,
                     gripper_speed):
        twist = self._compute_twist(eef_pos, eef_rot, target_pos, target_rot)
        return action9(twist=twist, gripper_speed=gripper_speed)

    def _advance(self, phase, ts):
        if ts == self._last_advance_ts:
            return
        self._last_advance_ts = ts
        self._phase = phase
        self._phase_start_ts = ts

    def _finish(self, task_state, ts):
        if ts == self._last_advance_ts:
            return
        self._last_advance_ts = ts
        self._mark_uninstalled(task_state)
        self._phase = "done"
        self._done = True
        logging.info("EEF '%s' uninstalled", self._active_eef)

    def _mark_uninstalled(self, task_state):
        task_state["eef"] = "gripper"
        task_state["carried_bite_id"] = None
        task_state["has_acquired_item"] = False

    def _cache_action(self, ts, action):
        self._last_action_ts = ts
        self._last_action = action.copy()
