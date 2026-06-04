"""Skill: install a requested EEF, uninstalling the current EEF if needed."""

import logging

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..base_skill import BaseSkill
from .common import (
    action9,
    at_pose,
    bracket_pose,
    clearance_z,
    fit_action9,
    get_bracket_config,
    load_eef_configs,
    pose_from_config,
    with_z,
)
from .uninstall import EEFUninstall


class EEFInstall(BaseSkill):
    """Compose uninstall-if-needed plus bracket pickup for target EEF."""

    def __init__(
        self,
        eef,
        eef_config_path=None,
        gain=1.0,
        max_linear_speed=0.05,
        max_angular_speed=0.4,
        pos_tolerance=0.01,
        rot_tolerance=0.1,
        grip_duration_s=0.5,
        close_duration_s=None,
        open_duration_s=None,
        lift_distance=0.08,
        close_gripper_speed=-1.0,
        open_gripper_speed=1.0,
        **kwargs):

        super().__init__()
        self.target_eef = eef
        self.motion_kwargs = {
            "eef_config_path": eef_config_path,
            "gain": gain,
            "max_linear_speed": max_linear_speed,
            "max_angular_speed": max_angular_speed,
            "pos_tolerance": pos_tolerance,
            "rot_tolerance": rot_tolerance,
        }
        self.install_kwargs = {
            **self.motion_kwargs,
            "close_duration_s": (
                grip_duration_s if close_duration_s is None
                else close_duration_s),
            "open_duration_s": (
                grip_duration_s if open_duration_s is None
                else open_duration_s),
            "lift_distance": lift_distance,
            "close_gripper_speed": close_gripper_speed,
            "open_gripper_speed": open_gripper_speed,
        }
        self.reset()

    def reset(self):
        super().reset()
        self._sequence = None
        self._index = 0
        self._done = False
        self._last_advance_ts = None
        self._last_action_ts = None
        self._last_action = None

    def get_action(self, task_state):
        ts = task_state.get("ts", 0)
        if self._last_action_ts == ts and self._last_action is not None:
            return self._last_action.copy()

        self._maybe_build_sequence(task_state)
        if self._done or not self._sequence or self._index >= len(self._sequence):
            self._done = True
            action = np.zeros(9, dtype=np.float32)
            self._cache_action(ts, action)
            return action

        skill = self._sequence[self._index]
        action = fit_action9(skill.get_action(task_state))
        if skill.is_complete(task_state) and ts != self._last_advance_ts:
            self._last_advance_ts = ts
            self._index += 1
            if self._index < len(self._sequence):
                self._sequence[self._index].reset()
            else:
                self._done = True

        self._cache_action(ts, action)
        return action

    def get_candidates(self, task_state):
        return {type(self).__name__: self.get_action(task_state)}

    def is_complete(self, task_state):
        self._maybe_build_sequence(task_state)
        return self._done

    def _maybe_build_sequence(self, task_state):
        if self._sequence is not None:
            return

        current_eef = task_state.get("eef", "gripper")
        if current_eef == self.target_eef:
            self._sequence = []
            self._done = True
            return

        install_motion = _EEFInstallMotion(
            eef=self.target_eef, **self.install_kwargs)
        if current_eef == "gripper":
            self._sequence = [install_motion]
        else:
            self._sequence = [
                EEFUninstall(eef="auto", **self.motion_kwargs),
                install_motion,
            ]

        for skill in self._sequence:
            skill.reset()

    def _cache_action(self, ts, action):
        self._last_action_ts = ts
        self._last_action = action.copy()


class _EEFInstallMotion(BaseSkill):
    """Close gripper, dock into bracket, open inside tool, and slide out."""

    def __init__(
        self,
        eef,
        eef_config_path=None,
        gain=1.0,
        max_linear_speed=0.05,
        max_angular_speed=0.4,
        pos_tolerance=0.01,
        rot_tolerance=0.1,
        close_duration_s=0.5,
        open_duration_s=0.5,
        lift_distance=0.08,
        close_gripper_speed=-1.0,
        open_gripper_speed=1.0,
        **kwargs):

        super().__init__()
        self.eef = eef
        self.eef_configs = load_eef_configs(eef_config_path)
        self.gain = gain
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.pos_tolerance = pos_tolerance
        self.rot_tolerance = rot_tolerance
        self.close_duration_s = close_duration_s
        self.open_duration_s = open_duration_s
        self.lift_distance = lift_distance
        self.close_gripper_speed = close_gripper_speed
        self.open_gripper_speed = open_gripper_speed
        self.reset()

    def reset(self):
        super().reset()
        self._bracket = None
        self._phase = "close"
        self._phase_start_ts = None
        self._done = False
        self._last_advance_ts = None

    def get_action(self, task_state):
        self._ensure_initialized()
        if self._done or self._bracket is None:
            return np.zeros(9, dtype=np.float32)

        ts = task_state.get("ts", 0)
        bracket_pos, bracket_rot, slide_axis, slide_distance = bracket_pose(
            self._bracket)
        lineup_pos = bracket_pos + slide_axis * slide_distance
        nominal_lift_pos = bracket_pos + np.array([0.0, 0.0, self.lift_distance])
        safe_z = clearance_z(
            self._bracket, [bracket_pos, lineup_pos, nominal_lift_pos],
            default_offset=self.lift_distance)

        eef_pos = np.asarray(task_state["eef_pos"], dtype=np.float64)
        eef_rot = R.from_quat(task_state["eef_quat"])
        transit_z = max(float(eef_pos[2]), safe_z)
        raise_pos = with_z(eef_pos, transit_z)
        transit_pos = with_z(bracket_pos, transit_z)
        bracket_high_pos = with_z(bracket_pos, safe_z)
        ready_pos, ready_rot = pose_from_config(
            self._bracket.get("install_ready_pose")
            or self._bracket.get("install_finish_pose"),
            with_z(lineup_pos, safe_z),
            bracket_rot,
            default_frame=self._bracket.get("rot_frame", "internal"),
        )

        if self._phase == "close":
            action = action9(gripper_speed=self.close_gripper_speed)
            if self._phase_start_ts is None:
                self._phase_start_ts = ts
            if (ts - self._phase_start_ts) >= self.close_duration_s * 1e9:
                self._advance("raise", ts)

        elif self._phase == "raise":
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
                eef_pos, eef_rot, bracket_high_pos, bracket_rot, 0.0)
            if at_pose(eef_pos, eef_rot, bracket_high_pos, bracket_rot,
                       self.pos_tolerance, self.rot_tolerance):
                self._advance("descend", ts)

        elif self._phase == "descend":
            action = self._move_action(
                eef_pos, eef_rot, bracket_pos, bracket_rot, 0.0)
            if at_pose(eef_pos, eef_rot, bracket_pos, bracket_rot,
                       self.pos_tolerance, self.rot_tolerance):
                self._advance("open", ts)

        elif self._phase == "open":
            action = action9(gripper_speed=self.open_gripper_speed)
            if self._phase_start_ts is None:
                self._phase_start_ts = ts
            if (ts - self._phase_start_ts) >= self.open_duration_s * 1e9:
                self._advance("slide_out", ts)

        elif self._phase == "slide_out":
            action = self._move_action(
                eef_pos, eef_rot, lineup_pos, bracket_rot, 0.0)
            if at_pose(eef_pos, eef_rot, lineup_pos, bracket_rot,
                       self.pos_tolerance, self.rot_tolerance):
                self._advance("lift_out", ts)

        elif self._phase == "lift_out":
            action = self._move_action(
                eef_pos, eef_rot, with_z(lineup_pos, safe_z), bracket_rot, 0.0)
            if at_pose(eef_pos, eef_rot, with_z(lineup_pos, safe_z), bracket_rot,
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

        return action

    def is_complete(self, task_state):
        return self._done

    def _ensure_initialized(self):
        if self._bracket is not None or self._done:
            return
        self._bracket = get_bracket_config(self.eef_configs, self.eef)
        if self._bracket is None:
            logging.warning("No EEF bracket config for '%s'", self.eef)
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
        task_state["eef"] = self.eef
        self._phase = "done"
        self._done = True
        logging.info("EEF '%s' installed", self.eef)
