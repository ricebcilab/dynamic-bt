"""Skill: twirl the installed utensil for acquisition."""

import time

import numpy as np

from ..base_skill import BaseSkill


class Twirl(BaseSkill):
    """Emit raw twirl velocity on action[7], then settle to a rest angle.

    Params: velocity_raw, duration_s, max_duration_s, hold_threshold,
    settle_velocity_raw, settle_tolerance_deg, settle_timeout_s, rest_angles.
    """

    def __init__(
        self,
        velocity_raw=150.0,
        duration_s=1.5,
        max_duration_s=5.0,
        hold_threshold=1e-3,
        settle_velocity_raw=60.0,
        settle_tolerance_deg=3.0,
        settle_timeout_s=3.0,
        rest_angles=None,
        **kwargs):

        super().__init__()

        self.velocity_raw = float(velocity_raw)
        self.duration_s = float(duration_s)
        self.max_duration_s = float(max_duration_s)
        self.hold_threshold = float(hold_threshold)
        self.settle_velocity_raw = abs(float(settle_velocity_raw))
        self.settle_tolerance_deg = float(settle_tolerance_deg)
        self.settle_timeout_s = float(settle_timeout_s)
        self.rest_angles = {
            "fork": 180.0,
            "spoon": 360.0,
        }
        if rest_angles is not None:
            self.rest_angles.update(rest_angles)

        self.start_time = None
        self.settle_start_time = None
        self.phase = "twirl"
        self.last_twirl_direction = self._direction_from(self.velocity_raw)

    def reset(self):
        super().reset()
        self.start_time = None
        self.settle_start_time = None
        self.phase = "twirl"
        self.last_twirl_direction = self._direction_from(self.velocity_raw)

    def get_action(self, task_state):
        if self.start_time is None:
            self.start_time = time.monotonic()

        action = np.zeros(9, dtype=np.float32)
        if self.phase == "settle":
            if not self._at_rest_angle(task_state):
                action[7] = (
                    self.last_twirl_direction * self.settle_velocity_raw)
            return action

        self._update_direction(task_state)
        action[7] = self.last_twirl_direction * abs(self.velocity_raw)
        return action

    def get_candidates(self, task_state):
        return {type(self).__name__: self.get_action(task_state)}

    def is_complete(self, task_state):
        if self.start_time is None:
            return False

        if self.phase == "settle":
            if self._at_rest_angle(task_state):
                return True
            if self.settle_start_time is None:
                self.settle_start_time = time.monotonic()
            return (
                time.monotonic() - self.settle_start_time
            ) >= self.settle_timeout_s

        elapsed = time.monotonic() - self.start_time
        operator_twirl = float(task_state.get("operator_twirl", 0.0))
        operator_active = abs(operator_twirl) > self.hold_threshold
        if operator_active:
            self.last_twirl_direction = self._direction_from(operator_twirl)

        if elapsed < self.duration_s and elapsed < self.max_duration_s:
            return False
        if operator_active and elapsed < self.max_duration_s:
            return False

        return self._start_settle_or_complete(task_state)

    def _start_settle_or_complete(self, task_state):
        if self._rest_angle(task_state) is None:
            return True
        if self._at_rest_angle(task_state):
            return True

        self.phase = "settle"
        self.settle_start_time = time.monotonic()
        return False

    def _update_direction(self, task_state):
        operator_twirl = float(task_state.get("operator_twirl", 0.0))
        if abs(operator_twirl) > self.hold_threshold:
            self.last_twirl_direction = self._direction_from(operator_twirl)

    def _rest_angle(self, task_state):
        eef = str(task_state.get("eef", ""))
        if eef not in self.rest_angles:
            return None
        return float(self.rest_angles[eef]) % 360.0

    def _twirl_angle(self, task_state):
        return float(task_state.get("twirl_angle", 0.0)) % 360.0

    def _at_rest_angle(self, task_state):
        target = self._rest_angle(task_state)
        if target is None:
            return True
        error = self._signed_angle_error(target, self._twirl_angle(task_state))
        return abs(error) <= self.settle_tolerance_deg

    @staticmethod
    def _direction_from(value):
        return -1.0 if float(value) < 0.0 else 1.0

    @staticmethod
    def _signed_angle_error(target, current):
        return ((float(target) - float(current) + 180.0) % 360.0) - 180.0
