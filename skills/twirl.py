"""Skill: twirl the installed utensil for acquisition."""

import logging
import time

import numpy as np

from .base_skill import BaseSkill


class Twirl(BaseSkill):
    """Emit raw twirl velocity on action[7], then settle to a rest angle.

    Params: velocity_raw, duration_s, duration_range ([min, max] seconds:
    resample duration_s uniformly on every reset, i.e. per trial), rounds /
    rounds_range ([min, max] inclusive ints: run for a number of full
    revolutions instead of a duration — needs a continuous twirl joint;
    completion and timeout scale with the sampled count), max_duration_s,
    hold_threshold, settle_velocity_raw, settle_tolerance_deg,
    settle_timeout_s, rest_angles.
    """

    def __init__(
        self,
        velocity_raw=150.0,
        duration_s=1.5,
        duration_range=None,
        rounds=None,
        rounds_range=None,
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
        self.duration_range = (
            None if duration_range is None
            else (float(duration_range[0]), float(duration_range[1])))
        self.rounds = None if rounds is None else int(rounds)
        self.rounds_range = (
            None if rounds_range is None
            else (int(rounds_range[0]), int(rounds_range[1])))
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
        self._last_angle = None
        self._accum_deg = 0.0
        self._sample_duration()
        self._sample_rounds()

    def reset(self):
        super().reset()
        self.start_time = None
        self.settle_start_time = None
        self.phase = "twirl"
        self.last_twirl_direction = self._direction_from(self.velocity_raw)
        self._last_angle = None
        self._accum_deg = 0.0
        self._sample_duration()
        self._sample_rounds()

    def _accumulate_rotation(self, task_state):
        # Integrate wrap-corrected per-step deltas: the continuous joint's
        # angle readback wraps (jumps by multiples of 360), so fold each delta
        # to (-180, 180] instead of differencing absolute angles
        cur = float(task_state.get("twirl_angle", 0.0))
        if self._last_angle is not None:
            d = (cur - self._last_angle + 180.0) % 360.0 - 180.0
            self._accum_deg += d
        self._last_angle = cur

    def _sample_duration(self):
        # Per-trial random twirl length (fresh instance/reset per trial)
        if self.duration_range is not None:
            self.duration_s = float(np.random.uniform(*self.duration_range))
            logging.info(f"Twirl: sampled duration {self.duration_s:.2f}s")

    def _sample_rounds(self):
        # Per-trial random number of full revolutions
        if self.rounds_range is not None:
            self.rounds = int(np.random.randint(
                self.rounds_range[0], self.rounds_range[1] + 1))
            logging.info(f"Twirl: sampled rounds {self.rounds}")

    def get_action(self, task_state):
        if self.start_time is None:
            self.start_time = time.monotonic()
        if self.rounds is not None:
            self._accumulate_rotation(task_state)

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

        # Rounds mode: complete on accumulated full revolutions (no settle;
        # stopping at N*360 deg is the rest orientation by construction)
        if self.rounds is not None:
            progress = self._accum_deg * self.last_twirl_direction
            if progress >= self.rounds * 360.0 - self.settle_tolerance_deg:
                return True
            return (time.monotonic() - self.start_time
                    ) >= self.max_duration_s * self.rounds

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
