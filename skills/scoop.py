"""Skill: actuate the installed utensil scoop joint for acquisition."""

import time

import numpy as np

from .base_skill import BaseSkill


class Scoop(BaseSkill):
    """Emit scoop direction command on action[8].

    Params: direction, duration_s, max_duration_s, hold_threshold.
    """

    def __init__(
        self,
        direction=1.0,
        duration_s=1.5,
        max_duration_s=5.0,
        hold_threshold=1e-3,
        **kwargs):

        super().__init__()

        self.direction = float(direction)
        self.duration_s = float(duration_s)
        self.max_duration_s = float(max_duration_s)
        self.hold_threshold = float(hold_threshold)
        self.start_time = None

    def reset(self):
        super().reset()
        self.start_time = None

    def get_action(self, task_state):
        if self.start_time is None:
            self.start_time = time.monotonic()

        action = np.zeros(9, dtype=np.float32)
        action[8] = self.direction
        return action

    def get_candidates(self, task_state):
        return {type(self).__name__: self.get_action(task_state)}

    def is_complete(self, task_state):
        if self.start_time is None:
            return False

        elapsed = time.monotonic() - self.start_time
        if elapsed >= self.max_duration_s:
            return True
        if elapsed < self.duration_s:
            return False

        return abs(float(task_state.get("operator_scoop", 0.0))) <= self.hold_threshold
