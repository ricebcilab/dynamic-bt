"""Skill: move EEF upward to disengage from the selected object.

Used in SA mode to allow the user to abandon the current object
selection and return to the approach phase.
"""

import numpy as np

from .base_skill import BaseSkill


class Retreat(BaseSkill):
    """Move EEF straight up at linear_speed.

    Params: linear_speed
    """

    def __init__(
        self,
        linear_speed=0.3,
        **kwargs):

        super().__init__()

        self.linear_speed = linear_speed

    def get_action(self, task_state):
        return np.array([0.0, 0.0, self.linear_speed, 0.0, 0.0, 0.0, 0.0])

    def get_candidates(self, task_state):
        return {type(self).__name__: self.get_action(task_state)}
