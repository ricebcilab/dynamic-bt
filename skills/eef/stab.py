"""Skill: drive the installed utensil downward until tip clearance is safe."""

import numpy as np

from ..base_skill import BaseSkill


class Stab(BaseSkill):
    """Emit variable downward z velocity based on tool-tip table clearance.

    Clearance comes from the environment as task_state["tip_clearance"].

    Params: fast_clearance, slow_clearance, fast_z_speed, slow_z_speed.
    """

    def __init__(
        self,
        fast_clearance=0.05,
        slow_clearance=0.01,
        fast_z_speed=-0.05,
        slow_z_speed=-0.02,
        **kwargs):

        super().__init__()

        self.fast_clearance = float(fast_clearance)
        self.slow_clearance = float(slow_clearance)
        self.fast_z_speed = float(fast_z_speed)
        self.slow_z_speed = float(slow_z_speed)

    def get_action(self, task_state):
        action = np.zeros(9, dtype=np.float32)
        clearance = float(task_state["tip_clearance"])
        if clearance > self.fast_clearance:
            action[2] = self.fast_z_speed
        elif clearance > self.slow_clearance:
            action[2] = self.slow_z_speed
        return action

    def get_candidates(self, task_state):
        return {type(self).__name__: self.get_action(task_state)}

    def is_complete(self, task_state):
        return float(task_state["tip_clearance"]) <= self.slow_clearance
