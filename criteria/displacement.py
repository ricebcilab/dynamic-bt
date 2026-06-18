"""Moved-beyond-threshold criterion."""

import numpy as np

from .base_criteria import AXIS_MAP, BaseCriteria


class Displacement(BaseCriteria):
    """Check if position has moved beyond threshold since activation.

    Origin is captured from the first call after reset().
    """

    def __init__(
        self,
        threshold: float,
        axes: str = "xyz",
        **kwargs
    ):
        super().__init__()
        self.threshold = threshold
        self.indices = AXIS_MAP.get(axes, [0, 1, 2])
        self._origin = None

    def reset(self):
        super().reset()
        self._origin = None

    def check(self, current) -> bool:
        current = np.asarray(current)
        if self._origin is None:
            self._origin = current.copy()
        diff = current - self._origin
        return float(np.linalg.norm(diff[self.indices])) > self.threshold


class SignedDisplacement(BaseCriteria):
    """Check signed movement along one axis from the activation origin.

    ``direction="positive"`` requires current - origin > threshold.
    ``direction="negative"`` requires origin - current > threshold.
    """

    def __init__(
        self,
        threshold: float,
        axis: str = "z",
        direction: str = "positive",
        **kwargs
    ):
        super().__init__()
        self.threshold = float(threshold)
        self.index = AXIS_MAP.get(axis, [2])[0]
        self.sign = self._direction_sign(direction)
        self._origin = None

    def reset(self):
        super().reset()
        self._origin = None

    def check(self, current) -> bool:
        current = np.asarray(current)
        if self._origin is None:
            self._origin = current.copy()
        displacement = float(current[self.index] - self._origin[self.index])
        return self.sign * displacement > self.threshold

    @staticmethod
    def _direction_sign(direction):
        if isinstance(direction, (int, float)):
            return 1.0 if direction >= 0 else -1.0

        direction = str(direction).lower()
        if direction in {"positive", "pos", "+", "up"}:
            return 1.0
        if direction in {"negative", "neg", "-", "down"}:
            return -1.0
        raise ValueError(
            "SignedDisplacement direction must be positive or negative")
