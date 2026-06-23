"""Moved-beyond-threshold criterion."""

import numpy as np

from .base_criteria import AXIS_MAP, BaseCriteria


class Displacement(BaseCriteria):
    """Check if position has moved beyond threshold since activation.

    Origin is captured from the first call after reset(). Without a
    ``direction``, the Euclidean norm of the displacement over ``axes`` is
    compared against the threshold. With ``direction`` ("positive"/"negative"),
    the signed displacement along the single ``axes`` axis is used instead.
    """

    def __init__(
        self,
        threshold: float,
        axes: str = "xyz",
        direction: str = None,
        **kwargs
    ):
        super().__init__()
        self.threshold = float(threshold)
        self.indices = AXIS_MAP.get(axes, [0, 1, 2])
        self.sign = self._direction_sign(direction)
        self._origin = None

    def reset(self):
        super().reset()
        self._origin = None

    def check(self, current) -> bool:
        current = np.asarray(current)
        if self._origin is None:
            self._origin = current.copy()
        diff = current[self.indices] - self._origin[self.indices]
        if self.sign is None:
            return float(np.linalg.norm(diff)) > self.threshold
        return self.sign * float(diff[0]) > self.threshold

    @staticmethod
    def _direction_sign(direction):
        if direction is None:
            return None
        if isinstance(direction, (int, float)):
            return 1.0 if direction >= 0 else -1.0

        direction = str(direction).lower()
        if direction in {"positive", "pos", "+"}:
            return 1.0
        if direction in {"negative", "neg", "-"}:
            return -1.0
        raise ValueError(
            "Displacement direction must be positive or negative")
