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
