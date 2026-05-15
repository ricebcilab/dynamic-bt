"""Distance-below-threshold criterion."""

import numpy as np

from .base_criteria import AXIS_MAP, BaseCriteria


class Proximity(BaseCriteria):
    """Check if distance between two positions is below threshold."""

    def __init__(
        self,
        threshold: float,
        axes: str = "xyz",
        **kwargs
    ):
        super().__init__()
        self.threshold = threshold
        self.indices = AXIS_MAP.get(axes, [0, 1, 2])

    def _check_proximity(self, a, b):
        diff = np.asarray(a) - np.asarray(b)
        return float(np.linalg.norm(diff[self.indices])) < self.threshold

    def check(self, a, b, tgt_id=None) -> bool:
        if isinstance(b, dict):
            # One to one check
            if tgt_id is not None:
                b = b.get(tgt_id)
                if b is None:
                    return False
                if self._check_proximity(a, b):
                    self.message = {"tgt_id": tgt_id}
                    return True
                return False

            # One to many check
            for oid, pos in b.items():
                if self._check_proximity(a, pos):
                    self.message = {"tgt_id": oid}
                    return True
            return False

        else:
            return self._check_proximity(a, b)
