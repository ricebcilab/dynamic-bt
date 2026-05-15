"""Point-within-AABB-with-margin criterion."""

from .base_criteria import BaseCriteria


class Margin(BaseCriteria):
    """Check if a point is within expanded AABB(s) along specified axes.

    On success, self.message["tgt_id"] is set to the matched object ID.
    """

    def __init__(
        self,
        margin: float = 0.02,
        axes: str = "xy",
        **kwargs
    ):
        super().__init__()
        self.margin = margin
        axis_to_bbox = {"x": (0, 1), "y": (2, 3), "z": (4, 5)}
        self.bbox_pairs = [axis_to_bbox[a] for a in axes]
        self.eef_indices = [{"x": 0, "y": 1, "z": 2}[a] for a in axes]

    def _check_bbox(self, point, bbox):
        for (lo, hi), ei in zip(self.bbox_pairs, self.eef_indices):
            if not (bbox[lo] - self.margin <= point[ei] <= bbox[hi] + self.margin):
                return False
        return True

    def check(self, point, bbox, tgt_id=None) -> bool:
        if isinstance(bbox, dict):
            # One to one check
            if tgt_id is not None:
                bb = bbox.get(tgt_id)
                if bb is None:
                    return False
                if self._check_bbox(point, bb):
                    self.message = {"tgt_id": tgt_id}
                    return True
                return False

            # One to many check
            for oid, bb in bbox.items():
                if self._check_bbox(point, bb):
                    self.message = {"tgt_id": oid}
                    return True
            return False

        else:
            return self._check_bbox(point, bbox)
