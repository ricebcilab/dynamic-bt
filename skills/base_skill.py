"""Base class for all skills in the dynamic BT framework."""

import numpy as np
from abc import ABC, abstractmethod

EPS = 1e-6


class BaseSkill(ABC):
    """Base class for dynamic BT skills.

    Subclasses receive YAML params as constructor kwargs and store them
    as instance attributes. ``criteria`` is assigned after construction
    by the graph loader.

    Attributes:
        WORLD_X/Y/Z: Unit axis constants shared across all skills.
    """

    WORLD_X = np.array([1.0, 0.0, 0.0])
    WORLD_Y = np.array([0.0, 1.0, 0.0])
    WORLD_Z = np.array([0.0, 0.0, 1.0])

    def __init__(self, **kwargs):
        self.criteria = []  # list of (criteria_obj, input_mapping)
        self.received_message = {}   # set by agent from previous skill's criteria

    def reset(self):
        """Reset internal state and all criteria."""
        for criteria, _ in self.criteria:
            criteria.reset()

    def is_complete(self, task_state: dict) -> bool:
        """True when all criteria are satisfied.

        Messages from earlier criteria are passed to subsequent ones,
        so they can constrain checks to the same object.
        """
        if not self.criteria:
            return False

        msg = {}
        for criteria, input_map in self.criteria:
            args = {}
            for input_name, source in input_map.items():
                args[input_name] = task_state.get(source)
            if not criteria.check(**{**msg, **args}):
                return False
            msg.update(criteria.message)
        return True

    @property
    def message(self):
        """Received message + criteria messages (criteria override received)."""
        msg = dict(self.received_message)
        for criteria, _ in self.criteria:
            msg.update(criteria.message)
        return msg

    @abstractmethod
    def get_action(self, task_state: dict) -> np.ndarray:
        """Return a single action for this frame."""

    def get_candidates(self, task_state: dict) \
            -> dict[str, np.ndarray] | None:
        """Return multiple candidate actions for SA blending, or None.

        Override in skills that produce multiple options (e.g.
        PlaceAndRelease returns one action per viable grid cell).
        """
        return None

    # ------------------------------------------------------------------
    # Twist computation
    # ------------------------------------------------------------------

    def _compute_twist(self, eef_pos, eef_rot, target_pos, target_rot):
        """6-DOF twist from configuration error, independently speed-clamped."""
        pos_error = target_pos - eef_pos
        rot_error = (target_rot * eef_rot.inv()).as_rotvec()

        twist = self.gain * np.concatenate([pos_error, rot_error])

        lin_norm = np.linalg.norm(twist[:3])
        ang_norm = np.linalg.norm(twist[3:6])
        if lin_norm > EPS:
            twist[:3] *= min(1.0, self.max_linear_speed / lin_norm)
        if ang_norm > EPS:
            twist[3:6] *= min(1.0, self.max_angular_speed / ang_norm)

        return twist

    # ------------------------------------------------------------------
    # Artificial Potential Field (collision avoidance)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_apf(eef_pos, obstacles_bbox, safe_dist, repulsive_gain):
        """APF repulsive force: EEF point vs obstacle AABBs."""
        force = np.zeros(3)
        for bbox in obstacles_bbox:
            closest = np.array([
                max(bbox[0], min(eef_pos[0], bbox[1])),
                max(bbox[2], min(eef_pos[1], bbox[3])),
                max(bbox[4], min(eef_pos[2], bbox[5])),
            ])
            direction = eef_pos - closest
            dist = max(np.linalg.norm(direction), EPS)
            if dist < safe_dist:
                direction = direction / dist
                force += direction * repulsive_gain \
                    * ((safe_dist - dist) / safe_dist) ** 2
        return force

    @staticmethod
    def _compute_apf_with_object(obj_bbox, obstacles_bbox, safe_dist,
                                  repulsive_gain):
        """APF repulsive force: grasped-object AABB vs obstacle AABBs."""
        force = np.zeros(3)
        if obj_bbox is None:
            return force
        for obs in obstacles_bbox:
            dx, dy, dz = 0.0, 0.0, 0.0
            if obj_bbox[1] < obs[0]:   dx = obj_bbox[1] - obs[0]
            elif obj_bbox[0] > obs[1]: dx = obj_bbox[0] - obs[1]
            if obj_bbox[3] < obs[2]:   dy = obj_bbox[3] - obs[2]
            elif obj_bbox[2] > obs[3]: dy = obj_bbox[2] - obs[3]
            if obj_bbox[5] < obs[4]:   dz = obj_bbox[5] - obs[4]
            elif obj_bbox[4] > obs[5]: dz = obj_bbox[4] - obs[5]

            direction = np.array([dx, dy, dz])
            dist = max(np.linalg.norm(direction), EPS)
            if dist < safe_dist:
                direction = direction / dist
                force += direction * repulsive_gain \
                    * ((safe_dist - dist) / safe_dist) ** 2
        return force

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _project_to_xy(vec):
        """Zero out Z, return (projected_vec, magnitude)."""
        projected = vec.copy()
        projected[2] = 0.0
        return projected, np.linalg.norm(projected)

    @staticmethod
    def _get_rotated_axis_in_xy(rot, ref):
        """Rotate ref by rot, project into XY, normalize."""
        axis = rot.apply(ref)
        axis, norm = BaseSkill._project_to_xy(axis)
        return axis / norm if norm > EPS else ref.copy()
