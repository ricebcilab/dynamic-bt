"""Skill: drive the installed utensil downward until tip clearance is safe."""

import numpy as np
from scipy.spatial.transform import Rotation as R

from .base_skill import BaseSkill, EPS


class Stab(BaseSkill):
    """Emit variable downward z velocity from env-provided tip clearance, optionally centering over the target first.

    Params: fast_clearance, slow_clearance,
            fast_z_speed, slow_z_speed, clearance_ref ("table" or
            "target_top": tracked tip vs the target's bbox top), stab_depth,
            floor_guard, xy_gain, xy_tolerance (target_top mode: steer the tip
            over the target's center, hold off the descent until within
            xy_tolerance, then descend stab_depth into its bbox).
    """

    def __init__(
        self,
        fast_clearance=0.05,
        slow_clearance=0.01,
        fast_z_speed=-0.05,
        slow_z_speed=-0.02,
        clearance_ref="table",
        stab_depth=0.0,
        floor_guard=0.005,
        xy_gain=3.0,
        xy_tolerance=None,
        hold_orientation=False,
        gain=1.0,
        max_angular_speed=1.0,
        **kwargs):

        super().__init__()

        self.clearance_ref = str(clearance_ref)
        self.stab_depth = float(stab_depth)
        self.floor_guard = float(floor_guard)
        self.xy_gain = float(xy_gain)
        self.xy_tolerance = None if xy_tolerance is None else float(xy_tolerance)
        # Regulate orientation toward the pose captured at skill start
        self.hold_orientation = bool(hold_orientation)
        self.gain = float(gain)
        self.max_angular_speed = float(max_angular_speed)
        self._hold_rot = None
        self.fast_clearance = float(fast_clearance)
        self.slow_clearance = float(slow_clearance)
        self.fast_z_speed = float(fast_z_speed)
        self.slow_z_speed = float(slow_z_speed)

    def get_action(self, task_state):
        action = np.zeros(9, dtype=np.float32)
        clearance = self._tip_clearance(task_state)
        # Steer the tip over the target's center; gate the descent on alignment
        centered = True
        if self.clearance_ref == "target_top":
            err = self._center_xy_error(task_state)
            if err is not None:
                action[:2] = np.clip(
                    self.xy_gain * err,
                    -abs(self.fast_z_speed), abs(self.fast_z_speed))
                if self.xy_tolerance is not None:
                    centered = float(np.linalg.norm(err)) <= self.xy_tolerance
        # Descend only once centered over the target
        if centered:
            if clearance > self.fast_clearance:
                action[2] = self.fast_z_speed
            elif clearance > self.slow_clearance:
                action[2] = self.slow_z_speed
        # Hold the orientation captured at skill start
        if self.hold_orientation and "eef_quat" in task_state:
            eef_rot = R.from_quat(task_state["eef_quat"])
            if self._hold_rot is None:
                self._hold_rot = eef_rot
            ang = self.gain * (self._hold_rot * eef_rot.inv()).as_rotvec()
            norm = np.linalg.norm(ang)
            if norm > EPS:
                ang *= min(1.0, self.max_angular_speed / norm)
            action[3:6] = ang
        return action

    def get_candidates(self, task_state):
        return {type(self).__name__: self.get_action(task_state)}

    def reset(self):
        super().reset()
        self._hold_rot = None

    def is_complete(self, task_state):
        if self._tip_clearance(task_state) > self.slow_clearance:
            return False
        if self.clearance_ref == "target_top" and self.xy_tolerance is not None:
            err = self._center_xy_error(task_state)
            if err is not None and float(np.linalg.norm(err)) > self.xy_tolerance:
                return False
        return True

    def _center_xy_error(self, task_state):
        tip = task_state.get("tip_pos")
        tgt = self.received_message.get("tgt_id", task_state.get("tgt_id"))
        objs = task_state.get("obj_pos", {})
        obj = objs.get(tgt, objs.get(str(tgt)))
        if tip is None or obj is None:
            return None
        return (np.asarray(obj, dtype=np.float64)[:2]
                - np.asarray(tip, dtype=np.float64)[:2])

    def _tip_clearance(self, task_state):
        if self.clearance_ref == "target_top":
            tip = task_state.get("tip_pos")
            tgt = self.received_message.get("tgt_id", task_state.get("tgt_id"))
            bboxes = task_state.get("obj_bbox", {})
            bbox = bboxes.get(tgt, bboxes.get(str(tgt)))
            if tip is None or bbox is None:
                return 0.0
            target_z = max(float(bbox[5]) - self.stab_depth,
                           float(bbox[4]) + self.floor_guard)
            return float(np.asarray(tip)[2]) - target_z
        return float(task_state["tip_clearance"])
