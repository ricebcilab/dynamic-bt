"""Skill: place a grasped object onto a safe empty spot on the table.

Divides a safe area into a grid, finds unoccupied cells, and moves the
EEF to the chosen cell before opening the gripper. Produces multiple
candidate actions (one per viable cell) for SA blending.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

from .base_skill import BaseSkill, EPS


class PlaceAndRelease(BaseSkill):
    """Move grasped object to an empty grid cell, then release.

    Params: gain, max_linear_speed, max_angular_speed, safe_area,
            table_z, grid_resolution, release_proximity
    """

    def __init__(
        self,
        gain=1.0, max_linear_speed=0.3, max_angular_speed=1.0,
        safe_area=(0, 0, 0.5, 0.5), grid_resolution=5,
        release_proximity=0.03,
        safe_dist=0.03, repulsive_gain=100.0,
        **kwargs):

        super().__init__()

        self.gain = gain
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.safe_area = safe_area
        self.grid_resolution = grid_resolution
        self.release_proximity = release_proximity
        self.safe_dist = safe_dist
        self.repulsive_gain = repulsive_gain
        
        self.table_z = None
        self.released = False

    def reset(self):
        super().reset()
        self.table_z = None
        self.released = False

    def get_action(self, task_state):
        """Move toward the nearest viable cell (non-SA mode)."""
        eef_pos = task_state['eef_pos']
        viable = self._get_viable_cells(task_state)
        if not viable:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        best = min(viable, key=lambda c: np.linalg.norm(c[:2] - eef_pos[:2]))
        return self._action_toward(best, task_state)

    def get_candidates(self, task_state):
        """One candidate action per viable grid cell (SA mode)."""
        viable = self._get_viable_cells(task_state)
        if not viable:
            return {"fallback": np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])}
        return {f"cell_{i}": self._action_toward(pos, task_state)
                for i, pos in enumerate(viable)}

    # ------------------------------------------------------------------
    # Grid logic
    # ------------------------------------------------------------------

    def _get_viable_cells(self, task_state):
        """Return list of [x, y, target_z] for unoccupied grid cells."""
        x_min, y_min, x_max, y_max = self.safe_area
        cell_w = (x_max - x_min) / self.grid_resolution
        cell_h = (y_max - y_min) / self.grid_resolution

        eef_pos = task_state['eef_pos']
        grasped_id = task_state.get('grasped_id')

        # Collect non-grasped objects' XY rects and z_mins in one pass
        obj_rects = []
        z_mins = []
        for oid, bbox in task_state['obj_bbox'].items():
            if oid != grasped_id:
                obj_rects.append((bbox[0], bbox[2], bbox[1], bbox[3]))
                z_mins.append(bbox[4])

        # Infer table top z once from the max of z_mins
        if self.table_z is None:
            self.table_z = max(z_mins) if z_mins else eef_pos[2]

        # Target EEF z: place object bottom at table surface
        obj_bbox = task_state['obj_bbox'].get(grasped_id)
        eef_to_obj_bottom = (eef_pos[2] - obj_bbox[4]) \
            if obj_bbox is not None else 0.0
        target_z = self.table_z + eef_to_obj_bottom

        viable = []
        for row in range(self.grid_resolution):
            for col in range(self.grid_resolution):
                c_xmin = x_min + col * cell_w
                c_xmax = c_xmin + cell_w
                c_ymin = y_min + row * cell_h
                c_ymax = c_ymin + cell_h

                # Check overlap (expanded by safe_dist for APF clearance)
                occupied = any(
                    c_xmin - self.safe_dist < ox_max
                    and c_xmax + self.safe_dist > ox_min
                    and c_ymin - self.safe_dist < oy_max
                    and c_ymax + self.safe_dist > oy_min
                    for ox_min, oy_min, ox_max, oy_max in obj_rects)

                if not occupied:
                    viable.append(np.array([
                        x_min + (col + 0.5) * cell_w,
                        y_min + (row + 0.5) * cell_h,
                        target_z]))
                    
        return viable

    # ------------------------------------------------------------------
    # Action computation
    # ------------------------------------------------------------------

    def _action_toward(self, target_pos, task_state):
        """Twist toward target, yaw toward +X, APF avoidance, open gripper when close."""
        eef_pos = task_state['eef_pos']
        eef_rot = R.from_quat(task_state['eef_quat'])
        grasped_id = task_state.get('grasped_id')

        target_rot = self._yaw_to_world_x(eef_rot)
        twist = self._compute_twist(eef_pos, eef_rot, target_pos, target_rot)

        # APF: grasped-object bbox vs obstacle bboxes
        obj_bbox = task_state['obj_bbox'].get(grasped_id)
        obstacles = [bbox for oid, bbox in task_state['obj_bbox'].items()
                     if oid != grasped_id]
        apf = self._compute_apf_with_object(
            obj_bbox, obstacles, self.safe_dist, self.repulsive_gain)

        # Vertical escape when APF strongly opposes the twist
        if (np.dot(twist[:3], apf)
                < -np.linalg.norm(twist[:3]) * np.linalg.norm(apf) * 0.8):
            twist[:3] += np.array(
                [0.0, 0.0, abs(np.dot(twist[:3], apf))])
        twist[:3] += apf

        # Latch: once released, hold still and keep gripper open
        if not self.released and np.linalg.norm(eef_pos - target_pos) < self.release_proximity:
            self.released = True
        if self.released:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        return np.concatenate([twist, [-1.0]])

    def _yaw_to_world_x(self, eef_rot):
        """Yaw EEF around world Z so its z-axis projects onto +X in XY."""
        eef_fwd_xy, norm_fwd = self._project_to_xy(
            eef_rot.apply(self.WORLD_Z))
        
        if norm_fwd < EPS:
            return eef_rot
        
        yaw_current = np.arctan2(eef_fwd_xy[1], eef_fwd_xy[0])
        delta_yaw = -yaw_current % (2 * np.pi)
        
        return R.from_rotvec(delta_yaw * self.WORLD_Z) * eef_rot
