"""Skill: deliberately miss the grasp, then re-open and grasp correctly.

For miss-recovery data collection. Descends to a randomized aim-off point
above the object's top face, commits a close on nothing, re-opens the
gripper while lifting clear, then descends to the true grasp point and
closes as normal. Occasionally injects a second miss before the true
attempt. Miss offset, committed-close duration, reopen duration, and lift
speed are re-sampled per activation (and per retry) so recovery does not
become a fixed temporal schedule. If the object attaches at any point,
behaves like DescendAndGrasp from then on.
"""

import numpy as np

from .descend_and_grasp import DescendAndGrasp


class DescendAndGraspRecovery(DescendAndGrasp):
    """DescendAndGrasp with randomized injected misses + recovery.

    Params: miss_height_min/max (metres above the object's top face for the
    miss close), miss_lateral_max, miss_close_frames_min/max,
    reopen_frames_min/max, reopen_lift_speed_min/max, double_miss_prob,
    plus all DescendAndGrasp params.
    """

    def __init__(
        self, miss_height_min=0.04, miss_height_max=0.07,
        miss_lateral_max=0.02,
        miss_close_frames_min=8, miss_close_frames_max=30,
        reopen_frames_min=15, reopen_frames_max=35,
        reopen_lift_speed_min=0.10, reopen_lift_speed_max=0.20,
        double_miss_prob=0.3,
        **kwargs):

        super().__init__(**kwargs)

        self.miss_height_min = float(miss_height_min)
        self.miss_height_max = float(miss_height_max)
        self.miss_lateral_max = float(miss_lateral_max)
        self.miss_close_frames_min = int(miss_close_frames_min)
        self.miss_close_frames_max = int(miss_close_frames_max)
        self.reopen_frames_min = int(reopen_frames_min)
        self.reopen_frames_max = int(reopen_frames_max)
        self.reopen_lift_speed_min = float(reopen_lift_speed_min)
        self.reopen_lift_speed_max = float(reopen_lift_speed_max)
        self.double_miss_prob = float(double_miss_prob)

        self._phase = "miss"
        self._misses_left = 0
        self._miss_height = 0.0
        self._miss_lateral = np.zeros(2)
        self._close_frames = 0
        self._reopen_frames = 0
        self._lift_speed = 0.0
        self._close_count = 0
        self._reopen_count = 0

    def reset(self):
        super().reset()
        self._phase = "miss"
        self._misses_left = (
            2 if np.random.uniform() < self.double_miss_prob else 1)
        self._sample_miss()

    def _sample_miss(self):
        self._miss_height = np.random.uniform(
            self.miss_height_min, self.miss_height_max)
        r = np.random.uniform(0.0, self.miss_lateral_max)
        theta = np.random.uniform(0.0, 2 * np.pi)
        self._miss_lateral = np.array([r * np.cos(theta), r * np.sin(theta)])
        self._close_frames = np.random.randint(
            self.miss_close_frames_min, self.miss_close_frames_max + 1)
        self._reopen_frames = np.random.randint(
            self.reopen_frames_min, self.reopen_frames_max + 1)
        self._lift_speed = np.random.uniform(
            self.reopen_lift_speed_min, self.reopen_lift_speed_max)
        self._close_count = 0
        self._reopen_count = 0

    def get_action(self, task_state):
        if task_state.get("is_grasping"):
            self._phase = "done"
            action = super().get_action(task_state)
            action[6] = -1.0
            return action

        if self._phase == "reopen":
            self._reopen_count += 1
            if self._reopen_count >= self._reopen_frames:
                if self._misses_left > 0:
                    self._phase = "miss"
                    self._sample_miss()
                else:
                    self._phase = "regrasp"
            return np.array(
                [0.0, 0.0, self._lift_speed, 0.0, 0.0, 0.0, 1.0])

        state = task_state
        if (self._phase == "miss" and self.tgt_id is not None
                and self.tgt_id in task_state["obj_pos"]):
            pos = np.asarray(
                task_state["obj_pos"][self.tgt_id], dtype=float)
            bbox = task_state["obj_bbox"][self.tgt_id]
            state = dict(task_state)
            state["obj_pos"] = dict(task_state["obj_pos"])
            state["obj_pos"][self.tgt_id] = pos + np.array([
                self._miss_lateral[0],
                self._miss_lateral[1],
                (float(bbox[5]) - pos[2]) + self._miss_height,
            ])

        action = super().get_action(state)

        if self._phase == "miss" and action[6] < 0.0:
            self._close_count += 1
            if self._close_count >= self._close_frames:
                self._misses_left -= 1
                self._phase = "reopen"

        return action
