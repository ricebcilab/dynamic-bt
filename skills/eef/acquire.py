"""Skill: run the configured EEF acquisition recipe."""

import json
import logging

import numpy as np

from ..base_skill import BaseSkill, SKILL_REGISTRY


class Acquire(BaseSkill):
    """Run the ordered acquisition recipe for the current target and EEF.

    The recipe is read from ``food_json[tgt_id]["acquisition"][eef]`` and is
    expected to contain DynamicBT skill class names such as ``Stab`` or
    ``DescendAndGrasp``. Bare-gripper configs without acquisition metadata
    fall back to ``DescendAndGrasp`` to preserve the original behavior.
    """

    def __init__(
        self,
        obj_cfg_path=None,
        default_gripper_recipe=("DescendAndGrasp",),
        attach_on=None,
        **kwargs,
    ):
        super().__init__()

        self._skill_kwargs = dict(kwargs)
        self._default_gripper_recipe = tuple(default_gripper_recipe)
        # Publish carried_bite_id as soon as this recipe skill completes
        # (e.g. "Stab": the bite is on the fork before Scoop/Twirl run)
        self._attach_on = None if attach_on is None else str(attach_on)

        if obj_cfg_path:
            with open(obj_cfg_path, "r", encoding="utf-8") as f:
                self.food_json = json.load(f)
        else:
            self.food_json = None

        self._sequence = []
        self._index = 0
        self._tgt_id = None
        self._eef = None
        self._acquired = False
        self._missing_recipe_warned = set()

    def reset(self):
        super().reset()
        self._sequence = []
        self._index = 0
        self._tgt_id = None
        self._eef = None
        self._acquired = False

    @property
    def message(self):
        msg = super().message
        if self._tgt_id is not None:
            msg["tgt_id"] = self._tgt_id
        if self._acquired and self._tgt_id is not None:
            msg["carried_bite_id"] = self._tgt_id
        return msg

    def get_action(self, task_state):
        self._ensure_sequence(task_state)
        if not self._has_active_skill():
            return np.zeros(9, dtype=np.float32)
        return self._sequence[self._index].get_action(task_state)

    def get_candidates(self, task_state):
        self._ensure_sequence(task_state)
        if not self._has_active_skill():
            return None
        return {type(self).__name__: self.get_action(task_state)}

    def is_complete(self, task_state):
        self._ensure_sequence(task_state)
        if not self._sequence:
            return False

        if self._index >= len(self._sequence):
            self._mark_acquired()
            return True

        skill = self._sequence[self._index]
        if not self._skill_complete(skill, task_state):
            return False

        if (self._attach_on is not None
                and type(skill).__name__ == self._attach_on):
            self._mark_acquired()
        self._index += 1
        if self._index < len(self._sequence):
            self._sequence[self._index].received_message = self.message.copy()
            return False

        self._mark_acquired()
        return True

    def _ensure_sequence(self, task_state):
        tgt_id = self._target_id(task_state)
        eef = str(task_state.get("eef", "gripper"))
        if tgt_id == self._tgt_id and eef == self._eef and self._sequence:
            return

        self._tgt_id = tgt_id
        self._eef = eef
        self._index = 0
        self._sequence = []

        recipe = self._recipe_for(tgt_id, eef)
        if not recipe:
            self._warn_missing_recipe(tgt_id, eef)
            return

        for skill_name in recipe:
            skill_cls = SKILL_REGISTRY.get(skill_name)
            if skill_cls is None:
                logging.error("Unknown acquisition skill '%s'", skill_name)
                self._sequence = []
                return

            skill = skill_cls(**self._skill_kwargs)
            skill.received_message = self.message.copy()
            if hasattr(skill, "food_json"):
                skill.food_json = self.food_json
            skill.reset()
            self._sequence.append(skill)

    def _recipe_for(self, tgt_id, eef):
        cfg = {}
        if self.food_json and tgt_id is not None:
            cfg = self.food_json.get(str(tgt_id), {})

        acquisition = cfg.get("acquisition", {}) if isinstance(cfg, dict) else {}
        recipe = acquisition.get(eef)
        if recipe is None and eef == "gripper":
            recipe = self._default_gripper_recipe

        if recipe is None:
            return []
        return [str(skill_name) for skill_name in recipe]

    def _target_id(self, task_state):
        tgt_id = self.received_message.get("tgt_id", task_state.get("tgt_id"))
        return None if tgt_id is None else str(tgt_id)

    def _has_active_skill(self):
        return bool(self._sequence) and self._index < len(self._sequence)

    def _skill_complete(self, skill, task_state):
        if type(skill).__name__ == "DescendAndGrasp":
            return bool(task_state.get("is_grasping", False))
        return bool(skill.is_complete(task_state))

    def _mark_acquired(self):
        if self._tgt_id is not None:
            self._acquired = True

    def _warn_missing_recipe(self, tgt_id, eef):
        key = (tgt_id, eef)
        if key in self._missing_recipe_warned:
            return
        self._missing_recipe_warned.add(key)
        logging.warning(
            "No acquisition recipe for target %s with EEF '%s'", tgt_id, eef)
