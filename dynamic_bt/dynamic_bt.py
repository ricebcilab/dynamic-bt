"""DynamicBT: state-skill graph engine.

Parses a YAML graph and executes it. Pure compute — no Redis, no env.
Subclasses can override ``get_candidate_actions`` (or other public methods)
to side-effect on FSM activity, e.g. the feedingFSM-flavored ``DynamicBTAgent``
publishes to a Redis stream on each state change.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from . import criteria as _criteria
from . import skills as _skills


# ------------------------------------------------------------------
# Graph data classes (internal — DynamicBT owns the graph directly)
# ------------------------------------------------------------------

@dataclass
class _State:
    """A named state in the graph."""
    name: str
    description: str = ""
    invariant: str | None = None
    fallback: str | None = None
    idle_action: list[float] | None = None
    admin: bool = False

    def check_invariant(self, task_state: dict) -> bool:
        """True if the invariant holds or there is none."""
        if self.invariant is None:
            return True
        negated = self.invariant.startswith("!")
        key = self.invariant.lstrip("!")
        value = bool(task_state.get(key, False))
        return (not value) if negated else value


@dataclass
class _Edge:
    """A connection from one state to another, driven by a skill."""
    from_state: str
    to_state: str
    skill: object | None = field(default=None)
    pass_message: bool = True  # forward completed skill's message to next skill
    trigger: str | None = None


# ------------------------------------------------------------------
# DynamicBT
# ------------------------------------------------------------------

class DynamicBT:
    """State-skill graph engine driven by a YAML config.

    Parses the YAML in ``__init__`` and exposes ``get_action`` / ``step`` /
    ``reset`` for execution. Skills and criteria are fully instantiated at
    construction; skills communicate across transitions via a message dict.

    Args:
        dynamic_bt_path: Path to the dynamic BT YAML config.
        obj_cfg_path: Optional path to a per-object metadata JSON
            (forwarded to all skill constructors).
        **kwargs: Extra kwargs forwarded to all skill constructors.
    """

    def __init__(
        self,
        dynamic_bt_path: str,
        obj_cfg_path: str | None = None,
        **kwargs,
    ):
        skill_kwargs = dict(kwargs)
        if obj_cfg_path is not None:
            skill_kwargs["obj_cfg_path"] = obj_cfg_path

        self._states, self._edges, self._initial_state, \
            self._num_dof, self._completion_skill = \
            self._parse_yaml(dynamic_bt_path, skill_kwargs)

        self.current_state: str = self._initial_state
        if self._completion_skill is not None:
            self.is_complete: bool = False
        self.message = {}  # message from last completed skill

        self._warned_skill: set[str] = set()
        self._active_edges: list[_Edge] = []
        self._activate_edges()

        skill_names = [
            f"trigger:{e.trigger}" if e.trigger is not None
            else type(e.skill).__name__
            for e in self._edges
        ]
        logging.info(
            "DynamicBT initialized  state=%s  skills=%s",
            self.current_state, skill_names)

    # ------------------------------------------------------------------
    # YAML parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_yaml(path, skill_kwargs):
        with open(Path(path), "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        num_dof = raw.get("num_dof", 7)
        states = {name: _State(name=name, **(sdef or {}))
                  for name, sdef in raw.get("states", {}).items()}
        initial_state = raw.get("initial_state", next(iter(states)))
        completion_skill = raw.get("completion_skill", None)

        edges = []
        for edef in raw.get("edges", []):
            trigger = edef.get("trigger")
            has_skill = "skill" in edef
            if trigger is not None and has_skill:
                raise ValueError("Trigger edges are control-only and cannot define a skill")

            skill = None
            if has_skill:
                # Build skill instance
                skill_cfg = dict(edef.get("skill", {}))
                skill_cls = getattr(_skills, skill_cfg.pop("class"))
                skill = skill_cls(**skill_cfg, **skill_kwargs)

                # Build criteria list and attach to skill
                criteria_raw = edef.get("criteria", [])
                if isinstance(criteria_raw, dict):
                    criteria_raw = [criteria_raw]
                for cdef in criteria_raw:
                    cdef = dict(cdef)
                    criteria_cls = getattr(_criteria, cdef.pop("class"))
                    input_map = cdef.pop("inputs", {})
                    skill.criteria.append((criteria_cls(**cdef), input_map))
            elif trigger is None:
                raise ValueError("Edges must define either a skill or a trigger")

            if trigger is not None and "criteria" in edef:
                raise ValueError("Trigger edges are control-only and cannot define criteria")

            edges.append(_Edge(
                from_state=edef["from_state"],
                to_state=edef["to_state"],
                skill=skill,
                pass_message=edef.get("pass_message", True),
                trigger=trigger,
            ))

        # Validate
        all_names = set(states)
        for e in edges:
            assert e.from_state in all_names, f"Unknown from_state '{e.from_state}'"
            assert e.to_state in all_names, f"Unknown to_state '{e.to_state}'"
        for s in states.values():
            if s.fallback:
                assert s.fallback in all_names, f"Unknown fallback '{s.fallback}'"
        assert initial_state in all_names, f"Unknown initial_state '{initial_state}'"

        return states, edges, initial_state, num_dof, completion_skill

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def num_dof(self) -> int:
        return self._num_dof

    @property
    def state(self) -> str:
        """Current state name."""
        return self.current_state

    def get_state(self):
        """Return the current state name (legacy alias for ``self.state``)."""
        return self.current_state

    def is_in_admin_state(self) -> bool:
        """True when the current state is marked as an admin state."""
        return bool(self._states[self.current_state].admin)

    def handle_event(self, event_name: str) -> bool:
        """Apply a control-only event transition from the current state.

        Returns True when an edge consumed the event and changed state.
        """
        for edge in self._edges:
            if edge.from_state != self.current_state or edge.trigger != event_name:
                continue

            prev = self.current_state
            self.message = {}
            self.current_state = edge.to_state
            self._activate_edges()
            logging.info(
                "Event transition: %s -[%s]-> %s",
                prev, event_name, self.current_state)
            return True

        return False

    def get_action(self, task_state) -> np.ndarray:
        """Return a single action (num_dof) for the first active edge's skill."""
        state_def = self._states[self.current_state]
        idle = state_def.idle_action

        if not state_def.check_invariant(task_state) or not self._active_edges:
            if idle:
                return self._fit_action(np.array(idle), skill_name="idle")
            return np.zeros(self.num_dof)

        skill = self._active_edges[0].skill
        try:
            raw_action = skill.get_action(task_state)
        except KeyError:
            logging.warning("Failed to get action from active skill")
            return np.zeros(self.num_dof)

        return self._fit_action(raw_action, skill_name=type(skill).__name__)

    def get_candidate_actions(self, task_state) -> dict[str, np.ndarray]:
        """Return all candidate actions for SA blending.

        Each skill provides its own candidates via ``get_candidates()``.
        The current state's ``idle_action`` (if any) is included as ``"idle"``.
        """
        state_def = self._states[self.current_state]
        if not state_def.check_invariant(task_state):
            return {}

        candidates = {}
        if state_def.idle_action is not None:
            candidates["idle"] = self._fit_action(
                np.array(state_def.idle_action), skill_name="idle")

        for edge in self._active_edges:
            skill_candidates = edge.skill.get_candidates(task_state)
            if skill_candidates is None:
                continue
            skill_name = type(edge.skill).__name__
            for name, action in skill_candidates.items():
                candidates[name] = self._fit_action(action, skill_name=skill_name)

        return candidates

    def step(self, task_state):
        """Advance the state machine. Call once per frame after action is taken.

        Edge completions are checked before invariants so that a skill can
        complete even when its condition overlaps with invariant violation.
        """
        for edge in self._active_edges:
            if edge.skill.is_complete(task_state):
                # Skip self-loop transitions (e.g., hold edges)
                if edge.to_state == self.current_state:
                    continue

                prev = self.current_state
                skill_name = type(edge.skill).__name__

                # Capture message from completed skill
                if edge.pass_message:
                    self.message = edge.skill.message.copy()
                else:
                    self.message = {}

                self.current_state = edge.to_state
                self._activate_edges()

                if self._completion_skill and skill_name == self._completion_skill:
                    self.is_complete = True

                logging.info(
                    "Transition: %s -[%s]-> %s (message=%s)",
                    prev, skill_name, self.current_state, self.message)
                return

        # Check state invariant (fallback on violation)
        state_def = self._states[self.current_state]
        if not state_def.check_invariant(task_state):
            logging.warning(
                "Invariant violated in '%s', fallback to '%s'",
                self.current_state, state_def.fallback)
            self.current_state = state_def.fallback
            self._activate_edges()
            return

    def reset(self, *args, **kwargs):
        """Reset to initial state and reactivate edges."""
        self.current_state = self._initial_state
        if self._completion_skill is not None:
            self.is_complete = False
        self.message = {}
        self._activate_edges()

    def update_scene_info(self, food_json):
        """Push dynamic object config to all skills that support it."""
        for edge in self._edges:
            if hasattr(edge.skill, "food_json"):
                edge.skill.food_json = food_json

    def cleanup(self):
        pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _activate_edges(self):
        """Load outgoing edges for current state, reset skills, inject message."""
        self._active_edges = [
            e for e in self._edges
            if e.from_state == self.current_state and e.trigger is None
        ]
        for edge in self._active_edges:
            edge.skill.received_message = self.message.copy()
            edge.skill.reset()

    def _fit_action(self, raw_action: np.ndarray, skill_name: str) -> np.ndarray:
        """Fit raw_action to self.num_dof by padding with zeros or truncating.

        Logs at most one warning per skill_name over the agent's lifetime.
        """
        n = self.num_dof
        k = raw_action.shape[0]
        if k == n:
            return raw_action

        if skill_name not in self._warned_skill:
            self._warned_skill.add(skill_name)
            verb = "padded" if k < n else "truncated"
            logging.warning("Action %s for skill '%s': %d -> %d", verb, skill_name, k, n)

        if k < n:
            out = np.zeros(n)
            out[:k] = raw_action
            return out
        return raw_action[:n]
