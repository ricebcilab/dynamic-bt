# AGENTS.md — dynamic-bt

Guidance for AI coding agents working in this repo. `dynamic-bt` provides the
behavior-tree skills consumed by `brand-rice`'s `feedingFSM` `DynamicBTAgent`.
It is editable-installed into the `robotics` conda env.

## ACTIVE SUGGESTION — surface before large edits to the approach/retract skills

**Before undertaking significant work on `skills/approach_and_align.py` or
`skills/retract_object.py`, surface the pending refactor below to the user and
confirm whether it should be folded into the current task. Do not start it
silently as part of unrelated work.**

### Split the tool-mode branches out of `ApproachAndAlign` and `RetractObject`

Both skills currently multiplex two end-effectors in one class via a runtime
`eef` check (`_is_installed_tool` / `_is_tool_mode` → `eef in {fork, spoon}`).
`RetractObject` is mostly tool code, with the bare-gripper path reduced to a
single method (`_get_gripper_action`). This couples the plain reach-3d gripper
expert — the eval baseline — to in-flight fork/spoon feeding work in the same
file.

Goal: make each skill single-purpose so the gripper baseline cannot be regressed
by tool-feeding changes, matching the existing `skills/eef/` convention (stab,
scoop, twirl, acquire, install already live there).

Recommended shape (from a prior brainstorm; not yet design-approved):
- Add `skills/eef/approach.py` → `ToolApproach(BaseSkill)` and
  `skills/eef/present.py` → `ToolPresent(BaseSkill)` (names TBD), reusing
  `skills/eef/common.py` tip geometry and `BaseSkill`'s twist/APF helpers.
- Strip the tool params/branches from `ApproachAndAlign` and `RetractObject`,
  leaving them gripper-only.
- Select the new classes by `class:` name in the BT config.

Migration surface is small: only `brand-rice`'s
`configs/dynamic_bt/kinova_reach_3d_tool_sa.yaml` drives the tool path today —
repoint its `ApproachAndAlign` / `RetractObject` `class:` fields to the new tool
skills. The 8 gripper BT configs (`omni_reach_3d_ol`, `_sa`, `free_use`,
`recovery`, `reach_2d`) set no tool params and are untouched.

Status: done — ToolApproach (skills/eef/approach.py) and ToolPresent
(skills/eef/present.py) landed; ApproachAndAlign and RetractObject are
gripper-only; both tool BT configs repointed.
