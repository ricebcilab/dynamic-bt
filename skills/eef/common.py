"""Shared helpers for EEF DynamicBT skills."""

from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from ..base_skill import EPS


KINOVA_WEBAPP_TO_INTERNAL = R.from_euler("z", -90.0, degrees=True)


def load_eef_configs(config_path=None):
    path = Path(config_path) if config_path else (
        Path(__file__).parent / "eef_configs.yaml"
    )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_bracket_config(configs, eef):
    cfg = configs.get(eef)
    if not cfg or cfg.get("is_bare"):
        return None
    return cfg.get("bracket", {})


def normalize(vec):
    arr = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(arr)
    return arr / norm if norm > EPS else arr


def with_z(pos, z):
    out = np.asarray(pos, dtype=np.float64).copy()
    out[2] = z
    return out


def clearance_z(bracket, positions, default_offset=0.08):
    min_z = max(np.asarray(pos, dtype=np.float64)[2] for pos in positions)
    if "clearance_z" in bracket:
        return max(float(bracket["clearance_z"]), min_z)
    offset = float(bracket.get("clearance_offset", default_offset))
    return min_z + offset


def scoop_angle_deg(task_state, default=270.0):
    if "scoop" in task_state:
        return float(task_state["scoop"])
    return float(task_state.get("fork_scoop", default))


def tool_tip_contact_eef_z(
    task_state,
    gripper_to_scoop_servo=0.13,
    scoop_servo_to_tip=0.167,
    table_z=0.0,
    z_calibration_offset=0.0,
    default_scoop_angle_deg=270.0,
):
    angle = scoop_angle_deg(task_state, default=default_scoop_angle_deg)
    return (
        float(table_z)
        + float(z_calibration_offset)
        + float(gripper_to_scoop_servo)
        + float(scoop_servo_to_tip) * np.sin(np.deg2rad(angle - 180.0))
    )


def tool_tip_clearance(
    task_state,
    gripper_to_scoop_servo=0.13,
    scoop_servo_to_tip=0.167,
    table_z=0.0,
    z_calibration_offset=0.0,
    default_scoop_angle_deg=270.0,
):
    eef_z = float(np.asarray(task_state["eef_pos"], dtype=np.float64)[2])
    contact_eef_z = tool_tip_contact_eef_z(
        task_state,
        gripper_to_scoop_servo=gripper_to_scoop_servo,
        scoop_servo_to_tip=scoop_servo_to_tip,
        table_z=table_z,
        z_calibration_offset=z_calibration_offset,
        default_scoop_angle_deg=default_scoop_angle_deg,
    )
    return eef_z - contact_eef_z


def rotation_from_config(cfg, default_rot=None, default_frame="internal"):
    cfg = cfg or {}
    if "rot_xyz" not in cfg:
        return default_rot if default_rot is not None else R.identity()

    rot = R.from_euler("xyz", cfg["rot_xyz"], degrees=True)
    rot_frame = cfg.get("rot_frame", default_frame)
    if rot_frame in {"kinova_webapp", "kortex_webapp"}:
        return rot * KINOVA_WEBAPP_TO_INTERNAL
    if rot_frame == "internal":
        return rot
    raise ValueError(f"Unsupported EEF pose rot_frame: {rot_frame}")


def pose_from_config(cfg, default_pos, default_rot, default_frame="internal"):
    cfg = cfg or {}
    pos = np.asarray(cfg.get("pos", default_pos), dtype=np.float64)
    rot = rotation_from_config(
        cfg, default_rot=default_rot, default_frame=default_frame)
    return pos, rot


def bracket_pose(bracket):
    pos = np.asarray(bracket.get("pos", [0.0, 0.0, 0.0]), dtype=np.float64)
    rot = rotation_from_config(bracket)
    slide_axis = normalize(bracket.get("slide_axis", [0.0, -1.0, 0.0]))
    slide_distance = float(bracket.get("slide_distance", 0.05))
    return pos, rot, slide_axis, slide_distance


def action9(twist=None, gripper_speed=0.0, twirl_cmd=0.0, scoop_cmd=0.0):
    action = np.zeros(9, dtype=np.float32)
    if twist is not None:
        action[:6] = np.asarray(twist, dtype=np.float32)[:6]
    action[6] = gripper_speed
    action[7] = twirl_cmd
    action[8] = scoop_cmd
    return action


def fit_action9(raw_action):
    raw = np.asarray(raw_action, dtype=np.float32)
    if raw.shape[0] == 9:
        return raw
    action = np.zeros(9, dtype=np.float32)
    n = min(raw.shape[0], 9)
    action[:n] = raw[:n]
    return action


def at_pose(eef_pos, eef_rot, target_pos, target_rot,
            pos_tolerance, rot_tolerance):
    pos_ok = np.linalg.norm(target_pos - eef_pos) <= pos_tolerance
    rot_ok = (target_rot * eef_rot.inv()).magnitude() <= rot_tolerance
    return pos_ok and rot_ok
