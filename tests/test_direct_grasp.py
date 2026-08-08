"""Unit tests for DirectGrasp. Run directly:
/home/user/miniconda3/envs/robotics/bin/python lib/dynamic-bt/tests/test_direct_grasp.py
"""
import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from skills.direct_grasp import DirectGrasp
from skills.base_skill import SKILL_REGISTRY

TGT = "r1_d2_a3_b4"
DROP = "0"


def state(obj_pos, obj_quat, drop_pos=None, drop_quat=None):
    op = {TGT: np.asarray(obj_pos, dtype=float)}
    oq = {TGT: np.asarray(obj_quat, dtype=float)}
    if drop_pos is not None:
        op[DROP] = np.asarray(drop_pos, dtype=float)
        oq[DROP] = np.asarray(drop_quat, dtype=float)
    return {"eef_pos": np.zeros(3),
            "eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "tgt_id": TGT, "obj_pos": op, "obj_quat": oq}


def test_registered():
    assert "DirectGrasp" in SKILL_REGISTRY


def test_translation_toward_object():
    s = DirectGrasp(gain=1.0, max_linear_speed=1.0, max_angular_speed=1.0,
                    gripper_action=1.0)
    a = s.get_action(state([0.2, 0.0, 0.0], [0, 0, 0, 1]))
    assert a.shape == (7,)
    assert a[0] > 0 and abs(a[1]) < 1e-9 and abs(a[2]) < 1e-9
    assert np.allclose(a[3:6], 0.0), "identity target quat must give zero rotation"
    assert a[6] == 1.0


def test_rotation_toward_target_quat():
    s = DirectGrasp(gain=1.0, max_linear_speed=1.0, max_angular_speed=1.0,
                    gripper_action=-1.0)
    q = R.from_rotvec([0.0, 0.0, np.pi / 6]).as_quat()
    a = s.get_action(state([0.0, 0.0, 0.0], q))
    assert a[5] > 0.1, "should rotate about +z toward the target quat"
    assert a[6] == -1.0


def test_drop_target_source():
    s = DirectGrasp(target_source="drop", gain=1.0,
                    max_linear_speed=1.0, max_angular_speed=1.0,
                    gripper_action=-1.0)
    a = s.get_action(state([0.5, 0, 0], [0, 0, 0, 1],
                           drop_pos=[0.0, 0.3, 0.0], drop_quat=[0, 0, 0, 1]))
    assert a[1] > 0 and abs(a[0]) < 1e-9, "must head to drop, not object"


def test_tip_offset():
    s = DirectGrasp(tip_offset=(0.0, 0.0, 0.1), gain=1.0,
                    max_linear_speed=10.0, max_angular_speed=1.0,
                    gripper_action=1.0)
    a = s.get_action(state([0.0, 0.0, 0.0], [0, 0, 0, 1]))
    # Target moves to obj - R@offset = (0,0,-0.1): twist points -z
    assert a[2] < 0


def test_speed_clamp():
    s = DirectGrasp(gain=100.0, max_linear_speed=0.5, max_angular_speed=1.0,
                    gripper_action=1.0)
    a = s.get_action(state([5.0, 0.0, 0.0], [0, 0, 0, 1]))
    assert abs(np.linalg.norm(a[:3]) - 0.5) < 1e-6


def test_candidates():
    s = DirectGrasp(gain=1.0, max_linear_speed=1.0, max_angular_speed=1.0,
                    gripper_action=1.0)
    c = s.get_candidates(state([0.2, 0, 0], [0, 0, 0, 1]))
    assert list(c.keys()) == ["DirectGrasp"]


if __name__ == "__main__":
    for name, fn in sorted({k: v for k, v in globals().items()
                            if k.startswith("test_")}.items()):
        fn()
        print(f"PASS {name}")
    print("OK")
