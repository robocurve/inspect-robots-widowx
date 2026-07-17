from __future__ import annotations

import numpy as np
import pytest

from inspect_robots_widowx import packing


def test_constants_and_labels() -> None:
    assert packing.POSE_DIM == 6
    assert packing.GRIPPER_IDX == 6
    assert packing.TOTAL_DIM == 7
    assert packing.STATE_KEY == "eef_pose"
    assert packing.DIM_LABELS == (
        "dx",
        "dy",
        "dz",
        "droll",
        "dpitch",
        "dyaw",
        "gripper",
    )
    assert len(set(packing.DIM_LABELS)) == packing.TOTAL_DIM


def test_validate_dim_accepts_list_and_returns_float64() -> None:
    out = packing.validate_dim(list(range(7)))
    assert np.array_equal(out, np.arange(7))
    assert out.dtype == np.float64


@pytest.mark.parametrize("bad", [np.zeros(6), np.zeros(8), np.zeros((1, 7))])
def test_validate_dim_rejects_wrong_shape(bad: np.ndarray) -> None:
    with pytest.raises(ValueError, match="expected a 7-D vector"):
        packing.validate_dim(bad)


def test_accessors_return_expected_values_and_displacement_copy() -> None:
    vec = np.arange(7, dtype=float)
    pose = packing.displacement(vec)
    assert np.array_equal(pose, np.arange(6))
    assert packing.gripper(vec) == 6.0
    pose[0] = 99.0
    assert vec[0] == 0.0


def test_reference_start_transform_is_byte_exact_and_substitutes_translation() -> None:
    fixture = np.asarray(
        [
            [0.267, 0.0, 0.963, 0.3],
            [0.0, 1.0, 0.0, -0.09],
            [-0.963, 0.0, 0.267, 0.26],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    assert np.array_equal(packing.START_TRANSFORM, fixture)
    built = packing.build_start_transform((0.31, -0.08, 0.25))
    assert np.array_equal(built[:3, :3], fixture[:3, :3])
    assert np.array_equal(built[:3, 3], np.asarray([0.31, -0.08, 0.25]))
    built[0, 0] = 99
    assert packing.START_TRANSFORM[0, 0] == 0.267


def test_start_transform_override_and_validation() -> None:
    override = np.arange(16, dtype=float)
    built = packing.build_start_transform((0.0, 0.0, 0.0), override)
    assert np.array_equal(built, override.reshape(4, 4))
    built[0, 0] = 99
    assert override[0] == 0
    with pytest.raises(ValueError, match="16 row-major"):
        packing.build_start_transform((0.0, 0.0, 0.0), np.zeros(15))
    with pytest.raises(ValueError, match="3 entries"):
        packing.build_start_transform((0.0, 0.0))
