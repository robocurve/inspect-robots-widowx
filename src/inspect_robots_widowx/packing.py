"""Canonical 7-D BridgeData packing for WidowX end-effector actions.

The shared vector is ``[dx, dy, dz, droll, dpitch, dyaw, gripper]``. The first
six slots are per-step displacements. The final slot is an absolute normalized
target with 0 closed and 1 open. This module is pure NumPy.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

POSE_DIM = 6
GRIPPER_IDX = 6
TOTAL_DIM = 7
DIM_LABELS = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")
STATE_KEY = "eef_pose"

# OpenVLA's reference WidowX wrapper uses this matrix directly. Its controller
# composes rotations relative to an internal default, so this rotation must not
# be reconstructed from a quaternion.
START_TRANSFORM: npt.NDArray[np.float64] = np.asarray(
    [
        [0.267, 0.0, 0.963, 0.3],
        [0.0, 1.0, 0.0, -0.09],
        [-0.963, 0.0, 0.267, 0.26],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

Vec = npt.NDArray[np.float64]


def validate_dim(vec: npt.ArrayLike) -> Vec:
    """Return a one-dimensional float vector of length seven."""
    arr: Vec = np.asarray(vec, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] != TOTAL_DIM:
        raise ValueError(f"expected a {TOTAL_DIM}-D vector, got shape {np.shape(vec)}")
    return arr


def displacement(vec: npt.ArrayLike) -> Vec:
    """Return a copy of the six end-effector displacement slots."""
    out: Vec = validate_dim(vec)[:POSE_DIM].copy()
    return out


def gripper(vec: npt.ArrayLike) -> float:
    """Return the absolute normalized gripper target as a scalar."""
    return float(validate_dim(vec)[GRIPPER_IDX])


def build_start_transform(
    start_eef_pos: npt.ArrayLike,
    override: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    """Build the reset matrix from the reference rotation or a full override."""
    if override is not None:
        values = np.asarray(override, dtype=np.float64)
        if values.size != 16:
            raise ValueError("start_transform must have 16 row-major entries")
        matrix: npt.NDArray[np.float64] = values.reshape(4, 4).copy()
        return matrix
    position = np.asarray(start_eef_pos, dtype=np.float64)
    if position.shape != (3,):
        raise ValueError("start_eef_pos must have 3 entries")
    transform = START_TRANSFORM.copy()
    transform[:3, 3] = position
    built: npt.NDArray[np.float64] = transform
    return built
