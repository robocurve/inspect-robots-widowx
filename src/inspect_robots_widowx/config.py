"""Validated configuration and shared spaces for WidowX policy pairs.

The action's rotation is deliberately declared as ``rotation_repr="none"``.
Its ``droll``, ``dpitch``, and ``dyaw`` labels carry the displacement meaning
until inspect-robots issue #143 allows Euler displacement modes through the
default approver: https://github.com/robocurve/inspect-robots/issues/143.

BridgeData's gripper slot is absolute even though the other six slots are
displacements. It stays in [0, 1], with 1 open. The observed Euler angles are
relative to the controller's default rotation, not absolute world orientation.
The default OpenVLA transport approximates the reference Lanczos3 preprocessing
with Pillow Lanczos and an optional in-memory JPEG round trip.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, ClassVar, TypeVar

import numpy as np
import numpy.typing as npt
from inspect_robots.spaces import (
    ActionSemantics,
    Box,
    CameraSpec,
    ObservationSpace,
    StateField,
    StateSpec,
)

from inspect_robots_widowx.packing import DIM_LABELS, STATE_KEY, TOTAL_DIM

_T = TypeVar("_T", bound="_FromKwargs")

DEFAULT_DELTA_LOW = (-0.05, -0.05, -0.05, -0.25, -0.25, -0.25, 0.0)
DEFAULT_DELTA_HIGH = (0.05, 0.05, 0.05, 0.25, 0.25, 0.25, 1.0)
DEFAULT_START_EEF_POS = (0.3, -0.09, 0.26)

ACTION_SEMANTICS = ActionSemantics(
    control_mode="eef_delta_pose",
    rotation_repr="none",
    gripper="continuous",
    frame="base",
    dim_labels=DIM_LABELS,
)

STATE_SPEC = StateSpec(
    fields=(StateField(key=STATE_KEY, shape=(TOTAL_DIM,), unit="m+rad+normalized"),)
)


class _FromKwargs:
    """Build frozen dataclasses from flat CLI-friendly keyword arguments."""

    _FLOAT_TUPLE_FIELDS: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def from_kwargs(cls: type[_T], **flat: Any) -> _T:
        """Reject unknown keys and parse configured comma-separated tuples."""
        names = {field.name for field in dataclasses.fields(cls)}  # type: ignore[arg-type]
        unknown = set(flat) - names
        if unknown:
            raise TypeError(f"{cls.__name__} got unexpected config keys: {sorted(unknown)}")
        for key in cls._FLOAT_TUPLE_FIELDS & set(flat):
            value = flat[key]
            if isinstance(value, str):
                try:
                    flat[key] = tuple(float(part) for part in value.split(","))
                except ValueError:
                    raise ValueError(
                        f"{key} must be a comma-separated list of numbers, got {value!r}"
                    ) from None
        return cls(**flat)


def _positive_finite(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and > 0")


def _positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _valid_port(port: int) -> None:
    if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
        raise ValueError("port must be an integer in [1, 65535]")


@dataclass(frozen=True)
class WidowXConfig(_FromKwargs):
    """Static hardware, safety, reset, observation, and pacing configuration."""

    _FLOAT_TUPLE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"delta_low", "delta_high", "start_eef_pos", "start_transform"}
    )

    host: str = "localhost"
    port: int = 5556
    control_hz: float = 5.0
    move_duration: float = 0.2
    delta_low: tuple[float, ...] = DEFAULT_DELTA_LOW
    delta_high: tuple[float, ...] = DEFAULT_DELTA_HIGH
    move_to_start: bool = True
    start_eef_pos: tuple[float, ...] = DEFAULT_START_EEF_POS
    start_transform: tuple[float, ...] | None = None
    obs_timeout_s: float = 10.0
    unattended: bool = False
    docs_extra: str = ""

    def __post_init__(self) -> None:
        """Reject configurations that violate the fixed BridgeData contract."""
        if not self.host:
            raise ValueError("host must not be empty")
        _valid_port(self.port)
        _positive_finite("control_hz", self.control_hz)
        _positive_finite("move_duration", self.move_duration)
        _positive_finite("obs_timeout_s", self.obs_timeout_s)
        if not np.isclose(self.move_duration, 1.0 / self.control_hz, rtol=1e-9, atol=1e-12):
            raise ValueError("move_duration must equal 1/control_hz")
        if len(self.delta_low) != TOTAL_DIM or len(self.delta_high) != TOTAL_DIM:
            raise ValueError(f"delta_low and delta_high must have {TOTAL_DIM} entries")
        low, high = self.low, self.high
        if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
            raise ValueError("delta_low and delta_high must contain only finite values")
        if np.any(low >= high):
            raise ValueError("delta_low must be below delta_high in every dimension")
        if low[-1] != 0.0 or high[-1] != 1.0:
            raise ValueError("gripper bounds must be exactly [0, 1]")
        if len(self.start_eef_pos) != 3 or not np.all(np.isfinite(self.start_eef_pos)):
            raise ValueError("start_eef_pos must have 3 finite entries")
        if self.start_transform is not None:
            values = np.asarray(self.start_transform, dtype=np.float64)
            if values.shape != (16,) or not np.all(np.isfinite(values)):
                raise ValueError("start_transform must have 16 finite row-major entries")
            if not np.array_equal(values.reshape(4, 4)[3], np.asarray([0.0, 0.0, 0.0, 1.0])):
                raise ValueError("start_transform must be a homogeneous 4x4 matrix")

    @property
    def low(self) -> npt.NDArray[np.float64]:
        """Return configured lower per-step bounds as float64."""
        return np.asarray(self.delta_low, dtype=np.float64)

    @property
    def high(self) -> npt.NDArray[np.float64]:
        """Return configured upper per-step bounds as float64."""
        return np.asarray(self.delta_high, dtype=np.float64)


@dataclass(frozen=True)
class OpenVLAConfig(_FromKwargs):
    """OpenVLA REST transport and Bridge preprocessing configuration."""

    server_url: str = "http://127.0.0.1:8000"
    endpoint: str = "/act"
    unnorm_key: str = "bridge_orig"
    timeout_s: float = 30.0
    name: str = "openvla"
    resize_px: int = 256
    jpeg_roundtrip: bool = True

    def __post_init__(self) -> None:
        """Reject invalid transport settings before network use."""
        if not self.server_url:
            raise ValueError("server_url must not be empty")
        if not self.endpoint.startswith("/"):
            raise ValueError("endpoint must start with '/'")
        if not self.unnorm_key:
            raise ValueError("unnorm_key must not be empty")
        if not self.name:
            raise ValueError("name must not be empty")
        _positive_finite("timeout_s", self.timeout_s)
        _positive_int("resize_px", self.resize_px)
        if not isinstance(self.jpeg_roundtrip, bool):
            raise ValueError("jpeg_roundtrip must be a bool")

    @property
    def url(self) -> str:
        """Return the normalized endpoint URL."""
        return self.server_url.rstrip("/") + self.endpoint


@dataclass(frozen=True)
class OpenpiConfig(_FromKwargs):
    """Openpi websocket transport and action-chunk configuration."""

    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str | None = None
    action_horizon: int = 10
    replan_interval: int = 5
    name: str = "openpi"
    resize_px: int = 224

    def __post_init__(self) -> None:
        """Reject invalid transport and chunk metadata before network use."""
        if not self.host:
            raise ValueError("host must not be empty")
        _valid_port(self.port)
        if not self.name:
            raise ValueError("name must not be empty")
        for field_name in ("action_horizon", "replan_interval", "resize_px"):
            _positive_int(field_name, getattr(self, field_name))


def action_box(cfg: WidowXConfig | None = None) -> Box:
    """Build the shared 7-D displacement space, optionally with rig bounds."""
    base = cfg if cfg is not None else WidowXConfig()
    return Box(shape=(TOTAL_DIM,), low=base.low, high=base.high, semantics=ACTION_SEMANTICS)


def observation_space(include_state: bool) -> ObservationSpace:
    """Build the single-camera contract, optionally including EEF state."""
    camera = CameraSpec(name="external_cam", height=480, width=640, channels=3)
    return ObservationSpace(cameras=(camera,), state=STATE_SPEC if include_state else None)
