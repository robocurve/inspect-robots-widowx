"""Real WidowX embodiment backed by the BridgeData network client.

Every action is finite-checked and hard-clamped before reaching the server.
The embodiment owns 5 Hz pacing and reports success only through
``termination_reason="success"``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from enum import IntEnum
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from inspect_robots.conformance import DeviceSlot
from inspect_robots.embodiment import SELF_PACED, EmbodimentInfo
from inspect_robots.scene import Scene
from inspect_robots.types import Action, Observation, StepResult

from inspect_robots_widowx import packing
from inspect_robots_widowx._bridge import (
    EDGEML_INSTALL_COMMAND,
    WIDOWX_ENVS_INSTALL_COMMAND,
    _load_widowx_client,
)
from inspect_robots_widowx.config import WidowXConfig, action_box, observation_space
from inspect_robots_widowx.operator import OperatorIO, default_poll_end

_OBS_RETRY_S = 0.1

_DOCS = """A single WidowX 250S arm using BridgeData end-effector displacement control.
The base-frame action has six relative pose slots and one absolute gripper target.
Observed Euler angles are relative to the controller's default rotation.
- dx: end-effector x displacement in metres.
- dy: end-effector y displacement in metres.
- dz: end-effector z displacement in metres.
- droll: end-effector roll displacement in radians.
- dpitch: end-effector pitch displacement in radians.
- dyaw: end-effector yaw displacement in radians.
- gripper: absolute opening target, with 0 fully closed and 1 fully open.
Keep people clear of the workspace and verify the camera after deliberate motion."""


class Status(IntEnum):
    """All status values returned by the upstream BridgeData client."""

    NO_CONNECTION = 0
    SUCCESS = 1
    EXECUTION_FAILURE = 2
    NOT_INITIALIZED = 3


@runtime_checkable
class TaskEnvelopeLike(Protocol):
    """Read-only rollout metadata accepted by the optional task-binding hook."""

    @property
    def max_steps(self) -> int:
        """Return the framework-enforced rollout horizon."""
        ...


@runtime_checkable
class Client(Protocol):
    """Minimal BridgeData client surface used by the embodiment."""

    def init(self, env_params: Mapping[str, Any]) -> Status:
        """Initialize the remote robot environment."""
        ...

    def reset(self) -> Status:
        """Move the remote environment to its neutral reset state."""
        ...

    def move(self, pose_4x4: npt.NDArray[np.float64], duration: float, blocking: bool) -> Status:
        """Move to a homogeneous end-effector pose."""
        ...

    def step_action(self, action: npt.NDArray[np.float64], blocking: bool) -> Status:
        """Execute one BridgeData action."""
        ...

    def get_observation(self) -> Mapping[str, Any] | None:
        """Return the latest decoded observation, or None while unavailable."""
        ...

    def stop(self) -> None:
        """Release the client connection."""
        ...


ClientFactory = Callable[[WidowXConfig], Client]


def _default_client_factory(cfg: WidowXConfig) -> Client:  # pragma: no cover - real hardware
    client_type = _load_widowx_client()
    client: Client = client_type(cfg.host, cfg.port)
    return client


def _env_params(cfg: WidowXConfig) -> Mapping[str, Any]:
    """Return the canonical BridgeData evaluation environment parameters."""
    return {
        "fix_zangle": 0.1,
        "move_duration": cfg.move_duration,
        "adaptive_wait": True,
        "move_to_rand_start_freq": 1,
        "override_workspace_boundaries": [
            [0.1, -0.15, -0.1, -1.57, 0.0],
            [0.45, 0.25, 0.18, 1.57, 0.0],
        ],
        "action_clipping": "xyz",
        "catch_environment_except": False,
        "start_state": [0.3, 0.0, 0.15, 0.0, 0.0, 0.0, 1.0],
        "skip_move_to_neutral": False,
        "return_full_image": False,
        "camera_topics": [{"name": "/blue/image_raw"}],
    }


def _require_success(operation: str, code: int) -> None:
    """Raise a descriptive fault for every non-success status value."""
    if code == Status.SUCCESS:
        return
    try:
        status = Status(code)
        rendered = f"{status.name} ({status.value})"
    except (TypeError, ValueError):
        rendered = repr(code)
    raise RuntimeError(f"WidowX {operation} failed with status {rendered}")


class WidowXEmbodiment:
    """Inspect Robots embodiment for a server-backed WidowX 250S."""

    RUNTIME_REQUIREMENTS: ClassVar[Mapping[str, str]] = {
        "widowx_envs": WIDOWX_ENVS_INSTALL_COMMAND,
        "edgeml": EDGEML_INSTALL_COMMAND,
    }
    DEVICE_SLOTS: ClassVar[tuple[DeviceSlot, ...]] = ()

    def __init__(
        self,
        config: WidowXConfig | None = None,
        *,
        client_factory: ClientFactory | None = None,
        operator: OperatorIO | None = None,
        poll_end: Callable[[], bool] | None = None,
        clock: Callable[[], float] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        **flat: Any,
    ) -> None:
        self._cfg = config if config is not None else WidowXConfig.from_kwargs(**flat)
        self._client_factory: ClientFactory = client_factory or _default_client_factory
        self._operator = operator if operator is not None else OperatorIO()
        self._poll_end: Callable[[], bool] = poll_end or default_poll_end
        self._clock: Callable[[], float] = clock or time.perf_counter
        self._sleep: Callable[[float], None] = sleep_fn or time.sleep
        self._client: Client | None = None
        self._instruction: str | None = None
        self._t_last = 0.0
        self._bound_max_steps: int | None = None
        self.num_steps = 0

        docs = _DOCS
        docs_extra = self._cfg.docs_extra.strip()
        if docs_extra:
            docs += "\n\n" + docs_extra
        self.info = EmbodimentInfo(
            name="widowx",
            action_space=action_box(self._cfg),
            observation_space=observation_space(include_state=True),
            control_hz=self._cfg.control_hz,
            is_simulated=False,
            capabilities=frozenset({SELF_PACED}),
            docs=docs,
        )

    def bind_task(self, envelope: TaskEnvelopeLike) -> None:
        """Store the rollout horizon for operator-facing status."""
        self._bound_max_steps = int(envelope.max_steps)

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Connect lazily, reset, optionally move to start, and observe."""
        if self._client is None:
            self._client = self._client_factory(self._cfg)
            _require_success("init", self._client.init(_env_params(self._cfg)))
        client = self._client
        _require_success("reset", client.reset())
        if self._cfg.move_to_start:
            transform = packing.build_start_transform(
                self._cfg.start_eef_pos, self._cfg.start_transform
            )
            _require_success(
                "move",
                client.move(transform, duration=self._cfg.start_move_duration_s, blocking=True),
            )
        if not self._cfg.unattended:
            self._operator.wait_ready()
            horizon = self._horizon_seconds()
            limit = f" Max {horizon:.0f}s." if horizon is not None else ""
            self._operator.output_fn(
                "Running: press Enter to end the episode, then y/N to score." + limit
            )
        self._instruction = scene.instruction
        self.num_steps = 0
        observation = self._observe(scene.instruction)
        self._t_last = self._clock()
        return observation

    def step(self, action: Action) -> StepResult:
        """Clamp, command, pace, observe, and optionally collect a verdict."""
        client = self._require_client()
        command = packing.validate_dim(action.data)
        if not np.isfinite(command).all():
            raise ValueError("action must contain only finite values")
        clamped = np.clip(command, self._cfg.low, self._cfg.high)
        _require_success("step_action", client.step_action(clamped, blocking=False))
        self.num_steps += 1
        self._pace()
        observation = self._observe(self._instruction)
        if not self._cfg.unattended and self._poll_end():
            success = self._operator.confirm_success()
            return StepResult(
                observation=observation,
                terminated=True,
                termination_reason="success" if success else "failure",
                info={"operator_confirmed": success},
            )
        return StepResult(observation=observation, terminated=False)

    def close(self) -> None:
        """Stop the client and clear the handle, including when stop raises."""
        self._bound_max_steps = None
        client = self._client
        if client is None:
            return
        try:
            client.stop()
        finally:
            self._client = None

    def _require_client(self) -> Client:
        """Return the live client or reject step-before-reset use."""
        if self._client is None:
            raise RuntimeError("step() called before reset() (or after close())")
        return self._client

    def _pace(self) -> None:
        """Sleep for the remainder of the configured control period."""
        elapsed = self._clock() - self._t_last
        self._sleep(max(0.0, 1.0 / self._cfg.control_hz - elapsed))
        self._t_last = self._clock()

    def _horizon_seconds(self) -> float | None:
        """Return the bound rollout horizon in self-paced wall-clock seconds."""
        if self._bound_max_steps is None:
            return None
        return self._bound_max_steps / self._cfg.control_hz

    def _observe(self, instruction: str | None) -> Observation:
        """Retry the server, then adapt full_image and the 7-D EEF state."""
        client = self._require_client()
        started = self._clock()
        waited = 0.0
        raw = client.get_observation()
        while raw is None:
            elapsed = max(self._clock() - started, waited)
            if elapsed >= self._cfg.obs_timeout_s:
                raise RuntimeError(
                    f"WidowX observation timed out after {self._cfg.obs_timeout_s:g}s"
                )
            delay = min(_OBS_RETRY_S, self._cfg.obs_timeout_s - elapsed)
            self._sleep(delay)
            waited += delay
            raw = client.get_observation()
        if "full_image" not in raw:
            raise ValueError("WidowX observation missing 'full_image'")
        if "state" not in raw:
            raise ValueError("WidowX observation missing 'state'")
        image = np.asarray(raw["full_image"], dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"full_image must have shape (H, W, 3), got {image.shape}")
        state = packing.validate_dim(raw["state"])
        if not np.isfinite(state).all():
            raise ValueError("WidowX state must contain only finite values")
        observed_at = self._clock()
        return Observation(
            images={"external_cam": image},
            state={packing.STATE_KEY: state.copy()},
            instruction=instruction,
            image_times={"external_cam": observed_at},
            state_time=observed_at,
        )
