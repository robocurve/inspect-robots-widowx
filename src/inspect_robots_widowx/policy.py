"""OpenVLA REST and openpi websocket policies for BridgeData actions."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any, ClassVar

import numpy as np
from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.scene import Scene
from inspect_robots.types import Action, ActionChunk, Observation

from inspect_robots_widowx import packing
from inspect_robots_widowx.config import (
    OpenpiConfig,
    OpenVLAConfig,
    action_box,
    observation_space,
)

OPENPI_CLIENT_INSTALL_COMMAND = (
    'pip install "openpi-client @ '
    "git+https://github.com/Physical-Intelligence/openpi.git"
    '#subdirectory=packages/openpi-client"'
)

PostFn = Callable[[str, Mapping[str, Any]], Any]
OpenpiObservation = Mapping[str, Any]
OpenpiResponse = Mapping[str, Any]
InferFn = Callable[[OpenpiObservation], OpenpiResponse]


def _default_post(cfg: OpenVLAConfig) -> PostFn:  # pragma: no cover - live HTTP transport
    """Build the canonical Bridge image preprocessing and REST transport."""
    from io import BytesIO

    import json_numpy
    import requests
    from PIL import Image

    def post(url: str, payload: Mapping[str, Any]) -> Any:
        image = Image.fromarray(np.asarray(payload["image"], dtype=np.uint8))
        resized = image.resize((cfg.resize_px, cfg.resize_px), Image.Resampling.LANCZOS)
        if cfg.jpeg_roundtrip:
            encoded = BytesIO()
            resized.save(encoded, format="JPEG")
            encoded.seek(0)
            with Image.open(encoded) as decoded_image:
                prepared = np.asarray(decoded_image.convert("RGB"), dtype=np.uint8).copy()
        else:
            prepared = np.asarray(resized.convert("RGB"), dtype=np.uint8)
        request_payload = {**payload, "image": prepared}
        response = requests.post(
            url,
            data=json_numpy.dumps(request_payload),
            timeout=cfg.timeout_s,
        )
        response.raise_for_status()
        decoded = json_numpy.loads(response.content)
        if isinstance(decoded, str):
            decoded = json_numpy.loads(decoded)
        return decoded

    return post


def _default_infer(cfg: OpenpiConfig) -> InferFn:  # pragma: no cover - live websocket transport
    """Build the upstream websocket client and resize-with-pad transport."""
    try:
        from openpi_client import image_tools, websocket_client_policy
    except ModuleNotFoundError as exc:
        if exc.name != "openpi_client" and not (exc.name or "").startswith("openpi_client."):
            raise
        raise ModuleNotFoundError(
            "The Physical Intelligence openpi-client is git-only. Install it with: "
            f"{OPENPI_CLIENT_INSTALL_COMMAND}",
            name=exc.name,
        ) from exc

    client = websocket_client_policy.WebsocketClientPolicy(
        host=cfg.host,
        port=cfg.port,
        api_key=cfg.api_key,
    )

    def infer(observation: OpenpiObservation) -> OpenpiResponse:
        payload = dict(observation)
        payload["observation/image"] = image_tools.resize_with_pad(
            np.asarray(payload["observation/image"]), cfg.resize_px, cfg.resize_px
        )
        response: OpenpiResponse = client.infer(payload)
        return response

    return infer


class OpenVLAPolicy:
    """Inspect Robots policy for OpenVLA's first-party ``/act`` server."""

    RUNTIME_REQUIREMENTS: ClassVar[Mapping[str, str]] = {
        "requests": "uv pip install inspect-robots-widowx",
        "json_numpy": "uv pip install inspect-robots-widowx",
        "PIL": "uv pip install inspect-robots-widowx",
    }

    def __init__(
        self,
        config: OpenVLAConfig | None = None,
        *,
        post_fn: PostFn | None = None,
        clock: Callable[[], float] | None = None,
        **flat: Any,
    ) -> None:
        self._cfg = config if config is not None else OpenVLAConfig.from_kwargs(**flat)
        self._post_fn = post_fn
        self._clock: Callable[[], float] = clock or time.perf_counter
        self._instruction: str | None = None
        self.num_inferences = 0
        self.info = PolicyInfo(
            name=self._cfg.name,
            action_space=action_box(),
            observation_space=observation_space(include_state=False),
            control_hz=None,
        )
        self.config = PolicyConfig(action_horizon=1, replan_interval=None)

    def _post(self) -> PostFn:
        """Lazily construct the default REST transport."""
        if self._post_fn is None:
            self._post_fn = _default_post(self._cfg)
        return self._post_fn

    def reset(self, scene: Scene) -> None:
        """Stash the language instruction and reset inference accounting."""
        self._instruction = scene.instruction
        self.num_inferences = 0

    def act(self, observation: Observation) -> ActionChunk:
        """Query OpenVLA and return its one BridgeData action unchanged."""
        try:
            image = observation.images["external_cam"]
        except KeyError as exc:
            raise ValueError(
                "observation missing camera 'external_cam' required by openvla"
            ) from exc
        payload: dict[str, Any] = {
            "image": np.asarray(image, dtype=np.uint8),
            "instruction": self._instruction or "",
            "unnorm_key": self._cfg.unnorm_key,
        }
        started = self._clock()
        response = self._post()(self._cfg.url, payload)
        elapsed = self._clock() - started
        action = np.asarray(response, dtype=np.float64)
        if action.ndim != 1 or action.shape[0] != packing.TOTAL_DIM:
            raise ValueError(
                f"OpenVLA returned action of shape {action.shape}; expected ({packing.TOTAL_DIM},)"
            )
        if not np.isfinite(action).all():
            raise ValueError("OpenVLA returned a non-finite action")
        self.num_inferences += 1
        return ActionChunk(
            actions=[Action(data=action.copy())],
            control_hz=5.0,
            inference_latency_s=elapsed,
        )


class OpenpiPolicy:
    """Inspect Robots policy for openpi Bridge fine-tune websocket servers."""

    def __init__(
        self,
        config: OpenpiConfig | None = None,
        *,
        infer_fn: InferFn | None = None,
        clock: Callable[[], float] | None = None,
        **flat: Any,
    ) -> None:
        self._cfg = config if config is not None else OpenpiConfig.from_kwargs(**flat)
        self._infer_fn = infer_fn
        self._clock: Callable[[], float] = clock or time.perf_counter
        self._instruction: str | None = None
        self.num_inferences = 0
        self.info = PolicyInfo(
            name=self._cfg.name,
            action_space=action_box(),
            observation_space=observation_space(include_state=False),
            control_hz=None,
        )
        self.config = PolicyConfig(
            action_horizon=self._cfg.action_horizon,
            replan_interval=self._cfg.replan_interval,
        )

    def _infer(self) -> InferFn:
        """Lazily construct the real websocket inference closure."""
        if self._infer_fn is None:
            self._infer_fn = _default_infer(self._cfg)
        return self._infer_fn

    def reset(self, scene: Scene) -> None:
        """Stash the language instruction and reset inference accounting."""
        self._instruction = scene.instruction
        self.num_inferences = 0

    def act(self, observation: Observation) -> ActionChunk:
        """Return a finite 7-D Bridge action chunk without polarity changes."""
        try:
            image = observation.images["external_cam"]
        except KeyError as exc:
            raise ValueError(
                "observation missing camera 'external_cam' required by openpi"
            ) from exc
        request: dict[str, Any] = {
            "observation/image": np.asarray(image, dtype=np.uint8),
            "prompt": self._instruction or "",
        }
        started = self._clock()
        response = self._infer()(request)
        elapsed = self._clock() - started
        if "actions" not in response:
            raise ValueError("openpi response missing 'actions'")
        actions = np.asarray(response["actions"], dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != packing.TOTAL_DIM:
            raise ValueError(
                f"openpi returned actions of shape {actions.shape}; "
                f"expected (N, {packing.TOTAL_DIM})"
            )
        if actions.shape[0] == 0:
            raise ValueError("openpi returned an empty action chunk")
        if not np.isfinite(actions).all():
            raise ValueError("openpi returned non-finite actions")
        adapted = actions[: self._cfg.action_horizon]
        self.num_inferences += 1
        return ActionChunk(
            actions=[Action(data=row.copy()) for row in adapted],
            control_hz=5.0,
            inference_latency_s=elapsed,
        )
