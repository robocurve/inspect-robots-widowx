from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pytest
from inspect_robots.policy import PolicyConfig
from inspect_robots.scene import Scene
from inspect_robots.types import Observation

from inspect_robots_widowx.config import OpenpiConfig, OpenVLAConfig
from inspect_robots_widowx.policy import OpenpiPolicy, OpenVLAPolicy


def _observation() -> Observation:
    image = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    return Observation(
        images={"external_cam": image},
        state={"eef_pose": np.arange(7, dtype=float)},
        instruction="observation instruction",
    )


def test_policy_info_config_and_api_key_exclusion() -> None:
    openvla = OpenVLAPolicy()
    assert openvla.info.name == "openvla"
    assert openvla.info.action_space.dim == 7
    assert openvla.info.control_hz is None
    assert openvla.info.observation_space.state_keys == frozenset()
    assert openvla.config == PolicyConfig(action_horizon=1, replan_interval=None)

    openpi = OpenpiPolicy(OpenpiConfig(api_key="secret"))
    assert openpi.info.name == "openpi"
    assert openpi.info.action_space.dim == 7
    assert openpi.info.control_hz is None
    assert openpi.info.observation_space.state_keys == frozenset()
    assert openpi.config == PolicyConfig(action_horizon=10, replan_interval=5)
    assert "api_key" not in asdict(openpi.config)


def test_openvla_payload_is_exact_raw_and_threads_instruction() -> None:
    captured: dict[str, Any] = {}

    def post(url: str, payload: dict[str, Any]) -> np.ndarray:
        captured["url"] = url
        captured["payload"] = payload
        return np.asarray([0.01, -0.02, 0.03, -0.04, 0.05, -0.06, 0.73])

    clock = iter([2.0, 2.4]).__next__
    policy = OpenVLAPolicy(post_fn=post, clock=clock)  # type: ignore[arg-type]
    policy.reset(Scene(id="s", instruction="pick the mug"))
    chunk = policy.act(_observation())
    assert captured["url"] == "http://127.0.0.1:8000/act"
    payload = captured["payload"]
    assert list(payload) == ["image", "instruction", "unnorm_key"]
    assert np.array_equal(payload["image"], _observation().images["external_cam"])
    assert payload["image"].shape == (2, 4, 3)
    assert payload["instruction"] == "pick the mug"
    assert payload["unnorm_key"] == "bridge_orig"
    assert len(chunk) == 1
    assert chunk.control_hz == 5.0
    assert chunk.inference_latency_s == pytest.approx(0.4)
    assert chunk.actions[0].data[-1] == pytest.approx(0.73)
    assert policy.num_inferences == 1


def test_openvla_default_transport_factory_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[OpenVLAConfig] = []

    def factory(cfg: OpenVLAConfig):
        calls.append(cfg)
        return lambda _url, _payload: np.zeros(7)

    monkeypatch.setattr("inspect_robots_widowx.policy._default_post", factory)
    policy = OpenVLAPolicy(clock=lambda: 0.0)
    assert calls == []
    policy.act(_observation())
    assert calls == [policy._cfg]


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (np.zeros(6), r"expected \(7,\)"),
        (np.zeros((1, 7)), r"expected \(7,\)"),
        (np.full(7, np.nan), "non-finite"),
    ],
)
def test_openvla_response_validation(response: np.ndarray, message: str) -> None:
    policy = OpenVLAPolicy(post_fn=lambda _url, _payload: response, clock=lambda: 0.0)
    with pytest.raises(ValueError, match=message):
        policy.act(_observation())


def test_openpi_request_passes_image_instruction_and_asymmetric_gripper_unchanged() -> None:
    captured: dict[str, Any] = {}
    raw = np.asarray(
        [
            [0.01, -0.02, 0.03, -0.04, 0.05, -0.06, 0.73],
            [0.02, -0.01, 0.04, -0.03, 0.06, -0.05, 0.27],
        ]
    )

    def infer(payload: dict[str, Any]) -> dict[str, np.ndarray]:
        captured.update(payload)
        return {"actions": raw}

    policy = OpenpiPolicy(infer_fn=infer, clock=iter([4.0, 4.25]).__next__)  # type: ignore[arg-type]
    policy.reset(Scene(id="s", instruction="move the spoon"))
    chunk = policy.act(_observation())
    assert list(captured) == ["observation/image", "prompt"]
    assert np.array_equal(captured["observation/image"], _observation().images["external_cam"])
    assert captured["prompt"] == "move the spoon"
    assert len(chunk) == 2
    assert chunk.control_hz == 5.0
    assert chunk.inference_latency_s == pytest.approx(0.25)
    assert chunk.actions[0].data[-1] == pytest.approx(0.73)
    assert chunk.actions[1].data[-1] == pytest.approx(0.27)
    assert np.array_equal(chunk.actions[0].data, raw[0])
    assert policy.num_inferences == 1


def test_openpi_default_infer_factory_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[OpenpiConfig] = []

    def factory(cfg: OpenpiConfig):
        calls.append(cfg)
        return lambda _payload: {"actions": np.zeros((1, 7))}

    monkeypatch.setattr("inspect_robots_widowx.policy._default_infer", factory)
    policy = OpenpiPolicy(clock=lambda: 0.0)
    assert calls == []
    policy.act(_observation())
    assert calls == [policy._cfg]


def test_openpi_truncates_to_configured_action_horizon() -> None:
    actions = np.arange(35, dtype=float).reshape(5, 7)
    policy = OpenpiPolicy(
        OpenpiConfig(action_horizon=2),
        infer_fn=lambda _payload: {"actions": actions},
        clock=lambda: 0.0,
    )
    chunk = policy.act(_observation())
    assert len(chunk) == 2
    assert np.array_equal(chunk.actions[1].data, actions[1])


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ({}, "missing 'actions'"),
        ({"actions": np.zeros((2, 6))}, r"expected \(N, 7\)"),
        ({"actions": np.zeros(7)}, r"expected \(N, 7\)"),
        ({"actions": np.zeros((0, 7))}, "empty action chunk"),
        ({"actions": np.full((1, 7), np.inf)}, "non-finite"),
    ],
)
def test_openpi_response_validation(response: dict[str, np.ndarray], message: str) -> None:
    policy = OpenpiPolicy(infer_fn=lambda _payload: response, clock=lambda: 0.0)
    with pytest.raises(ValueError, match=message):
        policy.act(_observation())


@pytest.mark.parametrize("policy", [OpenVLAPolicy(), OpenpiPolicy()])
def test_helpful_missing_camera_errors(policy: OpenVLAPolicy | OpenpiPolicy) -> None:
    with pytest.raises(ValueError, match="missing camera 'external_cam'"):
        policy.act(Observation(images={}, state={}))


def test_reset_replaces_instruction_and_resets_counters() -> None:
    openvla_payloads: list[dict[str, Any]] = []
    openvla = OpenVLAPolicy(
        post_fn=lambda _url, payload: openvla_payloads.append(dict(payload)) or np.zeros(7),
        clock=lambda: 0.0,
    )
    openvla.act(_observation())
    assert openvla_payloads[-1]["instruction"] == ""
    assert openvla.num_inferences == 1
    openvla.reset(Scene(id="s", instruction="new OpenVLA instruction"))
    assert openvla.num_inferences == 0
    openvla.act(_observation())
    assert openvla_payloads[-1]["instruction"] == "new OpenVLA instruction"

    openpi_payloads: list[dict[str, Any]] = []
    openpi = OpenpiPolicy(
        infer_fn=lambda payload: (
            openpi_payloads.append(dict(payload)) or {"actions": np.zeros((1, 7))}
        ),
        clock=lambda: 0.0,
    )
    openpi.act(_observation())
    assert openpi_payloads[-1]["prompt"] == ""
    assert openpi.num_inferences == 1
    openpi.reset(Scene(id="s", instruction="new openpi instruction"))
    assert openpi.num_inferences == 0
    openpi.act(_observation())
    assert openpi_payloads[-1]["prompt"] == "new openpi instruction"
