from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest
from inspect_robots.conformance import check_embodiment, device_slots, missing_runtime_requirements
from inspect_robots.embodiment import SELF_PACED
from inspect_robots.scene import Scene
from inspect_robots.task import TaskEnvelope
from inspect_robots.types import Action

from inspect_robots_widowx._bridge import EDGEML_INSTALL_COMMAND, WIDOWX_ENVS_INSTALL_COMMAND
from inspect_robots_widowx.config import WidowXConfig
from inspect_robots_widowx.embodiment import (
    Client,
    Status,
    WidowXEmbodiment,
    _env_params,
    _require_success,
)
from inspect_robots_widowx.operator import OperatorIO
from inspect_robots_widowx.packing import START_TRANSFORM


def _raw_observation() -> dict[str, np.ndarray]:
    return {
        "full_image": np.arange(18, dtype=np.uint8).reshape(2, 3, 3),
        "image": np.full((3, 1, 1), 255.0),
        "state": np.asarray([0.3, -0.09, 0.26, 0.1, -0.2, 0.3, 0.73]),
    }


class _FakeClient:
    def __init__(self, events: list[str] | None = None) -> None:
        self.events = events if events is not None else []
        self.statuses: dict[str, int] = {}
        self.observations: list[Mapping[str, Any] | None] = [_raw_observation()]
        self.env_params: Mapping[str, Any] | None = None
        self.moves: list[tuple[np.ndarray, float, bool]] = []
        self.actions: list[tuple[np.ndarray, bool]] = []
        self.stop_calls = 0
        self.stop_error = False

    def _status(self, operation: str) -> int:
        return self.statuses.get(operation, Status.SUCCESS)

    def init(self, env_params: Mapping[str, Any]) -> int:
        self.events.append("init")
        self.env_params = env_params
        return self._status("init")

    def reset(self) -> int:
        self.events.append("reset")
        return self._status("reset")

    def move(self, pose_4x4: np.ndarray, duration: float, blocking: bool) -> int:
        self.events.append("move")
        self.moves.append((np.asarray(pose_4x4).copy(), duration, blocking))
        return self._status("move")

    def step_action(self, action: np.ndarray, blocking: bool) -> int:
        self.events.append("step_action")
        self.actions.append((np.asarray(action).copy(), blocking))
        return self._status("step_action")

    def get_observation(self) -> Mapping[str, Any] | None:
        self.events.append("observe")
        if len(self.observations) > 1:
            return self.observations.pop(0)
        return self.observations[0]

    def stop(self) -> None:
        self.events.append("stop")
        self.stop_calls += 1
        if self.stop_error:
            raise RuntimeError("stop failed")


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self.now += delay


def _scene() -> Scene:
    return Scene(id="s", instruction="pick up the cube")


def _embodiment(
    client: _FakeClient,
    cfg: WidowXConfig | None = None,
    **kwargs: Any,
) -> WidowXEmbodiment:
    return WidowXEmbodiment(
        cfg or WidowXConfig(unattended=True),
        client_factory=lambda _cfg: client,  # type: ignore[arg-type,return-value]
        clock=lambda: 0.0,
        sleep_fn=lambda _delay: None,
        **kwargs,
    )


def test_init_is_inert_and_declares_contract() -> None:
    calls = 0

    def factory(_cfg: WidowXConfig) -> Client:
        nonlocal calls
        calls += 1
        return _FakeClient()  # type: ignore[return-value]

    embodiment = WidowXEmbodiment(client_factory=factory)
    assert calls == 0
    assert embodiment.info.name == "widowx"
    assert embodiment.info.action_space.dim == 7
    assert embodiment.info.control_hz == 5.0
    assert embodiment.info.capabilities == frozenset({SELF_PACED})
    assert embodiment.info.observation_space.state_keys == frozenset({"eef_pose"})


def test_reset_flow_uses_reference_4x4_blocking_move_then_operator() -> None:
    events: list[str] = []
    client = _FakeClient(events)
    operator = OperatorIO(
        input_fn=lambda _prompt: events.append("wait_ready") or "",
        output_fn=lambda _line: events.append("running"),
    )
    embodiment = _embodiment(client, WidowXConfig(), operator=operator, poll_end=lambda: False)
    observation = embodiment.reset(_scene())
    assert events[:6] == ["init", "reset", "move", "wait_ready", "running", "observe"]
    assert client.env_params == _env_params(WidowXConfig())
    assert client.env_params["override_workspace_boundaries"] == [
        [0.1, -0.15, -0.1, -1.57, 0.0],
        [0.45, 0.25, 0.18, 1.57, 0.0],
    ]
    assert client.env_params["camera_topics"] == [{"name": "/blue/image_raw"}]
    assert len(client.moves) == 1
    matrix, duration, blocking = client.moves[0]
    assert matrix.shape == (4, 4)
    assert np.array_equal(matrix, START_TRANSFORM)
    assert (duration, blocking) == (0.8, True)
    assert np.array_equal(observation.images["external_cam"], _raw_observation()["full_image"])
    assert observation.images["external_cam"].dtype == np.uint8
    assert np.array_equal(observation.state["eef_pose"], _raw_observation()["state"])
    assert observation.instruction == "pick up the cube"
    assert observation.image_times == {"external_cam": 0.0}
    assert observation.state_time == 0.0


def test_reset_can_skip_move_and_all_operator_io_when_unattended() -> None:
    client = _FakeClient()
    embodiment = _embodiment(client, WidowXConfig(move_to_start=False, unattended=True))
    embodiment.reset(_scene(), seed=42)
    assert client.moves == []
    assert client.events[:3] == ["init", "reset", "observe"]
    embodiment.reset(_scene())
    assert client.events.count("init") == 1
    assert client.events.count("reset") == 2


def test_reset_uses_full_advanced_transform_override() -> None:
    client = _FakeClient()
    override_matrix = np.eye(4)
    override_matrix[0, 3] = 0.4
    cfg = WidowXConfig(start_transform=tuple(override_matrix.reshape(-1)), unattended=True)
    _embodiment(client, cfg).reset(_scene())
    assert np.array_equal(client.moves[0][0], override_matrix)


@pytest.mark.parametrize("operation", ["init", "reset", "move"])
@pytest.mark.parametrize(
    ("status", "label"),
    [
        (Status.NO_CONNECTION, "NO_CONNECTION \\(0\\)"),
        (Status.EXECUTION_FAILURE, "EXECUTION_FAILURE \\(2\\)"),
        (Status.NOT_INITIALIZED, "NOT_INITIALIZED \\(3\\)"),
    ],
)
def test_reset_status_faults_name_every_non_success_code(
    operation: str, status: Status, label: str
) -> None:
    client = _FakeClient()
    client.statuses[operation] = status
    with pytest.raises(RuntimeError, match=rf"{operation}.*{label}"):
        _embodiment(client).reset(_scene())


@pytest.mark.parametrize(
    ("status", "label"),
    [
        (Status.NO_CONNECTION, "NO_CONNECTION \\(0\\)"),
        (Status.EXECUTION_FAILURE, "EXECUTION_FAILURE \\(2\\)"),
        (Status.NOT_INITIALIZED, "NOT_INITIALIZED \\(3\\)"),
    ],
)
def test_step_status_faults_name_every_non_success_code(status: Status, label: str) -> None:
    client = _FakeClient()
    embodiment = _embodiment(client)
    embodiment.reset(_scene())
    client.statuses["step_action"] = status
    with pytest.raises(RuntimeError, match=rf"step_action.*{label}"):
        embodiment.step(Action(data=np.zeros(7)))


def test_raw_integer_success_and_unknown_fault_code_are_safe() -> None:
    _require_success("raw", 1)
    with pytest.raises(RuntimeError, match="status 99"):
        _require_success("mystery", 99)


def test_observation_none_retries_then_succeeds_and_ignores_server_image() -> None:
    client = _FakeClient()
    client.observations = [None, None, _raw_observation()]
    clock = _Clock()
    embodiment = WidowXEmbodiment(
        WidowXConfig(unattended=True),
        client_factory=lambda _cfg: client,  # type: ignore[arg-type,return-value]
        clock=clock,
        sleep_fn=clock.sleep,
    )
    observation = embodiment.reset(_scene())
    assert clock.sleeps == pytest.approx([0.1, 0.1])
    assert observation.images["external_cam"].shape == (2, 3, 3)
    assert observation.images["external_cam"].shape != client.observations[0]["image"].shape  # type: ignore[index,union-attr]


def test_observation_none_times_out_even_with_a_frozen_clock() -> None:
    client = _FakeClient()
    client.observations = [None]
    sleeps: list[float] = []
    embodiment = WidowXEmbodiment(
        WidowXConfig(unattended=True, obs_timeout_s=0.2),
        client_factory=lambda _cfg: client,  # type: ignore[arg-type,return-value]
        clock=lambda: 0.0,
        sleep_fn=sleeps.append,
    )
    with pytest.raises(RuntimeError, match=r"timed out after 0\.2s"):
        embodiment.reset(_scene())
    assert sleeps == pytest.approx([0.1, 0.1])


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"state": np.zeros(7)}, "missing 'full_image'"),
        ({"full_image": np.zeros((2, 2, 3))}, "missing 'state'"),
        (
            {"full_image": np.zeros((2, 2)), "state": np.zeros(7)},
            "full_image must have shape",
        ),
        (
            {"full_image": np.zeros((2, 2, 3)), "state": np.zeros(6)},
            "expected a 7-D vector",
        ),
        (
            {"full_image": np.zeros((2, 2, 3)), "state": np.full(7, np.nan)},
            "state must contain only finite",
        ),
    ],
)
def test_observation_validation(raw: Mapping[str, Any], message: str) -> None:
    client = _FakeClient()
    client.observations = [raw]
    with pytest.raises(ValueError, match=message):
        _embodiment(client).reset(_scene())


def test_step_hard_clamps_passes_absolute_gripper_and_paces() -> None:
    client = _FakeClient()
    clock = _Clock()
    embodiment = WidowXEmbodiment(
        WidowXConfig(unattended=True),
        client_factory=lambda _cfg: client,  # type: ignore[arg-type,return-value]
        clock=clock,
        sleep_fn=clock.sleep,
    )
    embodiment.reset(_scene())
    result = embodiment.step(Action(data=np.asarray([1, -1, 1, -1, 1, -1, 0.73])))
    sent, blocking = client.actions[-1]
    assert sent == pytest.approx([0.05, -0.05, 0.05, -0.25, 0.25, -0.25, 0.73])
    assert blocking is False
    assert clock.sleeps == pytest.approx([0.2])
    assert result.terminated is False
    assert embodiment.num_steps == 1


def test_pacing_never_sleeps_negative_time() -> None:
    client = _FakeClient()
    clock = _Clock()
    embodiment = WidowXEmbodiment(
        WidowXConfig(unattended=True),
        client_factory=lambda _cfg: client,  # type: ignore[arg-type,return-value]
        clock=clock,
        sleep_fn=clock.sleep,
    )
    embodiment.reset(_scene())
    clock.now = 1.0
    embodiment.step(Action(data=np.zeros(7)))
    assert clock.sleeps == [0.0]


@pytest.mark.parametrize("bad", [np.full(7, np.nan), np.full(7, np.inf)])
def test_step_rejects_nonfinite_actions_without_sending(bad: np.ndarray) -> None:
    client = _FakeClient()
    embodiment = _embodiment(client)
    embodiment.reset(_scene())
    with pytest.raises(ValueError, match="finite"):
        embodiment.step(Action(data=bad))
    assert client.actions == []


@pytest.mark.parametrize(("answer", "reason"), [("yes", "success"), ("no", "failure")])
def test_operator_verdict_uses_termination_reason(answer: str, reason: str) -> None:
    client = _FakeClient()
    operator = OperatorIO(input_fn=lambda _prompt: answer, output_fn=lambda _line: None)
    embodiment = _embodiment(
        client,
        WidowXConfig(),
        operator=operator,
        poll_end=lambda: True,
    )
    embodiment.reset(_scene())
    result = embodiment.step(Action(data=np.zeros(7)))
    assert result.terminated is True
    assert result.termination_reason == reason
    assert result.info == {"operator_confirmed": answer == "yes"}


def test_unattended_skips_poll() -> None:
    client = _FakeClient()
    polled = 0

    def poll() -> bool:
        nonlocal polled
        polled += 1
        return True

    embodiment = _embodiment(client, poll_end=poll)
    embodiment.reset(_scene())
    assert embodiment.step(Action(data=np.zeros(7))).terminated is False
    assert polled == 0


def test_close_is_idempotent_and_clears_handle_even_on_stop_error() -> None:
    client = _FakeClient()
    embodiment = _embodiment(client)
    embodiment.reset(_scene())
    embodiment.close()
    embodiment.close()
    assert client.stop_calls == 1
    assert embodiment._client is None

    broken = _FakeClient()
    broken.stop_error = True
    embodiment2 = _embodiment(broken)
    embodiment2.reset(_scene())
    with pytest.raises(RuntimeError, match="stop failed"):
        embodiment2.close()
    embodiment2.close()
    assert broken.stop_calls == 1


def test_step_before_reset_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="before reset"):
        _embodiment(_FakeClient()).step(Action(data=np.zeros(7)))


def test_bind_task_runtime_requirements_device_slots_and_conformance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embodiment = WidowXEmbodiment()
    embodiment.bind_task(TaskEnvelope(name="task", max_steps=42))
    assert embodiment._bound_max_steps == 42
    assert embodiment._horizon_seconds() == pytest.approx(8.4)
    assert device_slots(WidowXEmbodiment) == ()
    monkeypatch.setattr("importlib.util.find_spec", lambda _name: None)
    assert missing_runtime_requirements(WidowXEmbodiment) == {
        "widowx_envs": WIDOWX_ENVS_INSTALL_COMMAND,
        "edgeml": EDGEML_INSTALL_COMMAND,
    }
    report = check_embodiment(embodiment.info)
    assert report.ok is True, report.summary()


def test_bound_horizon_appears_in_operator_status_and_close_clears_it() -> None:
    client = _FakeClient()
    output: list[str] = []
    embodiment = _embodiment(
        client,
        WidowXConfig(),
        operator=OperatorIO(input_fn=lambda _prompt: "", output_fn=output.append),
        poll_end=lambda: False,
    )
    embodiment.bind_task(TaskEnvelope(name="task", max_steps=30))
    embodiment.reset(_scene())
    assert output == ["Running: press Enter to end the episode, then y/N to score. Max 6s."]
    embodiment.close()
    assert embodiment._bound_max_steps is None


def test_close_without_connection_clears_bound_horizon() -> None:
    embodiment = WidowXEmbodiment()
    embodiment.bind_task(TaskEnvelope(name="task", max_steps=3))
    embodiment.close()
    assert embodiment._bound_max_steps is None


def test_nondefault_move_duration_reaches_env_params_and_pacing() -> None:
    client = _FakeClient()
    config = WidowXConfig(unattended=True, control_hz=4.0, move_duration=0.25)
    clock = _Clock()
    embodiment = WidowXEmbodiment(
        config,
        client_factory=lambda _cfg: client,  # type: ignore[arg-type,return-value]
        clock=clock,
        sleep_fn=clock.sleep,
    )
    embodiment.reset(_scene())
    assert client.env_params["move_duration"] == pytest.approx(0.25)
    clock.now += 0.05
    embodiment.step(Action(data=np.zeros(7)))
    assert clock.sleeps[-1] == pytest.approx(0.25 - 0.05)
