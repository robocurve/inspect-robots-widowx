from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from inspect_robots import eval as robots_eval

from inspect_robots_widowx.config import WidowXConfig
from inspect_robots_widowx.embodiment import Status, WidowXEmbodiment
from inspect_robots_widowx.operator import OperatorIO
from inspect_robots_widowx.policy import OpenVLAPolicy


class _Client:
    def __init__(self) -> None:
        self.state = np.asarray([0.3, -0.09, 0.26, 0.0, 0.0, 0.0, 1.0])

    def init(self, env_params: Mapping[str, Any]) -> Status:
        return Status.SUCCESS

    def reset(self) -> Status:
        return Status.SUCCESS

    def move(self, pose_4x4: np.ndarray, duration: float, blocking: bool) -> Status:
        return Status.SUCCESS

    def step_action(self, action: np.ndarray, blocking: bool) -> Status:
        self.state = self.state.copy()
        self.state[-1] = action[-1]
        return Status.SUCCESS

    def get_observation(self) -> Mapping[str, Any]:
        return {
            "full_image": np.zeros((4, 4, 3), dtype=np.uint8),
            "state": self.state.copy(),
        }

    def stop(self) -> None:
        return None


def test_full_eval_propagates_success_and_policy_metadata() -> None:
    client = _Client()
    policy = OpenVLAPolicy(
        post_fn=lambda _url, _payload: np.asarray([0.0] * 6 + [0.73]),
        clock=lambda: 0.0,
    )
    embodiment = WidowXEmbodiment(
        WidowXConfig(),
        client_factory=lambda _cfg: client,
        operator=OperatorIO(input_fn=lambda _prompt: "yes", output_fn=lambda _line: None),
        poll_end=lambda: True,
        sleep_fn=lambda _delay: None,
        clock=lambda: 0.0,
    )
    logs = robots_eval("cubepick-reach", policy, embodiment, sinks=[], seed=0)
    assert len(logs) == 1
    log = logs[0]
    assert log.status == "success"
    assert log.results.metrics["success_at_end"] == 1.0
    assert log.eval.policy_config == {
        "action_horizon": 1,
        "replan_interval": None,
        "temperature": None,
    }
    assert embodiment.num_steps == 1
    assert client.state[-1] == 0.73
