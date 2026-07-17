from __future__ import annotations

from typing import Any

import pytest
from inspect_robots.compat import check_compatibility
from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.registry import resolve
from inspect_robots.spaces import ActionSemantics, Box

from inspect_robots_widowx.config import action_box, observation_space
from inspect_robots_widowx.embodiment import WidowXEmbodiment
from inspect_robots_widowx.policy import OpenpiPolicy, OpenVLAPolicy


class _Policy:
    config = PolicyConfig()

    def __init__(self, info: PolicyInfo) -> None:
        self.info = info

    def reset(self, scene: object) -> None:
        return None

    def act(self, observation: object) -> Any:
        raise AssertionError("not called")


@pytest.mark.parametrize("policy", [OpenVLAPolicy(), OpenpiPolicy()])
def test_each_pair_has_zero_errors_and_zero_warnings(
    policy: OpenVLAPolicy | OpenpiPolicy,
) -> None:
    report = check_compatibility(policy, WidowXEmbodiment())
    assert report.ok is True
    assert report.errors == []
    assert report.warnings == []


@pytest.mark.parametrize("policy", [OpenVLAPolicy(), OpenpiPolicy()])
def test_builtin_cubepick_reach_is_realizable(
    policy: OpenVLAPolicy | OpenpiPolicy,
) -> None:
    task = resolve("task", "cubepick-reach")
    report = check_compatibility(policy, WidowXEmbodiment(), task)
    assert report.errors == []


def test_wrong_dimension_is_a_hard_error() -> None:
    info = PolicyInfo(
        name="wrong",
        action_space=Box(shape=(6,), semantics=ActionSemantics(control_mode="eef_delta_pose")),
    )
    report = check_compatibility(_Policy(info), WidowXEmbodiment())  # type: ignore[arg-type]
    assert any(issue.code == "action_dim" for issue in report.errors)


def test_advertised_policy_rate_warns() -> None:
    info = PolicyInfo(
        name="rated",
        action_space=action_box(),
        observation_space=observation_space(include_state=False),
        control_hz=10.0,
    )
    report = check_compatibility(_Policy(info), WidowXEmbodiment())  # type: ignore[arg-type]
    assert report.ok is True
    assert [issue.code for issue in report.warnings] == ["control_rate"]


def test_wrong_control_mode_is_a_hard_error() -> None:
    info = PolicyInfo(
        name="dishonest",
        action_space=Box(
            shape=(7,),
            semantics=ActionSemantics(
                control_mode="eef_abs_pose",
                rotation_repr="none",
                gripper="continuous",
                frame="base",
            ),
        ),
        observation_space=observation_space(include_state=False),
    )
    report = check_compatibility(_Policy(info), WidowXEmbodiment())  # type: ignore[arg-type]
    assert any(issue.code == "control_mode" for issue in report.errors)


def test_neither_policy_declares_state_keys() -> None:
    assert OpenVLAPolicy().info.observation_space.state_keys == frozenset()
    assert OpenpiPolicy().info.observation_space.state_keys == frozenset()
    assert WidowXEmbodiment().info.observation_space.state_keys == frozenset({"eef_pose"})
