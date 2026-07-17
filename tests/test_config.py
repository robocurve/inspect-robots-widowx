from __future__ import annotations

import numpy as np
import pytest

from inspect_robots_widowx.config import (
    ACTION_SEMANTICS,
    DEFAULT_DELTA_HIGH,
    DEFAULT_DELTA_LOW,
    DEFAULT_START_EEF_POS,
    OpenpiConfig,
    OpenVLAConfig,
    WidowXConfig,
    action_box,
    observation_space,
)
from inspect_robots_widowx.packing import DIM_LABELS, STATE_KEY, TOTAL_DIM


def test_widowx_pure_defaults_construct_and_match_bridge_contract() -> None:
    cfg = WidowXConfig()
    assert (cfg.host, cfg.port) == ("localhost", 5556)
    assert (cfg.control_hz, cfg.move_duration, cfg.obs_timeout_s) == (5.0, 0.2, 10.0)
    assert cfg.delta_low == DEFAULT_DELTA_LOW
    assert cfg.delta_high == DEFAULT_DELTA_HIGH
    assert cfg.start_eef_pos == DEFAULT_START_EEF_POS
    assert cfg.start_transform is None
    assert cfg.move_to_start is True
    assert cfg.low.shape == cfg.high.shape == (TOTAL_DIM,)
    assert (cfg.low[-1], cfg.high[-1]) == (0.0, 1.0)


def test_policy_defaults_and_url() -> None:
    openvla = OpenVLAConfig()
    assert openvla.url == "http://127.0.0.1:8000/act"
    assert (openvla.resize_px, openvla.jpeg_roundtrip) == (256, True)
    assert OpenVLAConfig(server_url="http://host/").url == "http://host/act"
    openpi = OpenpiConfig()
    assert (openpi.host, openpi.port) == ("127.0.0.1", 8000)
    assert (openpi.action_horizon, openpi.replan_interval, openpi.resize_px) == (10, 5, 224)


def test_from_kwargs_rejects_unknown_url_and_parses_tuple_fields() -> None:
    with pytest.raises(TypeError, match="unexpected config keys"):
        OpenVLAConfig.from_kwargs(url="http://wrong")
    cfg = WidowXConfig.from_kwargs(
        delta_low=",".join(str(value) for value in DEFAULT_DELTA_LOW),
        delta_high=DEFAULT_DELTA_HIGH,
        start_eef_pos="0.3,-0.09,0.26",
    )
    assert cfg.delta_low == pytest.approx(DEFAULT_DELTA_LOW)
    assert cfg.start_eef_pos == pytest.approx(DEFAULT_START_EEF_POS)
    with pytest.raises(ValueError, match="start_eef_pos must be a comma-separated"):
        WidowXConfig.from_kwargs(start_eef_pos="0,bad,1")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"host": ""}, "host"),
        ({"port": 0}, "port"),
        ({"port": True}, "port"),
        ({"control_hz": 0.0}, "control_hz"),
        ({"move_duration": np.inf}, "move_duration"),
        ({"obs_timeout_s": 0.0}, "obs_timeout_s"),
        ({"control_hz": 10.0}, "equal 1/control_hz"),
        ({"delta_low": (0.0,) * 6}, "must have 7"),
        ({"delta_high": (0.0,) * 6}, "must have 7"),
        ({"delta_low": (*DEFAULT_DELTA_LOW[:-1], np.nan)}, "only finite"),
        ({"delta_low": DEFAULT_DELTA_HIGH}, "must be below"),
        ({"delta_low": (*DEFAULT_DELTA_LOW[:-1], -0.1)}, "exactly \\[0, 1\\]"),
        ({"delta_high": (*DEFAULT_DELTA_HIGH[:-1], 0.9)}, "exactly \\[0, 1\\]"),
        ({"start_eef_pos": (0.0, 1.0)}, "3 finite"),
        ({"start_eef_pos": (0.0, np.nan, 1.0)}, "3 finite"),
        ({"start_transform": (0.0,) * 15}, "16 finite"),
        ({"start_transform": (*((0.0,) * 15), np.nan)}, "16 finite"),
        ({"start_transform": (*tuple(np.eye(4).reshape(-1)[:-1]), 2.0)}, "homogeneous"),
    ],
)
def test_widowx_validation(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        WidowXConfig(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"server_url": ""}, "server_url"),
        ({"endpoint": "act"}, "endpoint"),
        ({"unnorm_key": ""}, "unnorm_key"),
        ({"name": ""}, "name"),
        ({"timeout_s": 0.0}, "timeout_s"),
        ({"resize_px": True}, "resize_px"),
        ({"jpeg_roundtrip": 1}, "jpeg_roundtrip"),
    ],
)
def test_openvla_validation(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        OpenVLAConfig(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"host": ""}, "host"),
        ({"port": 65536}, "port"),
        ({"name": ""}, "name"),
        ({"action_horizon": 0}, "action_horizon"),
        ({"replan_interval": True}, "replan_interval"),
        ({"resize_px": 0}, "resize_px"),
    ],
)
def test_openpi_validation(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        OpenpiConfig(**kwargs)


def test_shared_spaces_have_pinned_semantics_camera_and_optional_state() -> None:
    cfg = WidowXConfig()
    box = action_box(cfg)
    assert box.shape == (7,)
    assert np.array_equal(box.low, cfg.low)
    assert np.array_equal(box.high, cfg.high)
    assert box.semantics is ACTION_SEMANTICS
    assert ACTION_SEMANTICS.control_mode == "eef_delta_pose"
    assert ACTION_SEMANTICS.rotation_repr == "none"
    assert ACTION_SEMANTICS.gripper == "continuous"
    assert ACTION_SEMANTICS.frame == "base"
    assert ACTION_SEMANTICS.dim_labels == DIM_LABELS
    assert action_box().low is not None
    with_state = observation_space(include_state=True)
    assert with_state.camera_names == frozenset({"external_cam"})
    assert with_state.cameras[0].height == 480
    assert with_state.cameras[0].width == 640
    assert with_state.state_keys == frozenset({STATE_KEY})
    assert with_state.state is not None
    assert with_state.state.fields[0].shape == (7,)
    assert with_state.state.fields[0].unit == "m+rad+normalized"
    without_state = observation_space(include_state=False)
    assert without_state.state is None
    assert without_state.state_keys == frozenset()
