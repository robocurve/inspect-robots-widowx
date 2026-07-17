from __future__ import annotations

import re

import inspect_robots_widowx

EXPECTED_API = [
    "WidowXConfig",
    "OpenVLAConfig",
    "OpenpiConfig",
    "WidowXEmbodiment",
    "OpenVLAPolicy",
    "OpenpiPolicy",
    "OperatorIO",
    "STATE_KEY",
    "TOTAL_DIM",
    "DIM_LABELS",
    "build",
    "run_preflight",
    "__version__",
]


def test_public_api_is_exact_ordered_and_importable() -> None:
    assert inspect_robots_widowx.__all__ == EXPECTED_API
    for name in inspect_robots_widowx.__all__:
        assert hasattr(inspect_robots_widowx, name)


def test_version_is_tag_derived_shape() -> None:
    assert re.match(r"\d+\.\d+", inspect_robots_widowx.__version__)


def test_entry_points_resolve() -> None:
    from inspect_robots.registry import resolve

    assert resolve("policy", "openvla").info.name == "openvla"
    assert resolve("policy", "openpi").info.name == "openpi"
    assert resolve("embodiment", "widowx").info.name == "widowx"
