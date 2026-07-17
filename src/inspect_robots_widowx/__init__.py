"""Inspect Robots adapters for WidowX 250S and Bridge-compatible policies."""

from __future__ import annotations

from inspect_robots_widowx.config import OpenpiConfig, OpenVLAConfig, WidowXConfig
from inspect_robots_widowx.embodiment import WidowXEmbodiment
from inspect_robots_widowx.operator import OperatorIO
from inspect_robots_widowx.packing import DIM_LABELS, STATE_KEY, TOTAL_DIM
from inspect_robots_widowx.policy import OpenpiPolicy, OpenVLAPolicy
from inspect_robots_widowx.preflight import build, run_preflight

try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("inspect-robots-widowx")
except PackageNotFoundError:  # pragma: no cover - only in a non-installed source tree
    __version__ = "0.0.0+unknown"

__all__ = [  # noqa: RUF022 - public order is pinned by the accepted design
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
