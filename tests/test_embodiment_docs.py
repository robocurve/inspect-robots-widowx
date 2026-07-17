from __future__ import annotations

from inspect_robots_widowx.config import DEFAULT_DELTA_HIGH, DEFAULT_DELTA_LOW, WidowXConfig
from inspect_robots_widowx.embodiment import _DOCS, WidowXEmbodiment
from inspect_robots_widowx.packing import DIM_LABELS


def test_docs_name_every_dimension_exactly_once_in_bullets() -> None:
    docs = WidowXEmbodiment().info.docs
    assert docs is not None
    bullets = [line for line in docs.splitlines() if line.startswith("- ")]
    for label in DIM_LABELS:
        assert sum(line.startswith(f"- {label}:") for line in bullets) == 1


def test_docs_do_not_leak_numeric_action_bounds() -> None:
    docs = WidowXEmbodiment().info.docs or ""
    for value in (*DEFAULT_DELTA_LOW, *DEFAULT_DELTA_HIGH):
        assert str(value) not in docs


def test_docs_extra_is_stripped_and_appended_once() -> None:
    embodiment = WidowXEmbodiment(WidowXConfig(docs_extra="  rig note {safe}\n"))
    assert embodiment.info.docs == _DOCS + "\n\nrig note {safe}"
    assert WidowXEmbodiment(WidowXConfig(docs_extra=" \n ")).info.docs == _DOCS
