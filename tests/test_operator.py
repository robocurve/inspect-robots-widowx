import sys
import types

import pytest
from inspect_robots.errors import EmbodimentFault

from inspect_robots_widowx.operator import OperatorIO, _drain_stdin, default_poll_end


def test_wait_ready_reads_prompt() -> None:
    seen: list[str] = []
    OperatorIO(input_fn=lambda prompt: seen.append(prompt) or "").wait_ready("ready?")
    assert seen == ["ready?"]


@pytest.mark.parametrize("exception", [EOFError, OSError])
def test_wait_ready_turns_dead_stdin_into_embodiment_fault(
    exception: type[Exception],
) -> None:
    def input_fn(_prompt: str) -> str:
        raise exception("closed")

    with pytest.raises(EmbodimentFault, match=r"WidowXConfig\(unattended=True\)"):
        OperatorIO(input_fn=input_fn).wait_ready()


@pytest.mark.parametrize("answer", ["y", "Yes", "1", "TRUE", "success", "pass"])
def test_confirm_success_affirmative(answer: str) -> None:
    assert OperatorIO(input_fn=lambda _prompt: answer).confirm_success() is True


@pytest.mark.parametrize("answer", ["n", "no", "", "nope"])
def test_confirm_success_negative(answer: str) -> None:
    assert OperatorIO(input_fn=lambda _prompt: answer).confirm_success() is False


def test_default_poll_is_exposed() -> None:
    assert callable(default_poll_end)


def test_win32_drain_stdin_consumes_keyboard_buffer(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    fake_msvcrt = types.ModuleType("msvcrt")
    fake_msvcrt.kbhit = lambda: len(calls) < 2  # type: ignore[attr-defined]
    fake_msvcrt.getwch = lambda: calls.append("key") or "a"  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    _drain_stdin()
    assert len(calls) == 2


def test_win32_default_poll_end_detects_enter(monkeypatch: pytest.MonkeyPatch) -> None:
    keys = ["a", "\r"]
    fake_msvcrt = types.ModuleType("msvcrt")
    fake_msvcrt.kbhit = lambda: bool(keys)  # type: ignore[attr-defined]
    fake_msvcrt.getwch = lambda: keys.pop(0)  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    assert default_poll_end() is True


def test_win32_default_poll_end_returns_false_without_enter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keys = ["a", "b"]
    fake_msvcrt = types.ModuleType("msvcrt")
    fake_msvcrt.kbhit = lambda: bool(keys)  # type: ignore[attr-defined]
    fake_msvcrt.getwch = lambda: keys.pop(0)  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    assert default_poll_end() is False
