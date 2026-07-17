from __future__ import annotations

import builtins
import sys
from types import ModuleType

import pytest

from inspect_robots_widowx._bridge import (
    EDGEML_INSTALL_COMMAND,
    WIDOWX_ENVS_INSTALL_COMMAND,
    _load_widowx_client,
)


def test_loader_returns_client_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    package = ModuleType("widowx_envs")
    package.__path__ = []  # type: ignore[attr-defined]
    service = ModuleType("widowx_envs.widowx_env_service")
    sentinel = object()
    service.WidowXClient = sentinel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "widowx_envs", package)
    monkeypatch.setitem(sys.modules, "widowx_envs.widowx_env_service", service)
    assert _load_widowx_client() is sentinel


@pytest.mark.parametrize("missing", ["widowx_envs", "widowx_envs.child", "edgeml", "edgeml.rpc"])
def test_loader_gives_both_pins_and_supply_chain_warning(
    monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    original_import = builtins.__import__

    def importing(name: str, *args: object, **kwargs: object) -> object:
        if name == "widowx_envs.widowx_env_service":
            raise ModuleNotFoundError("missing", name=missing)
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "widowx_envs", raising=False)
    monkeypatch.delitem(sys.modules, "widowx_envs.widowx_env_service", raising=False)
    monkeypatch.setattr(builtins, "__import__", importing)
    with pytest.raises(ModuleNotFoundError, match="unrelated Microsoft") as caught:
        _load_widowx_client()
    message = str(caught.value)
    assert EDGEML_INSTALL_COMMAND in message
    assert WIDOWX_ENVS_INSTALL_COMMAND in message


def test_loader_does_not_mask_nested_dependency_error(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def importing(name: str, *args: object, **kwargs: object) -> object:
        if name == "widowx_envs.widowx_env_service":
            raise ModuleNotFoundError("nested", name="zmq")
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "widowx_envs", raising=False)
    monkeypatch.setattr(builtins, "__import__", importing)
    with pytest.raises(ModuleNotFoundError) as caught:
        _load_widowx_client()
    assert caught.value.name == "zmq"
