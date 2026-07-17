"""Load the optional BridgeData WidowX client with safe install guidance."""

from __future__ import annotations

from typing import Any

EDGEML_INSTALL_COMMAND = (
    'pip install "edgeml @ git+https://github.com/youliangtan/edgeml'
    '@b4b8495b489e7c973187742d2f2fe9aa016d9aca"'
)
WIDOWX_ENVS_INSTALL_COMMAND = (
    'pip install "widowx_envs @ '
    "git+https://github.com/rail-berkeley/bridge_data_robot"
    "@b841131ecd512bafb303075bd8f8b677e0bf9f1f#subdirectory=widowx_envs"
    '"'
)


def _load_widowx_client() -> Any:
    """Import the git-only client or explain both pinned dependencies."""
    try:
        from widowx_envs.widowx_env_service import WidowXClient
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if (
            missing != "widowx_envs"
            and not missing.startswith("widowx_envs.")
            and not (missing == "edgeml" or missing.startswith("edgeml."))
        ):
            raise
        raise ModuleNotFoundError(
            "The BridgeData WidowX client requires two git-only packages. Install them with "
            f"`{EDGEML_INSTALL_COMMAND}` and `{WIDOWX_ENVS_INSTALL_COMMAND}`. The PyPI "
            "package named `edgeml` is an unrelated Microsoft package; do not install it "
            "for this adapter.",
            name=exc.name,
        ) from exc
    return WidowXClient
