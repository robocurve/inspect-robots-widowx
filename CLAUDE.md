# inspect-robots-widowx agent guide

Inspect Robots adapters for existing WidowX 250S rigs driven through the
BridgeData server and OpenVLA or openpi policy servers. The framework lives in
[inspect-robots](https://github.com/robocurve/inspect-robots).

## The one big idea

Inspect Robots swaps a policy and an embodiment. This package ships both sides:

- `openvla` calls the first-party OpenVLA `/act` endpoint.
- `openpi` calls an openpi websocket server for a Bridge fine-tune.
- `widowx` calls the BridgeData `WidowXClient` over ZMQ.

All three declare the same 7-D `eef_delta_pose` contract. The first six slots
are base-frame xyz and roll/pitch/yaw displacements. The gripper slot is an
absolute target with 0 closed and 1 open.

## Layout

- `src/inspect_robots_widowx/` contains package modules and a local module map.
- `tests/` contains fully injected hardware-free tests.
- `plans/0001-widowx-openvla-design.md` is the accepted binding design.

## Working here

- Set `UV_CACHE_DIR=$PWD/.uv-cache` for every uv command in this workspace.
- Install with `uv sync --extra dev`.
- Run `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`, and
  `uv run pytest --cov` before handing off.
- Keep strict mypy and 100 percent statement and branch coverage.
- Keep requests, json-numpy, Pillow, openpi, edgeml, and widowx_envs imports lazy.

## Safety invariants

- `WidowXEmbodiment.step()` rejects NaN and always clamps to configured limits.
- Every non-success client status raises and names the returned status value.
- Reset uses the named reference 4x4 rotation matrix and a blocking move.
- No code derives the start rotation from a quaternion.
- Policies preserve the absolute gripper slot without polarity conversion.
- Construction performs no hardware, network, image, or stdin work.
- Success reaches scoring only as `termination_reason="success"`.
- The embodiment declares `SELF_PACED` and sleeps inside `step()`.

## CI and releases

- CI installs from `uv.lock`. Run `uv lock` after dependency changes.
- `ci-ok` must need every blocking job.
- The two BridgeData dependencies are installed only from their pinned git SHAs.
- Versions come from git tags through hatch-vcs.

## Writing style

- Do not use em dashes in prose.
- Do not use decorative emoji, slogans, or "not just X, but Y".
- Use plain headers without trailing colons.
