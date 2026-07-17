# 0001: WidowX embodiment + OpenVLA/openpi policy plugin

Status: draft (critique loop in progress)
Issue: #1

## Goal

Ship the WidowX 250S sibling of inspect-robots-franka/-yam/-so101: a plugin
registering a `widowx` embodiment (real WidowX 250S through the BridgeData
`widowx_envs` ZMQ server-client) and two policy clients, `openvla` (REST
`/act` json-numpy servers, the de facto BridgeData V2 eval stack) and
`openpi` (websocket, bring-your-own fine-tune), all declaring one shared
7-D `eef_delta_pose` contract so `check_compatibility` passes with zero
errors and zero warnings.

Reference material: session scratchpad `widowx-research.md` (stack research
2026-07-17) and `franka/framework-contract.md`. Templates:
../inspect-robots-franka (freshest proven scaffolding),
../inspect-robots-yam (HTTP client policy pattern).

## Stack decision (from the research)

- **Driver: BridgeData `WidowXClient`** (rail-berkeley/bridge_data_robot).
  The robot host runs `widowx_env_service --server` inside the frozen
  ROS1-Noetic Docker image; our adapter is a plain-Python ZMQ client
  (edgeml ActionClient, req/rep :5556 + broadcast :5557). This is the
  interface every relevant policy eval (OpenVLA, Octo, openvla-mini) uses,
  and it keeps the EOLed interbotix stack contained in Docker on the robot
  host.
  - **Supply-chain trap: the PyPI name `edgeml` is an unrelated Microsoft
    package.** Both `edgeml` and `widowx_envs` are git-only:
    `pip install "edgeml @ git+https://github.com/youliangtan/edgeml"` and
    `pip install "widowx_envs @
    git+https://github.com/rail-berkeley/bridge_data_robot#subdirectory=widowx_envs"`.
    No PyPI extras; a `_bridge.py` guided lazy loader (yam `_i2rt.py`
    pattern) carries both install commands. Both repos are frozen since
    2024: the README says so and pins exact commits in the install
    commands (implementer selects current HEADs and records them).
  - Hardware status: Trossen discontinued the 250S (July 2025 EOL). The
    README states this package serves existing rigs; the successor WidowX
    AI has a different stack and is out of scope.
- **Policies:**
  - `openvla`: client for OpenVLA's first-party `vla-scripts/deploy.py`
    REST server: `POST /act` with a json-numpy payload
    `{"image": HxWx3 uint8, "instruction": str, "unnorm_key":
    "bridge_orig"}`, response = ONE 7-D action (no chunking:
    `action_horizon=1`, `replan_interval=None`). Transport via `requests`
    + `json_numpy` (both already house deps in yam; lazily imported).
  - `openpi`: websocket client (git-only openpi-client, guided install,
    signature-asserting seam CI job), for community pi0-Bridge fine-tunes;
    README notes there is no official Bridge checkpoint. Emits chunks;
    `action_horizon` config default 10, pass-through semantics (no
    velocity integration, no polarity flip: Bridge convention already
    matches the house 1=open).
- **Out of scope for v1**: joint-space control (interbotix ROS 2 or raw
  dynamixel backends), SimplerEnv sim backend, Octo/MiniVLA in-process
  policies, WidowX AI / Trossen AI arms, multi-camera rigs (dataset had
  them; the canonical eval is single-camera).

## The 7-D contract (BridgeData V2 conventions)

- `DIM_LABELS = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")`;
  `TOTAL_DIM=7`, `GRIPPER_IDX=6`, `STATE_KEY="eef_pose"`.
- `ActionSemantics(control_mode="eef_delta_pose",
  rotation_repr="euler_xyz", gripper="continuous", frame="base",
  dim_labels=DIM_LABELS)`. Slots 0-2 are EEF displacement in metres, 3-5
  are roll/pitch/yaw displacement in radians, slot 6 is the ABSOLUTE
  gripper target in [0, 1] with 1 = open (BridgeData wire convention,
  which happens to match the house convention: no polarity conversion
  anywhere; a test locks that policies pass the gripper through
  untouched). The absolute-gripper-inside-a-delta-action wrinkle is
  documented in config.py and README: the framework's ActionSemantics has
  no per-dim delta/absolute split, so the gripper slot's absoluteness is a
  documented convention, exactly as BridgeData defines it.
- Action bounds: per-step displacement box `delta_low/high` (default
  +-0.05 m translation, +-0.25 rad rotation per step: BridgeData-typical
  magnitudes, config-overridable) and gripper [0, 1]. Finite bounds keep
  DeltaLimitApprover constructible. README notes the CLI's default
  DeltaLimitApprover in displacement mode also clamps the absolute gripper
  slot to a 0.05 step (20-step transitions) and gives the Python per-dim
  `max_delta` snippet (franka's documented pattern).
- Observation: camera `external_cam` (single over-the-shoulder RGB,
  640x480 native: the canonical Bridge eval view), state
  `StateField(key="eef_pose", shape=(8,), unit="m+quat+normalized")` =
  xyz + quaternion (xyzw, as the server reports) + gripper. Conformance's
  exactly-one-proprio-field rule applies only to absolute control modes,
  so an 8-D state field with a 7-D delta action is legal; the compat
  zero/zero property only needs the policies to request no state keys the
  embodiment lacks (OpenVLA and openpi Bridge fine-tunes are image+text:
  they declare NO state keys, and a test locks that).
- `control_hz=5.0` (BridgeData convention). The WidowX server executes
  each action over `move_duration=0.2` s; `step()` calls
  `step_action(..., blocking=False)` then paces to the 5 Hz period with
  the injected clock/sleep (SELF_PACED declared), so cadence stays honest
  even if the server call returns early. Policy `control_hz=None`
  (zero-warnings property).
- Default reset pose: the server-side neutral (WidowXClient.reset());
  optionally the OpenVLA eval start pose via config
  `start_eef_pos`/`start_eef_quat` (defaults: the documented
  `[0.3, -0.09, 0.26]` / `[0, -0.259, 0, -0.966]`), applied with a `move`
  call after reset when `move_to_start=True` (default).

## Package layout

```
inspect-robots-widowx/
├── src/inspect_robots_widowx/
│   ├── __init__.py / CLAUDE.md / py.typed
│   ├── packing.py         # 7-D constants, validate_dim, accessors
│   ├── config.py          # WidowXConfig, OpenVLAConfig, OpenpiConfig, shared builders
│   ├── embodiment.py      # WidowXEmbodiment + Client protocol seam
│   ├── policy.py          # OpenVLAPolicy (REST) + OpenpiPolicy (websocket)
│   ├── operator.py        # OperatorIO (yam's EOF-hardened version)
│   ├── preflight.py       # inspect-robots-widowx-preflight CLI
│   └── _bridge.py         # lazy widowx_envs/edgeml loader + install commands
├── tests/                 # franka-style battery (see test plan)
├── plans/0001-widowx-openvla-design.md
├── .github/workflows/{ci,canary,release}.yml
├── .pre-commit-config.yaml / .env.example / CITATION.cff
├── pyproject.toml / uv.lock / README.md / CLAUDE.md / LICENSE / .gitignore
```

## Module contracts

### packing.py (pure)

Constants above; `validate_dim(vec)` (ndim==1, length 7, strict);
`displacement(vec) -> (6,)` and `gripper(vec) -> float` accessors. No wire
converters (the client takes the 7-vector verbatim).

### config.py

- `_FromKwargs` + `_FLOAT_TUPLE_FIELDS` (house pattern).
- `WidowXConfig` (frozen): `host="localhost"`, `port=5556`,
  `control_hz=5.0`, `move_duration=0.2` (forwarded to the server env
  params; must equal `1/control_hz`, validated), `delta_low/delta_high`
  (defaults above; length 7 with the gripper slot [0,1]),
  `image_size=256` (server-side resize request), `move_to_start=True`,
  `start_eef_pos`, `start_eef_quat`, `unattended=False`, `docs_extra=""`.
  `__post_init__` validates ranges/ordering/duration-consistency.
- `OpenVLAConfig` (frozen): `server_url="http://127.0.0.1:8000"`,
  `endpoint="/act"`, `unnorm_key="bridge_orig"`, `timeout_s=30.0`,
  `name="openvla"`. `.url` property; `from_kwargs` rejects `url`.
  `action_horizon` fixed at 1 (not configurable: the server returns one
  action; PolicyConfig(action_horizon=1, replan_interval=None)).
- `OpenpiConfig` (frozen): `host`, `port=8000`, `api_key=None`,
  `action_horizon=10`, `replan_interval=5`, `name="openpi"`,
  `resize_px=224`. Explicit PolicyConfig wiring; api_key never in
  asdict(policy.config) (franka pattern, tested).
- Shared builders: `ACTION_SEMANTICS`, `action_box()` (from
  delta_low/high), `observation_space()` (external_cam + eef_pose(8,)).
  Both policies and the embodiment build from these.

### embodiment.py

- `Client` Protocol (runtime_checkable), injected via `client_factory`:
  `init(env_params: Mapping) -> None`, `reset() -> None`,
  `move(pose: np.ndarray, duration: float) -> None`,
  `step_action(action: np.ndarray, blocking: bool) -> None`,
  `get_observation() -> Mapping | None` (server returns None until ready:
  the embodiment retries with injected sleep up to `obs_timeout_s=10`,
  then raises EmbodimentFault-compatible RuntimeError),
  `stop() -> None`.
- `_default_client_factory` (pragma'd): builds `WidowXClient(host, port)`
  through `_bridge.py`'s guided loader; env params carry
  `move_duration`, workspace defaults, and the camera topic left to the
  server's own config.
- `WidowXEmbodiment`: inert `__init__(config=None, *, client_factory=None,
  operator=None, poll_end=None, clock=None, sleep_fn=None, **flat)`;
  lazy connect at first `reset()`: init client, `client.reset()`,
  optional `move` to the start pose, operator `wait_ready()` (skipped
  when unattended), first observation.
  - Observation adaptation: server obs dict -> Observation(images
    {"external_cam": full_image as uint8 HxWx3}, state {"eef_pose":
    8-vec}); JPEG-compressed `full_image` is decoded by the client
    library itself (verify at implementation; if bytes arrive, decode via
    lazily imported cv2 and add cv2 to RUNTIME_REQUIREMENTS).
  - `step()`: `validate_dim` -> clamp to delta_low/high (hard backstop,
    independent of Approver; NaN rejected) -> `step_action(clamped,
    blocking=False)` -> pace to 1/control_hz -> observe -> `poll_end()` /
    `confirm_success()` -> StepResult (success only via
    termination_reason="success").
  - `close()`: idempotent; `stop()` always attempted; handle cleared.
  - `RUNTIME_REQUIREMENTS: ClassVar[Mapping[str, str]]` = widowx_envs and
    edgeml mapped to their git install commands. `DEVICE_SLOTS`: none
    (hardware lives behind the server; README documents the server
    setup). `bind_task()` + `EmbodimentInfo.docs` with all 7 dim labels
    (franka pattern).

### policy.py

- `OpenVLAPolicy(config=None, *, post_fn=None, clock=None, **flat)`, entry
  point `openvla`. `act()`: require `external_cam` (helpful error; no
  state keys consumed) -> payload `{"image": uint8 array (passed at
  native size; the SERVER does its own resize/crop handling per OpenVLA
  deploy conventions: verify at implementation whether client-side 256x256
  resize is required and pin the answer in config docstrings; if
  client-side resize is needed, lazy cv2 in the default transport only)
  , "instruction": str, "unnorm_key": cfg.unnorm_key}` ->
  `post_fn(url, payload) -> 7-element array` -> validate shape/finiteness
  -> single-Action chunk, `ActionChunk(control_hz=5.0,
  inference_latency_s=measured)`. `_default_post` (pragma'd):
  `requests.post(json=json_numpy-encoded payload)` with the documented
  double-encoding fallback from OpenVLA's deploy.py README.
- `OpenpiPolicy`: franka's shape at 7-D: `infer_fn` seam, DROID-free
  pass-through (no integration, no flip), git-only client guidance,
  resize-with-pad in default transport.
- Both: `info.control_hz=None`; explicit PolicyConfig; `num_inferences`;
  `reset()` stashes instruction.

### operator.py / preflight.py / _bridge.py

- operator.py: yam's hardened version, renamed.
- preflight.py: house pattern incl. `dry_run` key in `--json` output
  (fixes franka's carried nit). Console script
  `inspect-robots-widowx-preflight`.
- `_bridge.py`: `_load_widowx_client()` guided loader with BOTH git
  install commands (edgeml trap called out in the error message).

### __init__.py public API (pinned by test_api_snapshot.py)

`__all__` = `WidowXConfig`, `OpenVLAConfig`, `OpenpiConfig`,
`WidowXEmbodiment`, `OpenVLAPolicy`, `OpenpiPolicy`, `OperatorIO`,
`STATE_KEY`, `TOTAL_DIM`, `DIM_LABELS`, `build`, `run_preflight`,
`__version__`.

## pyproject

- Base deps: `inspect-robots>=0.12`, `numpy>=1.24`, `requests>=2.31`,
  `json-numpy>=2.0` (lazily imported; yam precedent). No extras for the
  git-only stacks (guided installs).
- dev extra: pytest, pytest-cov, ruff, mypy, pre-commit, numpy<2.5.
- Entry points: embodiment `widowx = ...:WidowXEmbodiment`; policies
  `openvla = ...:OpenVLAPolicy`, `openpi = ...:OpenpiPolicy`. Console
  script `inspect-robots-widowx-preflight`.
- mypy overrides: `requests.*`, `json_numpy.*`, `cv2.*`,
  `openpi_client.*`, `widowx_envs.*`, `edgeml.*`.
- Everything else identical to franka (hatch-vcs, fancy readme, ruff D1,
  coverage 100 branch).

## CI

franka's skeleton: `quality`, `test` (ubuntu+macos x py3.11/3.12),
`import-hygiene` (--no-deps + locked pins; assert `requests`,
`json_numpy`, `cv2`, `widowx_envs`, `edgeml`, `openpi_client`,
`websockets`, `torch` absent), `openpi-seam` (franka's
signature-asserting job verbatim), `ci-ok` needing all four,
`alert-red-main`; canary.yml + release.yml byte-copied. Ruleset already
active.

## Test plan (franka battery adapted; all seams injected)

- test_packing.py: constants/labels/validate_dim/accessors.
- test_config.py: from_kwargs rejection, tuple parsing, validation
  (duration-vs-hz consistency, delta bounds ordering, url rejection).
- test_embodiment.py: inert init; lazy connect; reset flow (init ->
  reset -> optional move-to-start with exact pose args -> wait_ready);
  obs-None retry then timeout fault; clamp backstop (out-of-bounds and
  NaN); blocking=False forwarded; pacing with injected clock; operator
  success/failure; unattended; close idempotency + stop-on-error;
  bind_task; docs labels; RUNTIME_REQUIREMENTS Mapping via conformance.
- test_policy.py: openvla payload keys byte-exact incl. unnorm_key;
  single-action chunk; shape/finiteness validation; gripper pass-through
  (asymmetric value, locks no-polarity-flip); openpi pass-through at 7-D;
  PolicyConfig wiring (openvla horizon 1/replan None; openpi 10/5; no
  api_key in asdict); instruction threading; num_inferences; helpful
  errors on missing camera.
- test_operator.py, test_preflight.py (incl. dry_run in --json),
  test_bridge.py (loader messages carry both git commands + edgeml trap).
- test_compat.py: zero/zero for BOTH policies vs embodiment;
  cubepick-reach realizable; wrong-dim negative; control_hz-advertising
  negative; a test locking that neither policy declares state keys.
- test_embodiment_docs.py, test_api_snapshot.py,
  test_eval_end_to_end.py (fake client + fake post_fn through eval()).

## README (yam structure; house style)

Sections: badges/intro, Install (client machine: pip package; robot host:
bridge_data_robot Docker server + udev note; GPU machine: OpenVLA
deploy.py serve command), Preflight, Run on hardware (config.ini),
Safety (clamp backstop, absolute-gripper-in-delta-action wrinkle,
DeltaLimitApprover gripper note + per-dim snippet, workspace boundaries
live server-side, discontinued-hardware notice, first-run verification),
Configuration (field tables, 7-D unit table), Development, Citation,
License.

## Sequencing

1. Critique loop until clean.
2. Codex implements on feat/widowx-plugin; `uv lock` before first push;
   Fable reviews the diff.
3. Push; PR (Closes #1) green; fresh-eyes review loop; merge.
4. Post-merge: PyPI pending publisher (owner action).
