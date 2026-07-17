<div align="center">

# inspect-robots-widowx

Run [Inspect Robots](https://github.com/robocurve/inspect-robots) evals on
existing WidowX 250S arms with
[OpenVLA](https://github.com/openvla/openvla) or
[openpi](https://github.com/Physical-Intelligence/openpi) policy servers.

![Status: alpha](https://img.shields.io/badge/status-alpha-blue)
[![CI](https://github.com/robocurve/inspect-robots-widowx/actions/workflows/ci.yml/badge.svg)](https://github.com/robocurve/inspect-robots-widowx/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/inspect-robots-widowx)](https://pypi.org/project/inspect-robots-widowx/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)](https://github.com/robocurve/inspect-robots-widowx/actions/workflows/ci.yml)
[![Built on Inspect Robots](https://img.shields.io/badge/built%20on-Inspect%20Robots-indigo)](https://github.com/robocurve/inspect-robots)

</div>

> [!NOTE]
> Trossen discontinued the WidowX 250S in July 2025. This package supports
> existing 250S rigs. The successor WidowX AI uses a different stack and is
> outside this package's scope.

Inspect Robots has two swappable inputs, a `Policy` and an `Embodiment`. This
package provides three registered components:

- **`openvla` policy:** a REST client for OpenVLA's first-party `/act` server.
- **`openpi` policy:** a websocket client for community Bridge fine-tunes.
- **`widowx` embodiment:** a client for the BridgeData WidowX robot server.

Every pairing uses the same 7-D `eef_delta_pose` contract. Slots 0 through 5
are xyz and roll/pitch/yaw displacements. Slot 6 is an absolute normalized
gripper target, where 0 is closed and 1 is open. Both policy pairings pass
compatibility with zero errors and zero warnings.

## Install

### Client machine

Create an environment and install the package:

```bash
uv venv && source .venv/bin/activate
uv pip install inspect-robots-widowx
uv pip install "edgeml @ git+https://github.com/youliangtan/edgeml@b4b8495b489e7c973187742d2f2fe9aa016d9aca"
uv pip install "widowx_envs @ git+https://github.com/rail-berkeley/bridge_data_robot@b841131ecd512bafb303075bd8f8b677e0bf9f1f#subdirectory=widowx_envs"
```

> [!CAUTION]
> The PyPI project named `edgeml` is an unrelated Microsoft package. Do not
> install it for this adapter. Use the pinned git command above.

Both BridgeData repositories are frozen and have seen no upstream development
since 2024. The exact commits above make the old robot stack reproducible and
limit dependency substitution risk. Review and vendor them for long-lived rigs.

Install openpi-client only when using an openpi policy server:

```bash
uv pip install "openpi-client @ git+https://github.com/Physical-Intelligence/openpi.git#subdirectory=packages/openpi-client"
```

### Robot host

The robot host runs the upstream ROS1 Noetic BridgeData server inside its
Docker environment. Pin the same server source used by the client install:

```bash
git clone https://github.com/rail-berkeley/bridge_data_robot.git
cd bridge_data_robot
git checkout b841131ecd512bafb303075bd8f8b677e0bf9f1f
```

Build and enter the ROS1 Noetic image using the scripts under the repository's
`widowx_envs/docker` directory, then start the service inside the container:

```bash
widowx_env_service --server --port 5556
```

The ZMQ request and broadcast ports are 5556 and 5557 by default. Expose both
ports to the client network. Install the upstream Interbotix udev rules on the
host, pass the robot USB devices into the container, and verify stable device
permissions after every replug. Run the upstream manual motion checks inside
the container before connecting a learned policy.

### GPU machine

Install OpenVLA from source and serve a Bridge checkpoint:

```bash
git clone https://github.com/openvla/openvla.git
cd openvla
pip install -e .
python vla-scripts/deploy.py \
  --openvla_path openvla/openvla-7b \
  --port 8000
```

The client sends `unnorm_key="bridge_orig"`. Use a checkpoint that carries the
Bridge normalization statistics or override `unnorm_key` for the fine-tune.
OpenVLA's server does not resize requests. The default client transport resizes
the raw 640x480 frame to 256x256 with Pillow Lanczos and performs an in-memory
JPEG encode/decode round trip to approximate the Bridge evaluation pipeline.

Openpi has no official Bridge checkpoint. The `openpi` entry point is for a
bring-your-own community or local Bridge fine-tune served by openpi.

> [!NOTE]
> inspect-robots-franka also registers a policy named `openpi` (the DROID
> client). With both packages installed, `--policy openpi` resolves to
> whichever distribution the registry enumerates first; a mismatch fails
> loudly at preflight with an action dimension error. Uninstall one package
> or use the Python API to disambiguate.

## Preflight

Preflight constructs metadata only. It does not connect to the robot or a
policy server.

```bash
inspect-robots-widowx-preflight
inspect-robots-widowx-preflight --task cubepick-reach
inspect-robots-widowx-preflight --dry-run
inspect-robots-widowx-preflight --json --dry-run
```

A green report verifies the action dimension, control mode, rotation
representation, gripper kind, frame, camera, state requirements, and optional
scene realizability. JSON output includes a `dry_run` field.

## Run on hardware

Add defaults to `~/.config/inspect-robots/config.ini`:

```ini
[defaults]
policy = openvla
embodiment = widowx
scorer = success_at_end
max_steps = 200
store_frames = true

[policy.args]
server_url = http://192.168.1.20:8000
endpoint = /act
unnorm_key = bridge_orig

[embodiment.args]
host = 192.168.1.10
port = 5556
control_hz = 5.0
move_duration = 0.2
move_to_start = true
```

Run a task or direct instruction:

```bash
inspect-robots run --task cubepick-reach --policy openvla --embodiment widowx
inspect-robots "pick up the red block" --policy openvla --embodiment widowx
```

For openpi, change `policy = openpi` and set `host`, `port`,
`action_horizon = 10`, and `replan_interval = 5` under `[policy.args]`.

Reset first calls the server's neutral reset. By default it then performs a
blocking move to `[0.3, -0.09, 0.26]` using the reference OpenVLA 4x4 rotation.
The attended flow asks for scene readiness after reset. Press Enter during the
episode and answer y/N to score it. `WidowXConfig(unattended=True)` skips all
prompts and operator polling.

## Safety

> [!WARNING]
> Keep an operator at the e-stop for first runs, after container or udev
> changes, and whenever a checkpoint or camera arrangement changes.

- **Hard clamp:** every `step()` rejects non-finite actions and clips all seven
  values to `delta_low/high` inside the embodiment, independent of an
  `Approver`.
- **Mixed delta and absolute semantics:** xyz and rpy are per-step
  displacements, but the gripper slot is an absolute target. It is never a
  gripper rate.
- **Control rate:** the server uses `move_duration=0.2` and the embodiment
  self-paces at 5 Hz. These fields must remain reciprocal.
- **Workspace:** the BridgeData workspace boundaries are enforced server-side.
  Client action bounds do not replace collision checking, physical stops, or
  workspace validation.
- **Discontinued hardware:** use this adapter only for maintained existing
  250S rigs. Confirm spare parts, e-stop behavior, and container recovery before
  unattended operation.
- **First-run verification:** with the arm clear, use small asymmetric actions,
  confirm all xyz and rpy signs, then command gripper values 0.25 and 0.75 while
  the pose slots remain zero. Verify that the larger value opens the gripper.

In displacement mode, `DeltaLimitApprover` derives no additional default limit
beyond the configured action box. There is no reference tracking. Clamping is
applied to the action value itself. Do not use a scalar
`--max-action-delta 0.05` with this contract. A scalar intersects every slot
with `[-0.05, 0.05]`, so the absolute gripper's `[0, 1]` range becomes
`[0, 0.05]`. That pins the gripper nearly shut forever. It does not rate-limit
the gripper.

Python callers can provide a per-dimension vector and leave the gripper limit
at 1.0:

```python
import numpy as np
from inspect_robots import eval
from inspect_robots.approver import ChainApprover, ClampApprover, DeltaLimitApprover
from inspect_robots_widowx import OpenVLAPolicy, WidowXEmbodiment

embodiment = WidowXEmbodiment(host="192.168.1.10")
space = embodiment.info.action_space
approver = ChainApprover(
    ClampApprover(space),
    DeltaLimitApprover(
        space,
        max_delta=np.asarray([0.05, 0.05, 0.05, 0.25, 0.25, 0.25, 1.0]),
    ),
)
logs = eval(
    "cubepick-reach",
    OpenVLAPolicy(server_url="http://192.168.1.20:8000"),
    embodiment,
    approver=approver,
)
```

## Configuration

### Action and state units

| Slot | Label | Action meaning | Observed state meaning |
|------|-------|----------------|------------------------|
| 0 | `dx` | x displacement in metres | x position in metres |
| 1 | `dy` | y displacement in metres | y position in metres |
| 2 | `dz` | z displacement in metres | z position in metres |
| 3 | `droll` | roll displacement in radians | controller-relative roll in radians |
| 4 | `dpitch` | pitch displacement in radians | controller-relative pitch in radians |
| 5 | `dyaw` | yaw displacement in radians | controller-relative yaw in radians |
| 6 | `gripper` | absolute normalized target | normalized opening, 0 closed and 1 open |

The action declares `rotation_repr="none"` deliberately. The dimension labels
carry the rpy meaning while
[inspect-robots issue #143](https://github.com/robocurve/inspect-robots/issues/143)
tracks support for Euler pose displacement guardrails. Declaring
`euler_xyz` today makes the default delta approver reject the space.

### WidowXConfig fields

| Field | Default | Meaning |
|-------|---------|---------|
| `host` | `localhost` | BridgeData robot server host |
| `port` | `5556` | ZMQ request port, with broadcast on port + 1 |
| `control_hz` | `5.0` | Self-paced command rate |
| `start_move_duration_s` | `0.8` | Duration of the blocking move to the start pose at reset |
| `move_duration` | `0.2` | Server step duration, required to equal `1/control_hz` |
| `delta_low` | translation -0.05, rotation -0.25, gripper 0 | Per-step hard lower bounds |
| `delta_high` | translation 0.05, rotation 0.25, gripper 1 | Per-step hard upper bounds |
| `move_to_start` | `True` | Apply the OpenVLA evaluation start pose after neutral reset |
| `start_eef_pos` | `(0.3, -0.09, 0.26)` | Translation inserted into the reference 4x4 transform |
| `start_transform` | `None` | Advanced 16-value row-major 4x4 override; replaces position and rotation |
| `obs_timeout_s` | `10.0` | Maximum wait for a non-None server observation |
| `unattended` | `False` | Skip readiness and scoring prompts |
| `docs_extra` | empty | Rig-specific notes appended to embodiment docs |

There is no `start_eef_quat` field. The reference controller's start rotation
cannot be reproduced from the commonly documented quaternion because the
controller composes an internal default rotation.

### OpenVLAConfig fields

| Field | Default | Meaning |
|-------|---------|---------|
| `server_url` | `http://127.0.0.1:8000` | OpenVLA server base URL |
| `endpoint` | `/act` | REST endpoint |
| `unnorm_key` | `bridge_orig` | Action normalization statistics key |
| `timeout_s` | `30.0` | HTTP request timeout |
| `name` | `openvla` | Policy label in eval logs |
| `resize_px` | `256` | Default transport square resize |
| `jpeg_roundtrip` | `True` | Match Bridge evaluation image compression |

OpenVLA always advertises action horizon 1 and no replanning interval. A custom
injected `post_fn` receives the raw camera frame and owns all preprocessing.

### OpenpiConfig fields

| Field | Default | Meaning |
|-------|---------|---------|
| `host`, `port` | `127.0.0.1`, `8000` | openpi websocket server |
| `api_key` | `None` | Optional authentication, excluded from eval policy config |
| `action_horizon` | `10` | Maximum returned action chunk length |
| `replan_interval` | `5` | Actions consumed before reinference |
| `name` | `openpi` | Policy label in eval logs |
| `resize_px` | `224` | Default resize-with-pad size |

## Development

Use the local cache path required by this workspace:

```bash
export UV_CACHE_DIR="$PWD/.uv-cache"
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest --cov
```

The tests inject clients, transports, clocks, sleeps, operator I/O, and end
polling. They require no robot, server, camera, network, or stdin. Coverage is
enforced at 100 percent with branch coverage enabled.

Dependency changes require `uv lock`. CI uses `uv sync --locked`; the weekly
canary resolves current allowed versions without the lockfile.

## Citation

```bibtex
@software{inspect-robots-widowx,
  author  = {Robocurve},
  title   = {Inspect Robots WidowX: OpenVLA and openpi adapters for WidowX 250S arms},
  year    = {2026},
  url     = {https://github.com/robocurve/inspect-robots-widowx},
  license = {MIT}
}
```

See [`CITATION.cff`](CITATION.cff) for citation metadata.

## License

[MIT](LICENSE)
