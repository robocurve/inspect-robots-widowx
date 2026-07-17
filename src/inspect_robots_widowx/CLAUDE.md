# inspect_robots_widowx package module map

The package supplies the `widowx` embodiment, `openvla` and `openpi` policies,
and the shared 7-D BridgeData contract.

## Modules

| Module | Responsibility |
|--------|----------------|
| `packing.py` | Pure constants, strict 7-D validation, accessors, and the reference start matrix. |
| `config.py` | Frozen configs and shared action and observation space builders. |
| `policy.py` | Lazy OpenVLA REST and openpi websocket clients. |
| `embodiment.py` | Lazy Bridge client, status faults, hard clamp, observation retry, pacing, and verdicts. |
| `_bridge.py` | Guided loader for pinned git-only BridgeData dependencies. |
| `operator.py` | Injectable readiness and scoring prompts. |
| `preflight.py` | Hardware-free compatibility CLI. |
| `__init__.py` | Reviewed public API fenced by `__all__`. |

## Invariants

- Construction performs no hardware or network I/O.
- `step()` rejects non-finite values and clamps every command.
- Status comparisons are by value because the upstream client returns raw ints.
- `full_image` is the only camera source. The server's `image` key is ignored.
- Both policies consume no state keys and preserve gripper polarity.
- Only `termination_reason="success"` reports success to a scorer.
