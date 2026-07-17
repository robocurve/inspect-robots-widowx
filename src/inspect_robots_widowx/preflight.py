"""Hardware-free compatibility preflight for WidowX policy pairs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable

from inspect_robots.compat import CompatibilityReport, check_compatibility
from inspect_robots.policy import Policy
from inspect_robots.registry import resolve
from inspect_robots.task import Task

from inspect_robots_widowx.config import OpenVLAConfig, WidowXConfig
from inspect_robots_widowx.embodiment import WidowXEmbodiment
from inspect_robots_widowx.policy import OpenVLAPolicy

CheckFn = Callable[..., CompatibilityReport]


def build(
    widowx_cfg: WidowXConfig | None = None,
    openvla_cfg: OpenVLAConfig | None = None,
) -> tuple[OpenVLAPolicy, WidowXEmbodiment]:
    """Construct the default policy pair without hardware or network access."""
    return OpenVLAPolicy(openvla_cfg), WidowXEmbodiment(widowx_cfg)


def run_preflight(
    task_name: str | None = None,
    *,
    policy: Policy | None = None,
    embodiment: WidowXEmbodiment | None = None,
    check: CheckFn = check_compatibility,
) -> CompatibilityReport:
    """Return compatibility findings, optionally including task realizability."""
    pol = policy if policy is not None else OpenVLAPolicy()
    emb = embodiment if embodiment is not None else WidowXEmbodiment()
    task: Task | None = resolve("task", task_name) if task_name else None
    return check(pol, emb, task)


def _format_human(report: CompatibilityReport, *, dry_run: bool) -> str:
    lines = ["OK: policy and embodiment are compatible." if report.ok else "INCOMPATIBLE:"]
    for issue in report.errors:
        lines.append(f"  ERROR   [{issue.code}] {issue.message}")
    for issue in report.warnings:
        lines.append(f"  WARNING [{issue.code}] {issue.message}")
    if dry_run:
        lines.append("(dry-run) No motion will be commanded.")
    return "\n".join(lines)


def main(argv: list[str] | None = None, *, run: CheckFn | None = None) -> int:
    """Print a compatibility report and return nonzero only for errors."""
    parser = argparse.ArgumentParser(prog="inspect-robots-widowx-preflight")
    parser.add_argument(
        "--task", default=None, help="optional task name to check scene realizability"
    )
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument("--dry-run", action="store_true", help="affirm no motion is commanded")
    args = parser.parse_args(argv)
    run_fn: Callable[..., CompatibilityReport] = run if run is not None else run_preflight
    report = run_fn(args.task)
    if args.json:
        payload = {
            "ok": report.ok,
            "errors": [{"code": item.code, "message": item.message} for item in report.errors],
            "warnings": [{"code": item.code, "message": item.message} for item in report.warnings],
            "dry_run": args.dry_run,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(_format_human(report, dry_run=args.dry_run))
    return 1 if report.errors else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
