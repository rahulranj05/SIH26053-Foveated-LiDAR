#!/usr/bin/env python3
"""
FoveaMap — Final SIH Demonstration Entrypoint

A40.3 final demo packaging.

This is a thin presentation/CLI wrapper around the frozen A39.3 live
dashboard. It does not introduce or modify any mapping algorithm.

Pipeline:

    Synthetic LiDAR
        ↓
    SensorFrame
        ↓
    A31 Temporal FoveaMap Pipeline
        ↓
    A30 Signal Integration
        ↓
    A20 FoveaMap Pipeline
        ↓
    A38 Safety Controller
        ↓
    A39.1 Dashboard Data Layer
        ↓
    A39.2 Visual Dashboard
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Repository import path
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"

for path in (ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the final demonstration CLI."""
    parser = argparse.ArgumentParser(
        description="Run the packaged FoveaMap SIH final demonstration."
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=60,
        help="Number of demonstration frames (default: 60).",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Demo frame rate in Hz (default: 10).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=39,
        help="Deterministic synthetic-data seed (default: 39).",
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable the live Matplotlib dashboard window.",
    )

    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Optional directory for saved dashboard outputs.",
    )

    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate CLI arguments before starting the demonstration."""
    if args.frames < 1:
        raise ValueError("--frames must be an integer >= 1.")

    if args.fps <= 0:
        raise ValueError("--fps must be greater than 0.")

    if args.save_dir is not None:
        args.save_dir = args.save_dir.expanduser().resolve()


# ---------------------------------------------------------------------------
# Demo execution
# ---------------------------------------------------------------------------

def run_final_demo(args: argparse.Namespace) -> int:
    """Run the frozen A39.3 demonstration using DemoConfig."""
    validate_arguments(args)

    from run_foveamap_dashboard import DemoConfig, run_demo

    save_dir = args.save_dir

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    config = DemoConfig(
        total_frames=args.frames,
        frame_rate_hz=args.fps,
        random_seed=args.seed,
        save_dir=save_dir,
    )

    dashboard_frames = run_demo(
        config=config,
        show=not args.no_show,
    )

    if len(dashboard_frames) != args.frames:
        raise RuntimeError(
            "Final demo produced "
            f"{len(dashboard_frames)} frames; "
            f"expected {args.frames}."
        )

    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return run_final_demo(args)
    except KeyboardInterrupt:
        print("\nFoveaMap final demo interrupted.")
        return 130
    except Exception as exc:
        print(
            f"\nFoveaMap final demo failed: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())