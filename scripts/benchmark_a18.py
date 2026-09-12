from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

from mapping.foveated_mapper import (
    build_foveated_map,
    reason_distribution,
    resolution_distribution,
)


ROOT = Path(__file__).resolve().parents[1]

FRAME_PATH = (
    ROOT
    / "datasets"
    / "TEST"
    / "000000.bin"
)

POINT_LIMIT = 100.0

WARMUP_RUNS = 3
TIMED_RUNS = 20


def load_kitti_frame(
    path: Path,
) -> np.ndarray:
    """
    Load a SemanticKITTI .bin point cloud.

    Format:
        x, y, z, intensity
    """

    raw = np.fromfile(
        path,
        dtype=np.float32,
    )

    if raw.size % 4 != 0:
        raise ValueError(
            "Invalid KITTI .bin file."
        )

    points = raw.reshape(
        -1,
        4,
    )

    return points[:, :3].astype(
        np.float64
    )


def make_synthetic_a17_inputs(
    xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create deterministic synthetic A17-style
    unified foveation inputs.

    These are NOT semantic or dynamic ground truth.

    The purpose is to benchmark the A18 mapper
    using a realistic mixture of resolutions and
    foveation reasons.
    """

    n = len(xyz)

    rng = np.random.default_rng(
        2026
    )

    importance = rng.uniform(
        0.0,
        1.0,
        size=n,
    )

    resolutions = np.empty(
        n,
        dtype=np.float64,
    )

    resolutions[
        importance >= 0.75
    ] = 0.05

    resolutions[
        (importance >= 0.50)
        & (importance < 0.75)
    ] = 0.10

    resolutions[
        (importance >= 0.25)
        & (importance < 0.50)
    ] = 0.20

    resolutions[
        importance < 0.25
    ] = 0.40

    reasons = np.array(
        [
            "DISTANCE",
            "KINEMATIC",
            "PREDICTED_PATH",
            "SEMANTIC",
            "DYNAMIC",
        ],
        dtype=object,
    )

    reason_indices = rng.integers(
        0,
        len(reasons),
        size=n,
    )

    dominant_reason = reasons[
        reason_indices
    ]

    return (
        resolutions,
        dominant_reason,
    )


def benchmark(
    xyz: np.ndarray,
    resolution: np.ndarray,
    dominant_reason: np.ndarray,
) -> tuple:
    """
    Benchmark A18 construction.
    """

    for _ in range(
        WARMUP_RUNS
    ):
        build_foveated_map(
            xyz,
            resolution,
            dominant_reason,
        )

    timings = []

    result = None

    for _ in range(
        TIMED_RUNS
    ):
        start = time.perf_counter()

        result = build_foveated_map(
            xyz,
            resolution,
            dominant_reason,
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        timings.append(
            elapsed * 1000.0
        )

    timings = np.asarray(
        timings,
        dtype=np.float64,
    )

    return (
        result,
        timings,
    )


def main() -> None:
    print("=" * 70)
    print("FOVEAMAP A18 — FOVEATED 2.5D MAPPER BENCHMARK")
    print("=" * 70)

    print()
    print(
        f"Frame: {FRAME_PATH}"
    )

    if not FRAME_PATH.exists():
        raise FileNotFoundError(
            f"Frame not found: {FRAME_PATH}"
        )

    xyz = load_kitti_frame(
        FRAME_PATH
    )

    distance = np.sqrt(
        xyz[:, 0] ** 2
        + xyz[:, 1] ** 2
    )

    mask = (
        distance
        <= POINT_LIMIT
    )

    xyz = xyz[mask]

    print()
    print(
        f"Input points within {POINT_LIMIT:.0f} m: "
        f"{len(xyz):,}"
    )

    resolution, dominant_reason = (
        make_synthetic_a17_inputs(
            xyz
        )
    )

    print()
    print(
        "Synthetic A17-style resolution distribution:"
    )

    unique, counts = np.unique(
        resolution,
        return_counts=True,
    )

    for value, count in zip(
        unique,
        counts,
    ):
        percentage = (
            count
            / len(resolution)
            * 100.0
        )

        print(
            f"  {value:.2f} m: "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )

    result, timings = benchmark(
        xyz,
        resolution,
        dominant_reason,
    )

    print()
    print("-" * 70)
    print("A18 RESULTS")
    print("-" * 70)

    print(
        f"Input points       : "
        f"{result.input_points:,}"
    )

    print(
        f"Active map cells   : "
        f"{result.active_cells:,}"
    )

    print()
    print("Resolution distribution:")

    distribution = (
        resolution_distribution(
            result
        )
    )

    for resolution_value in (
        0.05,
        0.10,
        0.20,
        0.40,
    ):
        count = distribution[
            resolution_value
        ]

        percentage = (
            count
            / result.active_cells
            * 100.0
            if result.active_cells
            else 0.0
        )

        print(
            f"  {resolution_value:.2f} m: "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )

    print()
    print("Hierarchy distribution:")

    levels = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
    }

    for cell in result.cells:
        levels[
            cell.level
        ] += 1

    for level in range(4):
        print(
            f"  Level {level}: "
            f"{levels[level]:,}"
        )

    print()
    print("Dominant foveation reasons:")

    reasons = reason_distribution(
        result
    )

    for reason, count in sorted(
        reasons.items()
    ):
        percentage = (
            count
            / result.active_cells
            * 100.0
            if result.active_cells
            else 0.0
        )

        print(
            f"  {reason:18s}: "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )

    print()
    print("Latency:")

    print(
        f"  Mean   : "
        f"{np.mean(timings):.3f} ms"
    )

    print(
        f"  Median : "
        f"{np.median(timings):.3f} ms"
    )

    print(
        f"  Min    : "
        f"{np.min(timings):.3f} ms"
    )

    print(
        f"  Max    : "
        f"{np.max(timings):.3f} ms"
    )

    print(
        f"  Std    : "
        f"{np.std(timings):.3f} ms"
    )

    print(
        f"  P95    : "
        f"{np.percentile(timings, 95):.3f} ms"
    )

    print()
    print("=" * 70)
    print("A18 BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()