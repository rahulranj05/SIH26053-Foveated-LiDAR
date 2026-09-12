from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

from mapping.hierarchical_foveated_mapper import (
    build_hierarchical_foveated_map,
    level_distribution,
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
    Deterministic synthetic A17-style inputs.

    These are NOT semantic/dynamic ground truth.
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

    resolution = np.empty(
        n,
        dtype=np.float64,
    )

    resolution[
        importance >= 0.75
    ] = 0.05

    resolution[
        (importance >= 0.50)
        & (importance < 0.75)
    ] = 0.10

    resolution[
        (importance >= 0.25)
        & (importance < 0.50)
    ] = 0.20

    resolution[
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
        dtype="U16",
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
        resolution,
        dominant_reason,
    )


def benchmark(
    xyz: np.ndarray,
    resolution: np.ndarray,
    dominant_reason: np.ndarray,
):

    for _ in range(
        WARMUP_RUNS
    ):

        build_hierarchical_foveated_map(
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

        result = (
            build_hierarchical_foveated_map(
                xyz,
                resolution,
                dominant_reason,
            )
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        timings.append(
            elapsed * 1000.0
        )

    return (
        result,
        np.asarray(
            timings,
            dtype=np.float64,
        ),
    )


def main() -> None:

    print("=" * 70)

    print(
        "FOVEAMAP A18.3 — TRUE HIERARCHICAL LEAF MAP"
    )

    print("=" * 70)

    xyz = load_kitti_frame(
        FRAME_PATH
    )

    distance = np.sqrt(
        xyz[:, 0] ** 2
        + xyz[:, 1] ** 2
    )

    xyz = xyz[
        distance <= POINT_LIMIT
    ]

    print()

    print(
        f"Input points within "
        f"{POINT_LIMIT:.0f} m: "
        f"{len(xyz):,}"
    )

    resolution, dominant_reason = (
        make_synthetic_a17_inputs(
            xyz
        )
    )

    print()

    print(
        "Synthetic A17-style point resolution:"
    )

    unique, counts = np.unique(
        resolution,
        return_counts=True,
    )

    for value, count in zip(
        unique,
        counts,
    ):

        print(
            f"  {value:.2f} m: "
            f"{count:,} "
            f"({count / len(resolution) * 100.0:.2f}%)"
        )

    result, timings = benchmark(
        xyz,
        resolution,
        dominant_reason,
    )

    print()

    print("-" * 70)

    print(
        "A18.3 RESULTS"
    )

    print("-" * 70)

    print(
        f"Input points       : "
        f"{result.input_points:,}"
    )

    print(
        f"Active leaf cells  : "
        f"{result.active_cells:,}"
    )

    print()

    print(
        "Resolution distribution:"
    )

    distribution = (
        resolution_distribution(
            result
        )
    )

    for value in (
        0.05,
        0.10,
        0.20,
        0.40,
    ):

        count = distribution[
            value
        ]

        percentage = (
            count
            / result.active_cells
            * 100.0
        )

        print(
            f"  {value:.2f} m: "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )

    print()

    print(
        "Hierarchy distribution:"
    )

    levels = (
        level_distribution(
            result
        )
    )

    for level in (
        0,
        1,
        2,
        3,
    ):

        print(
            f"  Level {level}: "
            f"{levels[level]:,}"
        )

    print()

    print(
        "Dominant foveation reasons:"
    )

    reasons = (
        reason_distribution(
            result
        )
    )

    for name in (
        "DISTANCE",
        "KINEMATIC",
        "PREDICTED_PATH",
        "SEMANTIC",
        "DYNAMIC",
    ):

        print(
            f"  {name:<17}: "
            f"{reasons.get(name, 0):,}"
        )

    print()

    print(
        "Point-count conservation:"
    )

    total_points = int(
        np.sum(
            result.point_count
        )
    )

    print(
        f"  Input points : "
        f"{result.input_points:,}"
    )

    print(
        f"  Mapped points: "
        f"{total_points:,}"
    )

    print(
        f"  Match        : "
        f"{total_points == result.input_points}"
    )

    print()

    print(
        "Latency:"
    )

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

    print(
        "A18.3 BENCHMARK COMPLETE"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()