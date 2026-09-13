from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mapping.temporal_motion import (
    TemporalMotionConfig,
    temporal_motion_evidence,
)


def benchmark_case(
    point_count: int,
    *,
    rng: np.random.Generator,
    repeats: int = 5,
) -> tuple[float, float, float, float]:
    current = rng.uniform(
        [-50.0, -30.0, -3.0],
        [100.0, 30.0, 5.0],
        size=(point_count, 3),
    )

    previous = current + rng.normal(
        0.0,
        0.15,
        size=current.shape,
    )

    config = TemporalMotionConfig()

    temporal_motion_evidence(
        current,
        previous,
        config=config,
    )

    timings = []

    for _ in range(repeats):
        start = time.perf_counter()

        temporal_motion_evidence(
            current,
            previous,
            config=config,
        )

        timings.append(
            (time.perf_counter() - start) * 1000.0
        )

    values = np.asarray(
        timings,
        dtype=np.float64,
    )

    mean_ms = float(np.mean(values))
    median_ms = float(np.median(values))
    p95_ms = float(np.percentile(values, 95))
    fps = 1000.0 / mean_ms

    return mean_ms, median_ms, p95_ms, fps


def main() -> None:
    rng = np.random.default_rng(26053)

    point_counts = [
        1_000,
        5_000,
        10_000,
        25_000,
        50_000,
        100_000,
        124_668,
    ]

    print("=" * 78)
    print("FOVEAMAP — A37 PERFORMANCE OPTIMIZATION")
    print("=" * 78)
    print()
    print(
        f"{'Points':>10} "
        f"{'Mean ms':>14} "
        f"{'Median ms':>14} "
        f"{'P95 ms':>14} "
        f"{'FPS':>12}"
    )
    print("-" * 78)

    for point_count in point_counts:
        mean_ms, median_ms, p95_ms, fps = benchmark_case(
            point_count,
            rng=rng,
        )

        print(
            f"{point_count:10d} "
            f"{mean_ms:14.3f} "
            f"{median_ms:14.3f} "
            f"{p95_ms:14.3f} "
            f"{fps:12.2f}"
        )

    print()
    print("=" * 78)
    print("A37 BENCHMARK COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
