import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mapping.distance_adaptive import build_distance_adaptive_map
from mapping.vectorized_adaptive import build_vectorized_adaptive_map


DATA_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "test"
    / "000000.bin"
)

WARMUP_RUNS = 3
TIMED_RUNS = 20
MAX_DISTANCE = 100.0


def load_kitti_frame(path: Path) -> np.ndarray:
    raw = np.fromfile(
        path,
        dtype=np.float32,
    )

    if raw.size % 4 != 0:
        raise ValueError(
            "KITTI point cloud file does not contain "
            "a multiple of 4 float32 values."
        )

    points = raw.reshape(-1, 4)

    return points[:, :3].astype(
        np.float64,
        copy=False,
    )


def filter_to_100m(xyz: np.ndarray) -> np.ndarray:
    distance = np.sqrt(
        xyz[:, 0] ** 2
        + xyz[:, 1] ** 2
    )

    return xyz[
        distance <= MAX_DISTANCE
    ]


def benchmark(
    function,
    xyz: np.ndarray,
    warmup_runs: int,
    timed_runs: int,
):
    for _ in range(warmup_runs):
        function(xyz)

    timings = []

    result = None

    for _ in range(timed_runs):
        start = time.perf_counter()

        result = function(xyz)

        elapsed = (
            time.perf_counter() - start
        ) * 1000.0

        timings.append(elapsed)

    timings = np.asarray(
        timings,
        dtype=np.float64,
    )

    return result, timings


def print_statistics(
    name: str,
    result,
    timings: np.ndarray,
):
    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"Active cells : {result.cell_count:,}"
    )

    print(
        f"Mean latency : {np.mean(timings):.3f} ms"
    )

    print(
        f"Median       : {np.median(timings):.3f} ms"
    )

    print(
        f"Minimum      : {np.min(timings):.3f} ms"
    )

    print(
        f"Maximum      : {np.max(timings):.3f} ms"
    )

    print(
        f"Std deviation: {np.std(timings):.3f} ms"
    )

    print(
        f"P95 latency  : {np.percentile(timings, 95):.3f} ms"
    )


def main():
    print()
    print("=" * 70)
    print("FOVEAMAP — A11 VECTORISATION BENCHMARK")
    print("=" * 70)

    print()
    print(f"Dataset: {DATA_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"SemanticKITTI test frame not found:\n"
            f"{DATA_PATH}"
        )

    xyz = load_kitti_frame(
        DATA_PATH
    )

    print(
        f"Original points : {len(xyz):,}"
    )

    xyz_100m = filter_to_100m(
        xyz
    )

    print(
        f"Points <=100 m  : {len(xyz_100m):,}"
    )

    print()
    print(
        f"Warmup runs     : {WARMUP_RUNS}"
    )

    print(
        f"Timed runs      : {TIMED_RUNS}"
    )

    print()

    a8_result, a8_timings = benchmark(
        build_distance_adaptive_map,
        xyz_100m,
        WARMUP_RUNS,
        TIMED_RUNS,
    )

    a11_result, a11_timings = benchmark(
        build_vectorized_adaptive_map,
        xyz_100m,
        WARMUP_RUNS,
        TIMED_RUNS,
    )

    print_statistics(
        "A8 — ORIGINAL ADAPTIVE MAPPER",
        a8_result,
        a8_timings,
    )

    print_statistics(
        "A11 — VECTORIZED ADAPTIVE MAPPER",
        a11_result,
        a11_timings,
    )

    a8_median = np.median(
        a8_timings
    )

    a11_median = np.median(
        a11_timings
    )

    speedup = (
        a8_median / a11_median
    )

    improvement = (
        (a8_median - a11_median)
        / a8_median
        * 100.0
    )

    print()
    print("=" * 70)
    print("A11 COMPARISON")
    print("=" * 70)

    print(
        f"A8 active cells   : "
        f"{a8_result.cell_count:,}"
    )

    print(
        f"A11 active cells  : "
        f"{a11_result.cell_count:,}"
    )

    print(
        f"Cell count match  : "
        f"{a8_result.cell_count == a11_result.cell_count}"
    )

    print()

    print(
        f"A8 median latency : "
        f"{a8_median:.3f} ms"
    )

    print(
        f"A11 median latency: "
        f"{a11_median:.3f} ms"
    )

    print(
        f"Speedup           : "
        f"{speedup:.2f}x"
    )

    print(
        f"Latency reduction : "
        f"{improvement:.2f}%"
    )

    print()

    if a11_median < a8_median:
        print(
            "RESULT: A11 is faster than A8."
        )
    else:
        print(
            "RESULT: A11 is not faster than A8 yet."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
    