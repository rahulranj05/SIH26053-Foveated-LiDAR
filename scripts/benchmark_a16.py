from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

from mapping.dynamic_foveation import (
    dynamic_importance,
    dynamic_resolution,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

TEST_FRAME = (
    REPO_ROOT
    / "datasets"
    / "TEST"
    / "000000.bin"
)


def load_semantic_kitti_frame(
    path: Path,
) -> np.ndarray:
    """
    Load a SemanticKITTI point cloud.

    Returns:
        Nx3 XYZ float32 array.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"SemanticKITTI test frame not found: {path}"
        )

    points = np.fromfile(
        path,
        dtype=np.float32,
    )

    if points.size % 4 != 0:
        raise ValueError(
            "SemanticKITTI .bin file does not contain "
            "valid x/y/z/intensity records."
        )

    points = points.reshape(-1, 4)

    return points[:, :3]


def make_synthetic_dynamic_inputs(
    xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a deterministic synthetic dynamic scene.

    Dynamic probability and velocity are deliberately varied
    across the point cloud so that A16 exercises:

        stationary environment
        slow-moving objects
        medium-speed objects
        fast-moving objects
        highly dynamic objects

    These values are NOT ground truth.

    They are used only to benchmark the A16 policy.
    """

    num_points = len(xyz)

    rng = np.random.default_rng(42)

    dynamic_probability = np.zeros(
        num_points,
        dtype=np.float64,
    )

    velocity = np.zeros(
        num_points,
        dtype=np.float64,
    )

    indices = np.arange(num_points)

    rng.shuffle(indices)

    # ---------------------------------------------------------
    # Scene distribution
    # ---------------------------------------------------------

    stationary_count = int(
        num_points * 0.70
    )

    slow_count = int(
        num_points * 0.10
    )

    medium_count = int(
        num_points * 0.08
    )

    fast_count = int(
        num_points * 0.07
    )

    highly_dynamic_count = (
        num_points
        - stationary_count
        - slow_count
        - medium_count
        - fast_count
    )

    cursor = 0

    # ---------------------------------------------------------
    # Stationary environment
    # ---------------------------------------------------------

    selected = indices[
        cursor : cursor + stationary_count
    ]

    dynamic_probability[selected] = (
        rng.uniform(
            0.0,
            0.10,
            size=len(selected),
        )
    )

    velocity[selected] = (
        rng.uniform(
            0.0,
            0.25,
            size=len(selected),
        )
    )

    cursor += stationary_count

    # ---------------------------------------------------------
    # Slow-moving objects
    # ---------------------------------------------------------

    selected = indices[
        cursor : cursor + slow_count
    ]

    dynamic_probability[selected] = (
        rng.uniform(
            0.30,
            0.60,
            size=len(selected),
        )
    )

    velocity[selected] = (
        rng.uniform(
            0.5,
            2.0,
            size=len(selected),
        )
    )

    cursor += slow_count

    # ---------------------------------------------------------
    # Medium-speed objects
    # ---------------------------------------------------------

    selected = indices[
        cursor : cursor + medium_count
    ]

    dynamic_probability[selected] = (
        rng.uniform(
            0.50,
            0.80,
            size=len(selected),
        )
    )

    velocity[selected] = (
        rng.uniform(
            2.0,
            5.0,
            size=len(selected),
        )
    )

    cursor += medium_count

    # ---------------------------------------------------------
    # Fast-moving objects
    # ---------------------------------------------------------

    selected = indices[
        cursor : cursor + fast_count
    ]

    dynamic_probability[selected] = (
        rng.uniform(
            0.70,
            0.95,
            size=len(selected),
        )
    )

    velocity[selected] = (
        rng.uniform(
            5.0,
            10.0,
            size=len(selected),
        )
    )

    cursor += fast_count

    # ---------------------------------------------------------
    # Highly dynamic objects
    # ---------------------------------------------------------

    selected = indices[
        cursor : cursor + highly_dynamic_count
    ]

    dynamic_probability[selected] = (
        rng.uniform(
            0.90,
            1.00,
            size=len(selected),
        )
    )

    velocity[selected] = (
        rng.uniform(
            10.0,
            20.0,
            size=len(selected),
        )
    )

    return (
        dynamic_probability,
        velocity,
    )


def benchmark(
    dynamic_probability: np.ndarray,
    velocity: np.ndarray,
    warmup_runs: int = 3,
    timed_runs: int = 20,
) -> dict[str, float]:
    """
    Benchmark A16 dynamic importance and resolution.
    """

    for _ in range(warmup_runs):
        importance = dynamic_importance(
            dynamic_probability,
            velocity,
        )

        dynamic_resolution(
            dynamic_probability,
            velocity,
        )

    latencies_ms = []

    for _ in range(timed_runs):
        start = time.perf_counter()

        importance = dynamic_importance(
            dynamic_probability,
            velocity,
        )

        resolution = dynamic_resolution(
            dynamic_probability,
            velocity,
        )

        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        latencies_ms.append(
            elapsed_ms
        )

    latencies = np.asarray(
        latencies_ms,
        dtype=np.float64,
    )

    return {
        "mean_ms": float(
            np.mean(latencies)
        ),
        "median_ms": float(
            np.median(latencies)
        ),
        "min_ms": float(
            np.min(latencies)
        ),
        "max_ms": float(
            np.max(latencies)
        ),
        "std_ms": float(
            np.std(latencies)
        ),
        "p95_ms": float(
            np.percentile(
                latencies,
                95,
            )
        ),
        "importance_mean": float(
            np.mean(importance)
        ),
        "importance_median": float(
            np.median(importance)
        ),
    }


def print_dynamic_distribution(
    dynamic_probability: np.ndarray,
    velocity: np.ndarray,
) -> None:
    """
    Print broad dynamic-scene categories.
    """

    print()
    print("DYNAMIC SCENE DISTRIBUTION")
    print("-" * 70)

    stationary = velocity < 0.5

    slow = (
        (velocity >= 0.5)
        & (velocity < 2.0)
    )

    medium = (
        (velocity >= 2.0)
        & (velocity < 5.0)
    )

    fast = (
        (velocity >= 5.0)
        & (velocity < 10.0)
    )

    highly_dynamic = velocity >= 10.0

    categories = [
        (
            "Stationary",
            stationary,
        ),
        (
            "Slow",
            slow,
        ),
        (
            "Medium",
            medium,
        ),
        (
            "Fast",
            fast,
        ),
        (
            "Highly dynamic",
            highly_dynamic,
        ),
    ]

    for name, mask in categories:
        count = int(
            np.count_nonzero(mask)
        )

        percentage = (
            100.0
            * count
            / len(velocity)
        )

        mean_probability = float(
            np.mean(
                dynamic_probability[mask]
            )
        ) if count > 0 else 0.0

        mean_velocity = float(
            np.mean(
                velocity[mask]
            )
        ) if count > 0 else 0.0

        print(
            f"{name:<20}"
            f"{count:>8,} "
            f"({percentage:>6.2f}%)  "
            f"mean Pdyn={mean_probability:.3f}  "
            f"mean v={mean_velocity:.2f} m/s"
        )


def print_resolution_distribution(
    resolution: np.ndarray,
) -> None:
    """
    Print A16 resolution distribution.
    """

    print()
    print("DYNAMIC RESOLUTION DISTRIBUTION")
    print("-" * 70)

    resolutions = [
        (0.05, "5 cm"),
        (0.10, "10 cm"),
        (0.20, "20 cm"),
        (0.40, "40 cm"),
    ]

    for value, name in resolutions:
        count = int(
            np.count_nonzero(
                resolution == value
            )
        )

        percentage = (
            100.0
            * count
            / len(resolution)
        )

        print(
            f"{name:<10}"
            f"{count:>8,} "
            f"({percentage:>6.2f}%)"
        )


def main() -> None:
    print("=" * 70)
    print(
        "FOVEAMAP — A16 DYNAMIC-AWARE "
        "RESOLUTION BENCHMARK"
    )
    print("=" * 70)

    print()
    print(
        f"Frame: {TEST_FRAME}"
    )

    xyz = load_semantic_kitti_frame(
        TEST_FRAME
    )

    print(
        f"Points: {len(xyz):,}"
    )

    print(
        f"XYZ shape: {xyz.shape}"
    )

    (
        dynamic_probability,
        velocity,
    ) = make_synthetic_dynamic_inputs(
        xyz
    )

    print()
    print(
        "NOTE: Dynamic probability and velocity "
        "are synthetic benchmark inputs."
    )

    print(
        "They are NOT ground truth."
    )

    print_dynamic_distribution(
        dynamic_probability,
        velocity,
    )

    results = benchmark(
        dynamic_probability,
        velocity,
    )

    importance = dynamic_importance(
        dynamic_probability,
        velocity,
    )

    resolution = dynamic_resolution(
        dynamic_probability,
        velocity,
    )

    print_resolution_distribution(
        resolution
    )

    print()
    print("A16 LATENCY")
    print("-" * 70)

    print(
        f"Mean       : "
        f"{results['mean_ms']:.3f} ms"
    )

    print(
        f"Median     : "
        f"{results['median_ms']:.3f} ms"
    )

    print(
        f"Min        : "
        f"{results['min_ms']:.3f} ms"
    )

    print(
        f"Max        : "
        f"{results['max_ms']:.3f} ms"
    )

    print(
        f"Std        : "
        f"{results['std_ms']:.3f} ms"
    )

    print(
        f"P95        : "
        f"{results['p95_ms']:.3f} ms"
    )

    print()
    print("IMPORTANCE")
    print("-" * 70)

    print(
        f"Mean       : "
        f"{results['importance_mean']:.6f}"
    )

    print(
        f"Median     : "
        f"{results['importance_median']:.6f}"
    )

    print()
    print("A16 STATUS")
    print("-" * 70)

    if results["p95_ms"] < 20.0:
        print(
            "PASS: dynamic policy computation "
            "is below 20 ms P95."
        )
    else:
        print(
            "INFO: dynamic policy computation "
            "exceeds 20 ms P95; profile before optimizing."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()