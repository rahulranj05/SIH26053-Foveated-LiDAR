from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mapping.semantic_foveation import (
    SEMANTIC_CLASSES,
    resolution_from_semantic_importance,
    semantic_importance,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_FRAME = REPO_ROOT / "datasets" / "TEST" / "000000.bin"


def load_semantic_kitti_frame(path: Path) -> np.ndarray:
    """
    Load a SemanticKITTI point cloud.

    Returns:
        xyz: Nx3 float32 array.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"SemanticKITTI test frame not found: {path}"
        )

    points = np.fromfile(path, dtype=np.float32)

    if points.size % 4 != 0:
        raise ValueError(
            "SemanticKITTI .bin file does not contain valid "
            "x/y/z/intensity records."
        )

    points = points.reshape(-1, 4)

    return points[:, :3]


def make_synthetic_semantic_labels(
    num_points: int,
) -> np.ndarray:
    """
    Create a deterministic synthetic semantic distribution.

    This is used because datasets/TEST/000000.bin contains only
    the point cloud and does not contain a matching semantic
    prediction file.

    The benchmark therefore measures the A15 semantic-policy
    computation itself rather than pretending that these labels
    are ground truth.
    """

    labels = np.full(
        num_points,
        SEMANTIC_CLASSES["DRIVABLE"],
        dtype=np.int32,
    )

    rng = np.random.default_rng(42)

    counts = {
        "UNKNOWN": int(num_points * 0.05),
        "NON_DRIVABLE": int(num_points * 0.10),
        "STATIC_OBSTACLE": int(num_points * 0.10),
        "VEHICLE": int(num_points * 0.05),
        "VULNERABLE_USER": int(num_points * 0.02),
    }

    remaining_indices = np.arange(num_points)
    rng.shuffle(remaining_indices)

    cursor = 0

    for class_name, count in counts.items():
        class_id = SEMANTIC_CLASSES[class_name]

        selected = remaining_indices[
            cursor : cursor + count
        ]

        labels[selected] = class_id

        cursor += count

    return labels


def benchmark(
    labels: np.ndarray,
    warmup_runs: int = 3,
    timed_runs: int = 20,
) -> dict[str, float]:
    """
    Benchmark semantic importance and semantic resolution generation.
    """

    for _ in range(warmup_runs):
        importance = semantic_importance(labels)

        resolution_from_semantic_importance(
            importance
        )

    latencies_ms = []

    for _ in range(timed_runs):
        start = time.perf_counter()

        importance = semantic_importance(labels)

        resolution = resolution_from_semantic_importance(
            importance
        )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        latencies_ms.append(elapsed_ms)

    latencies = np.asarray(latencies_ms)

    return {
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "std_ms": float(np.std(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "importance_mean": float(np.mean(importance)),
        "importance_median": float(np.median(importance)),
    }


def print_class_distribution(
    labels: np.ndarray,
) -> None:
    print()
    print("SEMANTIC DISTRIBUTION")
    print("-" * 70)

    for class_name, class_id in SEMANTIC_CLASSES.items():
        count = int(
            np.count_nonzero(labels == class_id)
        )

        percentage = (
            100.0 * count / len(labels)
        )

        print(
            f"{class_name:<20} "
            f"{count:>8,} "
            f"({percentage:>6.2f}%)"
        )


def print_resolution_distribution(
    resolution: np.ndarray,
) -> None:
    print()
    print("SEMANTIC RESOLUTION DISTRIBUTION")
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
            100.0 * count / len(resolution)
        )

        print(
            f"{name:<10} "
            f"{count:>8,} "
            f"({percentage:>6.2f}%)"
        )


def main() -> None:
    print("=" * 70)
    print(
        "FOVEAMAP — A15 SEMANTIC-AWARE "
        "RESOLUTION BENCHMARK"
    )
    print("=" * 70)

    print()
    print(f"Frame: {TEST_FRAME}")

    xyz = load_semantic_kitti_frame(
        TEST_FRAME
    )

    print(
        f"Points: {len(xyz):,}"
    )

    print(
        f"XYZ shape: {xyz.shape}"
    )

    labels = make_synthetic_semantic_labels(
        num_points=len(xyz)
    )

    print()
    print(
        "NOTE: SemanticKITTI TEST frame contains "
        "no semantic labels."
    )

    print(
        "A deterministic synthetic semantic distribution "
        "is therefore"
    )

    print(
        "used to benchmark the A15 semantic-policy "
        "computation."
    )

    print(
        "These labels are NOT ground truth."
    )

    print_class_distribution(labels)

    results = benchmark(labels)

    importance = semantic_importance(labels)

    resolution = resolution_from_semantic_importance(
        importance
    )

    print_resolution_distribution(
        resolution
    )

    print()
    print("A15 LATENCY")
    print("-" * 70)

    print(
        f"Mean       : {results['mean_ms']:.3f} ms"
    )

    print(
        f"Median     : {results['median_ms']:.3f} ms"
    )

    print(
        f"Min        : {results['min_ms']:.3f} ms"
    )

    print(
        f"Max        : {results['max_ms']:.3f} ms"
    )

    print(
        f"Std        : {results['std_ms']:.3f} ms"
    )

    print(
        f"P95        : {results['p95_ms']:.3f} ms"
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
    print("A15 STATUS")
    print("-" * 70)

    if results["p95_ms"] < 20.0:
        print(
            "PASS: semantic policy computation "
            "is below 20 ms P95."
        )
    else:
        print(
            "INFO: semantic policy computation "
            "exceeds 20 ms P95; profile before optimizing."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()