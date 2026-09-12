from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

from mapping.foveation_controller import (
    FOVEATION_REASONS,
    compute_foveation,
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
    """Load a SemanticKITTI XYZ point cloud."""

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
            "Invalid SemanticKITTI point-cloud format."
        )

    points = points.reshape(-1, 4)

    return points[:, :3]


def make_synthetic_importance_signals(
    num_points: int,
) -> dict[str, np.ndarray]:
    """
    Create deterministic subsystem importance signals.

    These are benchmark inputs, NOT ground truth.

    The distributions deliberately make different subsystems
    dominate different regions so that A17 actually exercises
    the fusion logic.
    """

    rng = np.random.default_rng(42)

    signals = {}

    # ---------------------------------------------------------
    # Distance importance
    # ---------------------------------------------------------

    distance = rng.uniform(
        0.0,
        1.0,
        num_points,
    )

    # ---------------------------------------------------------
    # Kinematic importance
    # ---------------------------------------------------------

    kinematic = rng.uniform(
        0.0,
        1.0,
        num_points,
    )

    # ---------------------------------------------------------
    # Predicted-path importance
    # ---------------------------------------------------------

    predicted_path = rng.uniform(
        0.0,
        1.0,
        num_points,
    )

    # ---------------------------------------------------------
    # Semantic importance
    # ---------------------------------------------------------

    semantic = rng.uniform(
        0.0,
        1.0,
        num_points,
    )

    # ---------------------------------------------------------
    # Dynamic importance
    # ---------------------------------------------------------

    dynamic = rng.uniform(
        0.0,
        1.0,
        num_points,
    )

    signals["DISTANCE"] = distance
    signals["KINEMATIC"] = kinematic
    signals["PREDICTED_PATH"] = predicted_path
    signals["SEMANTIC"] = semantic
    signals["DYNAMIC"] = dynamic

    return signals


def benchmark(
    signals: dict[str, np.ndarray],
    warmup_runs: int = 3,
    timed_runs: int = 20,
) -> dict[str, float]:
    """Benchmark the complete A17 controller."""

    for _ in range(warmup_runs):
        compute_foveation(
            signals
        )

    latencies_ms = []

    for _ in range(timed_runs):
        start = time.perf_counter()

        result = compute_foveation(
            signals
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
    }


def print_dominant_reasons(
    reasons: np.ndarray,
) -> None:
    """Print which subsystem dominated each point."""

    print()
    print("DOMINANT FOVEATION REASONS")
    print("-" * 70)

    for reason in FOVEATION_REASONS:
        count = int(
            np.count_nonzero(
                reasons == reason
            )
        )

        percentage = (
            100.0
            * count
            / len(reasons)
        )

        print(
            f"{reason:<20}"
            f"{count:>8,} "
            f"({percentage:>6.2f}%)"
        )


def print_resolution_distribution(
    resolution: np.ndarray,
) -> None:
    """Print unified resolution distribution."""

    print()
    print("UNIFIED RESOLUTION DISTRIBUTION")
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
        "FOVEAMAP — A17 UNIFIED FOVEATION "
        "CONTROLLER BENCHMARK"
    )
    print("=" * 70)

    print()
    print(
        f"Frame: {TEST_FRAME}"
    )

    xyz = load_semantic_kitti_frame(
        TEST_FRAME
    )

    num_points = len(xyz)

    print(
        f"Points: {num_points:,}"
    )

    print(
        f"XYZ shape: {xyz.shape}"
    )

    signals = (
        make_synthetic_importance_signals(
            num_points
        )
    )

    print()
    print(
        "NOTE: All subsystem importance values "
        "are synthetic benchmark inputs."
    )

    print(
        "They are NOT ground truth."
    )

    results = benchmark(
        signals
    )

    output = compute_foveation(
        signals
    )

    importance = output[
        "importance"
    ]

    resolution = output[
        "resolution"
    ]

    reasons = output[
        "dominant_reason"
    ]

    print()
    print("UNIFIED IMPORTANCE")
    print("-" * 70)

    print(
        f"Mean       : "
        f"{np.mean(importance):.6f}"
    )

    print(
        f"Median     : "
        f"{np.median(importance):.6f}"
    )

    print(
        f"Min        : "
        f"{np.min(importance):.6f}"
    )

    print(
        f"Max        : "
        f"{np.max(importance):.6f}"
    )

    print_resolution_distribution(
        resolution
    )

    print_dominant_reasons(
        reasons
    )

    print()
    print("A17 LATENCY")
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
    print("A17 STATUS")
    print("-" * 70)

    if results["p95_ms"] < 20.0:
        print(
            "PASS: unified foveation controller "
            "is below 20 ms P95."
        )
    else:
        print(
            "INFO: unified controller exceeds "
            "20 ms P95; profile before optimizing."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()