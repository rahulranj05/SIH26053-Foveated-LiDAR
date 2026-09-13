"""
A38 — Robustness and Safety Benchmark

Exercises the FoveaMap safety controller across healthy, degraded,
and safety-critical runtime conditions.

Run from the repository root:

    python3 scripts/benchmark_a38_safety.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np


# Ensure the repository root is importable when this file is executed
# directly as `python3 scripts/benchmark_a38_safety.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from mapping.safety_controller import (  # noqa: E402
    SafetyConfig,
    apply_safety_policy,
    assess_safety,
    safety_policy,
)


def make_points(
    count: int,
    *,
    distance_min: float = 2.0,
    distance_max: float = 40.0,
) -> np.ndarray:
    if count <= 0:
        return np.empty((0, 3), dtype=np.float64)

    x = np.linspace(
        distance_min,
        distance_max,
        count,
        dtype=np.float64,
    )

    y = np.sin(x * 0.15)
    z = np.cos(x * 0.10) * 0.2

    return np.column_stack((x, y, z))


def add_invalid_points(
    points: np.ndarray,
    fraction: float,
) -> np.ndarray:
    result = points.copy()

    count = int(round(result.shape[0] * fraction))

    if count <= 0:
        return result

    result[:count, 0] = np.nan

    return result


def benchmark_case(
    name: str,
    points: np.ndarray,
    *,
    latency_ms: float = 0.0,
    frame_gap_s: float | None = None,
    config: SafetyConfig,
) -> None:
    iterations = 100

    timings: list[float] = []
    assessment = None
    policy = None
    filtered = None

    for _ in range(iterations):
        start = time.perf_counter()

        assessment = assess_safety(
            points,
            latency_ms=latency_ms,
            frame_gap_s=frame_gap_s,
            semantic_available=True,
            dynamic_available=True,
            config=config,
        )

        policy = safety_policy(
            assessment,
            config=config,
        )

        filtered = apply_safety_policy(
            points,
            assessment,
            config=config,
        )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        timings.append(elapsed_ms)

    timings_array = np.asarray(
        timings,
        dtype=np.float64,
    )

    mean_ms = float(np.mean(timings_array))
    median_ms = float(np.median(timings_array))
    p95_ms = float(np.percentile(timings_array, 95))

    assert assessment is not None
    assert policy is not None
    assert filtered is not None

    print(
        f"{name:<22}"
        f"{assessment.mode:<11}"
        f"{assessment.point_count:>10}"
        f"{assessment.invalid_fraction:>10.3f}"
        f"{mean_ms:>12.3f}"
        f"{median_ms:>12.3f}"
        f"{p95_ms:>12.3f}"
        f"{filtered.shape[0]:>12}"
    )

    if assessment.reasons:
        print(
            f"{'':<22}"
            f"reasons: {', '.join(assessment.reasons)}"
        )

    if policy.max_range_m is not None:
        print(
            f"{'':<22}"
            f"policy: max_range={policy.max_range_m:.1f}m, "
            f"min_resolution={policy.minimum_resolution_m:.2f}m, "
            f"full_foveation={policy.allow_full_foveation}"
        )


def main() -> None:
    config = SafetyConfig()

    normal_points = make_points(
        10_000,
        distance_min=2.0,
        distance_max=40.0,
    )

    degraded_points = make_points(
        1_500,
        distance_min=2.0,
        distance_max=40.0,
    )

    safety_points = make_points(
        500,
        distance_min=2.0,
        distance_max=40.0,
    )

    invalid_points = add_invalid_points(
        normal_points,
        0.10,
    )

    print("=" * 110)
    print("FOVEAMAP A38 — ROBUSTNESS AND SAFETY BENCHMARK")
    print("=" * 110)
    print()
    print(
        f"{'Case':<22}"
        f"{'Mode':<11}"
        f"{'Points':>10}"
        f"{'Invalid':>10}"
        f"{'Mean ms':>12}"
        f"{'Median ms':>12}"
        f"{'P95 ms':>12}"
        f"{'Output':>12}"
    )
    print("-" * 110)

    benchmark_case(
        "NORMAL",
        normal_points,
        config=config,
    )

    benchmark_case(
        "DEGRADED_POINTS",
        degraded_points,
        config=config,
    )

    benchmark_case(
        "SAFETY_POINTS",
        safety_points,
        config=config,
    )

    benchmark_case(
        "HIGH_INVALID",
        invalid_points,
        config=config,
    )

    benchmark_case(
        "HIGH_LATENCY",
        normal_points,
        latency_ms=150.0,
        config=config,
    )

    benchmark_case(
        "CRITICAL_LATENCY",
        normal_points,
        latency_ms=300.0,
        config=config,
    )

    benchmark_case(
        "STALE_FRAME",
        normal_points,
        frame_gap_s=0.30,
        config=config,
    )

    benchmark_case(
        "CRITICAL_STALE",
        normal_points,
        frame_gap_s=0.75,
        config=config,
    )

    print("-" * 110)
    print()
    print("EXPECTED SAFETY BEHAVIOUR")
    print("-" * 110)
    print("NORMAL          -> full point cloud retained")
    print("DEGRADED        -> range limited to 50 m")
    print("SAFETY          -> range limited to 25 m")
    print("HIGH INVALID    -> degraded/safety escalation")
    print("HIGH LATENCY    -> degraded/safety escalation")
    print("STALE FRAME     -> degraded/safety escalation")
    print()

    # Explicit validation of the key safety guarantees.
    normal_assessment = assess_safety(
        normal_points,
        config=config,
    )
    assert normal_assessment.mode == "NORMAL"

    degraded_assessment = assess_safety(
        degraded_points,
        config=config,
    )
    assert degraded_assessment.mode == "DEGRADED"

    safety_assessment = assess_safety(
        safety_points,
        config=config,
    )
    assert safety_assessment.mode == "SAFETY"

    invalid_assessment = assess_safety(
        invalid_points,
        config=config,
    )
    assert invalid_assessment.mode == "DEGRADED"

    high_latency_assessment = assess_safety(
        normal_points,
        latency_ms=150.0,
        config=config,
    )
    assert high_latency_assessment.mode == "DEGRADED"

    critical_latency_assessment = assess_safety(
        normal_points,
        latency_ms=300.0,
        config=config,
    )
    assert critical_latency_assessment.mode == "SAFETY"

    stale_assessment = assess_safety(
        normal_points,
        frame_gap_s=0.30,
        config=config,
    )
    assert stale_assessment.mode == "DEGRADED"

    critical_stale_assessment = assess_safety(
        normal_points,
        frame_gap_s=0.75,
        config=config,
    )
    assert critical_stale_assessment.mode == "SAFETY"

    normal_output = apply_safety_policy(
        normal_points,
        normal_assessment,
        config=config,
    )
    assert normal_output.shape == normal_points.shape

    safety_output = apply_safety_policy(
        normal_points,
        safety_assessment,
        config=config,
    )

    assert safety_output.shape[0] < normal_points.shape[0]

    assert np.all(
        np.linalg.norm(
            safety_output,
            axis=1,
        )
        <= config.safety_max_range_m
    )

    print("A38 benchmark validation: PASS")
    print("=" * 110)


if __name__ == "__main__":
    main()