"""
A20 — FoveaMap Full Pipeline Benchmark

Full integration benchmark:

    Point cloud
        |
        v
    A17 Foveation Controller
        |
        v
    A18.3 Hierarchical Foveated Mapper
        |
        v
    Hierarchical Leaf Map

This benchmark intentionally does NOT modify any validated FoveaMap module.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np


# ============================================================================
# Repository setup
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from mapping.foveation_controller import compute_foveation
from mapping.hierarchical_foveated_mapper import (
    build_hierarchical_foveated_map,
)


# ============================================================================
# Configuration
# ============================================================================

NUM_POINTS = 124_668
NUM_WARMUP = 3
NUM_BENCHMARK_RUNS = 10

RESOLUTION_LABELS = {
    0.05: "5 cm",
    0.10: "10 cm",
    0.20: "20 cm",
    0.40: "40 cm",
}

VALID_LEVELS = {
    0: 0.05,
    1: 0.10,
    2: 0.20,
    3: 0.40,
}

REASONS = (
    "DISTANCE",
    "KINEMATIC",
    "PREDICTED_PATH",
    "SEMANTIC",
    "DYNAMIC",
)


# ============================================================================
# Synthetic point cloud
# ============================================================================

def make_point_cloud(
    num_points: int = NUM_POINTS,
    seed: int = 20260912,
) -> np.ndarray:
    """
    Generate a deterministic vehicle-centric LiDAR-like point cloud.
    """

    rng = np.random.default_rng(seed)

    x = rng.uniform(
        -20.0,
        60.0,
        size=num_points,
    )

    y = rng.uniform(
        -25.0,
        25.0,
        size=num_points,
    )

    z = (
        rng.normal(
            0.0,
            0.12,
            size=num_points,
        )
        + 0.015 * np.sin(x * 0.15)
        + 0.01 * np.cos(y * 0.20)
    )

    return np.column_stack(
        (x, y, z)
    ).astype(np.float64)


# ============================================================================
# Synthetic foveation signals
# ============================================================================

def make_importance_signals(
    xyz: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Create deterministic A17-compatible importance signals.
    """

    x = xyz[:, 0]
    y = xyz[:, 1]

    distance = np.sqrt(
        x * x + y * y
    )

    # ------------------------------------------------------------------------
    # Distance
    # ------------------------------------------------------------------------

    distance_importance = np.clip(
        1.0 - distance / 70.0,
        0.0,
        1.0,
    )

    # ------------------------------------------------------------------------
    # Kinematic
    # ------------------------------------------------------------------------

    forward_corridor = np.exp(
        -0.5 * (y / 4.5) ** 2
    )

    forward_progress = np.clip(
        (x + 10.0) / 50.0,
        0.0,
        1.0,
    )

    kinematic_importance = (
        0.65 * forward_corridor
        + 0.35 * forward_progress
    )

    # ------------------------------------------------------------------------
    # Predicted path
    # ------------------------------------------------------------------------

    predicted_center = (
        0.035
        * np.maximum(x, 0.0) ** 1.35
    )

    path_error = np.abs(
        y - predicted_center
    )

    predicted_path_importance = np.exp(
        -0.5 * (path_error / 3.5) ** 2
    )

    # ------------------------------------------------------------------------
    # Semantic
    # ------------------------------------------------------------------------

    obstacle_1 = np.exp(
        -0.5
        * (
            ((x - 18.0) / 5.0) ** 2
            + ((y + 3.0) / 3.0) ** 2
        )
    )

    obstacle_2 = np.exp(
        -0.5
        * (
            ((x - 32.0) / 4.0) ** 2
            + ((y - 6.0) / 2.5) ** 2
        )
    )

    semantic_importance = np.maximum(
        obstacle_1,
        obstacle_2,
    )

    # ------------------------------------------------------------------------
    # Dynamic
    # ------------------------------------------------------------------------

    dynamic_center = (
        -0.025
        * np.maximum(x, 0.0)
    )

    dynamic_error = np.abs(
        y - dynamic_center
    )

    dynamic_importance = (
        np.exp(
            -0.5
            * (dynamic_error / 2.5) ** 2
        )
        * np.clip(
            x / 40.0,
            0.0,
            1.0,
        )
    )

    return {
        "DISTANCE": np.clip(
            distance_importance,
            0.0,
            1.0,
        ),
        "KINEMATIC": np.clip(
            kinematic_importance,
            0.0,
            1.0,
        ),
        "PREDICTED_PATH": np.clip(
            predicted_path_importance,
            0.0,
            1.0,
        ),
        "SEMANTIC": np.clip(
            semantic_importance,
            0.0,
            1.0,
        ),
        "DYNAMIC": np.clip(
            dynamic_importance,
            0.0,
            1.0,
        ),
    }


# ============================================================================
# A20 pipeline
# ============================================================================

def run_pipeline(
    xyz: np.ndarray,
    signals: dict[str, np.ndarray],
) -> tuple[dict, dict[str, float]]:
    """
    Execute A17 -> A18.3.
    """

    total_start = time.perf_counter()

    # ------------------------------------------------------------------------
    # A17
    # ------------------------------------------------------------------------

    controller_start = time.perf_counter()

    foveation = compute_foveation(
        signals
    )

    controller_end = time.perf_counter()

    # ------------------------------------------------------------------------
    # A18.3
    # ------------------------------------------------------------------------

    mapper_start = time.perf_counter()

    leaf_map = build_hierarchical_foveated_map(
        xyz,
        foveation["resolution"],
        foveation["dominant_reason"],
    )

    mapper_end = time.perf_counter()

    total_end = time.perf_counter()

    timings = {
        "controller_ms": (
            controller_end
            - controller_start
        ) * 1000.0,

        "mapper_ms": (
            mapper_end
            - mapper_start
        ) * 1000.0,

        "total_ms": (
            total_end
            - total_start
        ) * 1000.0,
    }

    return (
        {
            "foveation": foveation,
            "leaf_map": leaf_map,
        },
        timings,
    )


# ============================================================================
# Timing statistics
# ============================================================================

def print_stats(
    name: str,
    values: np.ndarray,
) -> None:

    print(
        f"{name:<22}: "
        f"mean={np.mean(values):8.3f} ms | "
        f"median={np.median(values):8.3f} ms | "
        f"p95={np.percentile(values, 95):8.3f} ms"
    )


# ============================================================================
# Direct structural validation
# ============================================================================

def validate_leaf_map(
    leaf_map,
    expected_input_points: int,
) -> dict[str, bool]:
    """
    Validate the actual HierarchicalLeafMap fields directly.

    This deliberately avoids relying on convenience methods that may not
    exist in the current A18.3 implementation.
    """

    levels = np.asarray(
        leaf_map.levels
    )

    resolutions = np.asarray(
        leaf_map.resolutions
    )

    ix = np.asarray(
        leaf_map.ix
    )

    iy = np.asarray(
        leaf_map.iy
    )

    z_min = np.asarray(
        leaf_map.z_min
    )

    z_max = np.asarray(
        leaf_map.z_max
    )

    z_mean = np.asarray(
        leaf_map.z_mean
    )

    z_variance = np.asarray(
        leaf_map.z_variance
    )

    point_count = np.asarray(
        leaf_map.point_count
    )

    dominant_reason = np.asarray(
        leaf_map.dominant_reason
    )

    active_cells = int(
        leaf_map.active_cells
    )

    # ------------------------------------------------------------------------
    # Basic lengths
    # ------------------------------------------------------------------------

    all_lengths_match = all(
        len(array) == active_cells
        for array in (
            levels,
            resolutions,
            ix,
            iy,
            z_min,
            z_max,
            z_mean,
            z_variance,
            point_count,
            dominant_reason,
        )
    )

    # ------------------------------------------------------------------------
    # Point conservation
    # ------------------------------------------------------------------------

    point_conservation = (
        int(leaf_map.input_points)
        == expected_input_points
        and int(np.sum(point_count))
        == expected_input_points
        and np.all(point_count > 0)
    )

    # ------------------------------------------------------------------------
    # Valid hierarchy levels
    # ------------------------------------------------------------------------

    valid_level_values = np.all(
        np.isin(
            levels,
            np.array(
                [0, 1, 2, 3]
            ),
        )
    )

    # ------------------------------------------------------------------------
    # Level <-> resolution consistency
    # ------------------------------------------------------------------------

    level_resolution_consistency = True

    for level, resolution in VALID_LEVELS.items():

        mask = levels == level

        if np.any(mask):
            if not np.allclose(
                resolutions[mask],
                resolution,
            ):
                level_resolution_consistency = False
                break

    # ------------------------------------------------------------------------
    # Valid resolutions
    # ------------------------------------------------------------------------

    valid_resolutions = np.all(
        np.isin(
            resolutions,
            np.array(
                [0.05, 0.10, 0.20, 0.40]
            ),
        )
    )

    # ------------------------------------------------------------------------
    # Integer-aligned cell indices
    # ------------------------------------------------------------------------

    integer_indices = (
        np.all(
            np.isfinite(ix)
        )
        and np.all(
            np.isfinite(iy)
        )
        and np.all(
            ix == np.floor(ix)
        )
        and np.all(
            iy == np.floor(iy)
        )
    )

    # ------------------------------------------------------------------------
    # No duplicate leaves
    # ------------------------------------------------------------------------

    if active_cells == 0:

        unique_leaf_keys = True

    else:

        keys = np.column_stack(
            (
                levels.astype(np.int64),
                ix.astype(np.int64),
                iy.astype(np.int64),
            )
        )

        unique_leaf_keys = (
            len(
                np.unique(
                    keys,
                    axis=0,
                )
            )
            == active_cells
        )

    # ------------------------------------------------------------------------
    # Numeric validity
    # ------------------------------------------------------------------------

    numeric_validity = (
        np.all(np.isfinite(z_min))
        and np.all(np.isfinite(z_max))
        and np.all(np.isfinite(z_mean))
        and np.all(np.isfinite(z_variance))
        and np.all(z_variance >= 0.0)
        and np.all(z_max >= z_min)
        and np.all(np.isfinite(resolutions))
    )

    # ------------------------------------------------------------------------
    # Reason validity
    # ------------------------------------------------------------------------

    valid_reasons = np.all(
        np.isin(
            dominant_reason,
            np.array(REASONS),
        )
    )

    return {
        "array_lengths": all_lengths_match,
        "point_conservation": point_conservation,
        "valid_levels": valid_level_values,
        "level_resolution": level_resolution_consistency,
        "valid_resolutions": valid_resolutions,
        "integer_indices": integer_indices,
        "unique_leaves": unique_leaf_keys,
        "numeric_validity": numeric_validity,
        "valid_reasons": valid_reasons,
    }


# ============================================================================
# Main
# ============================================================================

def main() -> None:

    print("=" * 70)
    print("FOVEAMAP — A20 FULL PIPELINE BENCHMARK")
    print("=" * 70)

    print()
    print(
        f"Input points       : {NUM_POINTS:,}"
    )
    print(
        f"Warmup runs        : {NUM_WARMUP}"
    )
    print(
        f"Benchmark runs     : {NUM_BENCHMARK_RUNS}"
    )

    # ========================================================================
    # Prepare input
    # ========================================================================

    xyz = make_point_cloud()

    signals = make_importance_signals(
        xyz
    )

    print()
    print("Preparing pipeline...")
    print("  ✓ Point cloud generated")
    print("  ✓ Foveation signals generated")

    # ========================================================================
    # Warmup
    # ========================================================================

    print()
    print("Warmup")
    print("-" * 70)

    for i in range(NUM_WARMUP):

        _, timings = run_pipeline(
            xyz,
            signals,
        )

        print(
            f"  Run {i + 1}/{NUM_WARMUP}: "
            f"{timings['total_ms']:.3f} ms"
        )

    # ========================================================================
    # Benchmark
    # ========================================================================

    controller_times = []
    mapper_times = []
    total_times = []

    final_result = None

    print()
    print("Benchmark")
    print("-" * 70)

    for i in range(NUM_BENCHMARK_RUNS):

        result, timings = run_pipeline(
            xyz,
            signals,
        )

        final_result = result

        controller_times.append(
            timings["controller_ms"]
        )

        mapper_times.append(
            timings["mapper_ms"]
        )

        total_times.append(
            timings["total_ms"]
        )

        print(
            f"  Run {i + 1:2d}/{NUM_BENCHMARK_RUNS}: "
            f"controller={timings['controller_ms']:.3f} ms | "
            f"mapper={timings['mapper_ms']:.3f} ms | "
            f"total={timings['total_ms']:.3f} ms"
        )

    if final_result is None:
        raise RuntimeError(
            "A20 benchmark produced no result."
        )

    controller_array = np.asarray(
        controller_times
    )

    mapper_array = np.asarray(
        mapper_times
    )

    total_array = np.asarray(
        total_times
    )

    # ========================================================================
    # Latency
    # ========================================================================

    print()
    print("=" * 70)
    print("LATENCY")
    print("=" * 70)

    print_stats(
        "Controller",
        controller_array,
    )

    print_stats(
        "Hierarchical mapper",
        mapper_array,
    )

    print_stats(
        "Full pipeline",
        total_array,
    )

    # ========================================================================
    # Outputs
    # ========================================================================

    foveation = final_result[
        "foveation"
    ]

    leaf_map = final_result[
        "leaf_map"
    ]

    resolutions = np.asarray(
        foveation["resolution"]
    )

    dominant_reasons = np.asarray(
        foveation["dominant_reason"]
    )

    # ========================================================================
    # Point-level resolution distribution
    # ========================================================================

    print()
    print("=" * 70)
    print("POINT-LEVEL FOVEATION DISTRIBUTION")
    print("=" * 70)

    for resolution in (
        0.05,
        0.10,
        0.20,
        0.40,
    ):

        count = int(
            np.count_nonzero(
                np.isclose(
                    resolutions,
                    resolution,
                )
            )
        )

        percentage = (
            100.0
            * count
            / len(resolutions)
        )

        print(
            f"  {RESOLUTION_LABELS[resolution]:>5}: "
            f"{count:8,} points "
            f"({percentage:6.2f}%)"
        )

    # ========================================================================
    # Leaf map statistics
    # ========================================================================

    print()
    print("=" * 70)
    print("HIERARCHICAL LEAF MAP")
    print("=" * 70)

    print(
        f"Active leaf cells : "
        f"{leaf_map.active_cells:,}"
    )

    print(
        f"Input points      : "
        f"{leaf_map.input_points:,}"
    )

    leaf_resolutions = np.asarray(
        leaf_map.resolutions
    )

    print()
    print("Leaf resolution distribution:")

    for resolution in (
        0.05,
        0.10,
        0.20,
        0.40,
    ):

        count = int(
            np.count_nonzero(
                np.isclose(
                    leaf_resolutions,
                    resolution,
                )
            )
        )

        percentage = (
            100.0
            * count
            / leaf_map.active_cells
            if leaf_map.active_cells
            else 0.0
        )

        print(
            f"  {RESOLUTION_LABELS[resolution]:>5}: "
            f"{count:8,} cells "
            f"({percentage:6.2f}%)"
        )

    # ========================================================================
    # Dominant reasons
    # ========================================================================

    leaf_reasons = np.asarray(
        leaf_map.dominant_reason
    )

    unique_leaf_reasons, leaf_reason_counts = (
        np.unique(
            leaf_reasons,
            return_counts=True,
        )
    )

    leaf_reason_lookup = dict(
        zip(
            unique_leaf_reasons,
            leaf_reason_counts,
        )
    )

    print()
    print("Dominant foveation reasons:")

    for reason in REASONS:

        count = int(
            leaf_reason_lookup.get(
                reason,
                0,
            )
        )

        percentage = (
            100.0
            * count
            / leaf_map.active_cells
            if leaf_map.active_cells
            else 0.0
        )

        print(
            f"  {reason:<17}: "
            f"{count:8,} cells "
            f"({percentage:6.2f}%)"
        )

    # ========================================================================
    # Structural validation
    # ========================================================================

    validation = validate_leaf_map(
        leaf_map,
        expected_input_points=len(xyz),
    )

    print()
    print("=" * 70)
    print("INTEGRATION VALIDATION")
    print("=" * 70)

    validation_labels = {
        "array_lengths": "Array lengths",
        "point_conservation": "Point conservation",
        "valid_levels": "Valid hierarchy",
        "level_resolution": "Level/resolution",
        "valid_resolutions": "Valid resolutions",
        "integer_indices": "Integer cell indices",
        "unique_leaves": "Unique leaf cells",
        "numeric_validity": "Numeric validity",
        "valid_reasons": "Valid reasons",
    }

    for key, label in validation_labels.items():

        status = validation[key]

        print(
            f"  {label:<20}: "
            f"{'PASS' if status else 'FAIL'}"
        )

    # ========================================================================
    # Controller statistics
    # ========================================================================

    importance = np.asarray(
        foveation["importance"]
    )

    print()
    print("=" * 70)
    print("FOVEATION CONTROLLER")
    print("=" * 70)

    print(
        f"Importance mean    : "
        f"{np.mean(importance):.6f}"
    )

    print(
        f"Importance median  : "
        f"{np.median(importance):.6f}"
    )

    print(
        f"Importance min     : "
        f"{np.min(importance):.6f}"
    )

    print(
        f"Importance max     : "
        f"{np.max(importance):.6f}"
    )

    unique_reasons, reason_counts = np.unique(
        dominant_reasons,
        return_counts=True,
    )

    point_reason_lookup = dict(
        zip(
            unique_reasons,
            reason_counts,
        )
    )

    print()
    print("Point-level dominant reasons:")

    for reason in REASONS:

        count = int(
            point_reason_lookup.get(
                reason,
                0,
            )
        )

        percentage = (
            100.0
            * count
            / len(dominant_reasons)
        )

        print(
            f"  {reason:<17}: "
            f"{count:8,} points "
            f"({percentage:6.2f}%)"
        )

    # ========================================================================
    # Final status
    # ========================================================================

    all_pass = all(
        validation.values()
    )

    print()
    print("=" * 70)

    if all_pass:
        print("A20 STATUS: PASS")
    else:
        print("A20 STATUS: FAIL")

    print("=" * 70)

    if not all_pass:
        failed = [
            label
            for key, label in validation_labels.items()
            if not validation[key]
        ]

        raise RuntimeError(
            "A20 integration validation failed: "
            + ", ".join(failed)
        )


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    main()