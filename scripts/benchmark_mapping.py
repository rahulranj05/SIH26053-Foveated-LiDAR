import sys
import time
from pathlib import Path

import numpy as np


# =====================================================================
# PROJECT PATH
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =====================================================================
# FOVEAMAP IMPORTS
# =====================================================================

from mapping.basic_grid import build_uniform_grid
from mapping.distance_adaptive import build_distance_adaptive_map
from mapping.hierarchical_grid import hierarchy_cells_for_points


# =====================================================================
# CONFIGURATION
# =====================================================================

POINT_CLOUD_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "test"
    / "000000.bin"
)

# Common benchmark region.
#
# Every method receives exactly the same input point cloud:
# points within a 100 m horizontal radius.
#
# The uniform grid uses this square extent so that no benchmark
# point is clipped.
X_RANGE = (-100.0, 100.0)
Y_RANGE = (-100.0, 100.0)

UNIFORM_RESOLUTION = 0.05

WARMUP_RUNS = 2
TIMED_RUNS = 10


# =====================================================================
# DATA LOADING
# =====================================================================

def load_semantickitti_bin(path: Path) -> np.ndarray:
    """
    Load a SemanticKITTI .bin file.

    Format:
        x, y, z, remission

    Returns:
        xyz with shape (N, 3)
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Point cloud not found:\n{path}"
        )

    raw = np.fromfile(
        path,
        dtype=np.float32,
    )

    if raw.size % 4 != 0:
        raise ValueError(
            "SemanticKITTI file does not contain "
            "a whole number of XYZ+remission points."
        )

    points = raw.reshape(-1, 4)

    return points[:, :3].astype(
        np.float64,
        copy=False,
    )


# =====================================================================
# TIMING
# =====================================================================

def benchmark_function(
    function,
    *args,
    warmup_runs: int = WARMUP_RUNS,
    timed_runs: int = TIMED_RUNS,
    **kwargs,
):
    """
    Benchmark a function using repeated executions.

    Warm-up runs are excluded from the statistics.

    Returns:
        result from final run
        dictionary containing timing statistics
    """

    for _ in range(warmup_runs):
        function(*args, **kwargs)

    timings = []

    result = None

    for _ in range(timed_runs):

        start = time.perf_counter()

        result = function(
            *args,
            **kwargs,
        )

        elapsed = time.perf_counter() - start

        timings.append(
            elapsed * 1000.0
        )

    timings = np.asarray(
        timings,
        dtype=np.float64,
    )

    statistics = {
        "mean_ms": float(np.mean(timings)),
        "median_ms": float(np.median(timings)),
        "min_ms": float(np.min(timings)),
        "max_ms": float(np.max(timings)),
        "std_ms": float(np.std(timings)),
    }

    return result, statistics


# =====================================================================
# MEMORY ESTIMATION
# =====================================================================

def estimate_uniform_dense_memory_bytes(
    grid,
) -> int:
    """
    Actual numeric array payload allocated by the dense
    UniformGrid2D representation.
    """

    arrays = [
        grid.count,
        grid.z_min,
        grid.z_max,
        grid.z_mean,
        grid.z_var,
    ]

    return sum(
        array.nbytes
        for array in arrays
    )


def estimate_compact_cell_memory_bytes(
    cell_count: int,
) -> int:
    """
    Estimate memory for a compact numeric representation
    of one map cell.

    Assumed fields:

        x_index
        y_index
        resolution
        point_count
        z_min
        z_max
        z_mean
        z_var

    This deliberately uses a fixed numeric representation
    rather than Python object/list overhead.

    It is therefore useful for comparing the information
    content of different map representations.
    """

    if cell_count <= 0:
        return 0

    bytes_per_cell = (
        np.dtype(np.int64).itemsize * 3
        + np.dtype(np.float64).itemsize * 5
    )

    return (
        cell_count
        * bytes_per_cell
    )


def estimate_hierarchy_memory_bytes(
    cell_count: int,
) -> int:
    """
    Estimate compact numeric storage for:

        level
        x_index
        y_index
    """

    if cell_count <= 0:
        return 0

    bytes_per_cell = (
        np.dtype(np.int64).itemsize * 3
    )

    return (
        cell_count
        * bytes_per_cell
    )


# =====================================================================
# FORMATTING
# =====================================================================

def format_bytes(value: int) -> str:

    value = float(value)

    if value < 1024:
        return f"{value:.0f} B"

    if value < 1024**2:
        return f"{value / 1024:.2f} KB"

    if value < 1024**3:
        return f"{value / (1024**2):.2f} MB"

    return f"{value / (1024**3):.2f} GB"


def percentage_reduction(
    baseline: float,
    value: float,
) -> float:

    if baseline == 0:
        return 0.0

    return (
        (baseline - value)
        / baseline
        * 100.0
    )


def percentage_change(
    baseline: float,
    value: float,
) -> float:

    if baseline == 0:
        return 0.0

    return (
        (value - baseline)
        / baseline
        * 100.0
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("=" * 70)
    print("FOVEAMAP — A10 REFINED MEMORY + LATENCY BENCHMARK")
    print("=" * 70)

    print()
    print(
        f"Point cloud:\n{POINT_CLOUD_PATH}"
    )

    print()
    print(
        f"Warm-up runs: {WARMUP_RUNS}"
    )

    print(
        f"Timed runs:   {TIMED_RUNS}"
    )

    # =================================================================
    # LOAD INPUT
    # =================================================================

    xyz = load_semantickitti_bin(
        POINT_CLOUD_PATH
    )

    print()
    print(
        f"Input points: {len(xyz):,}"
    )

    horizontal_distance = np.sqrt(
        xyz[:, 0] ** 2
        + xyz[:, 1] ** 2
    )

    benchmark_mask = (
        horizontal_distance <= 100.0
    )

    benchmark_xyz = xyz[
        benchmark_mask
    ]

    ignored_points = (
        len(xyz)
        - len(benchmark_xyz)
    )

    print(
        f"Points within 100 m radius: "
        f"{len(benchmark_xyz):,}"
    )

    print(
        f"Points outside 100 m radius: "
        f"{ignored_points:,}"
    )

    # =================================================================
    # 1. UNIFORM 5 CM
    # =================================================================

    print()
    print("-" * 70)
    print("1. UNIFORM 5 CM GRID")
    print("-" * 70)

    uniform_grid, uniform_stats = benchmark_function(
        build_uniform_grid,
        benchmark_xyz,
        resolution=UNIFORM_RESOLUTION,
        x_range=X_RANGE,
        y_range=Y_RANGE,
    )

    uniform_total_cells = int(
        uniform_grid.count.size
    )

    uniform_active_cells = int(
        np.count_nonzero(
            uniform_grid.count > 0
        )
    )

    uniform_dense_memory = (
        estimate_uniform_dense_memory_bytes(
            uniform_grid
        )
    )

    uniform_compact_memory = (
        estimate_compact_cell_memory_bytes(
            uniform_active_cells
        )
    )

    print()
    print(
        f"Spatial extent: "
        f"X {X_RANGE[0]:.0f} to {X_RANGE[1]:.0f} m"
    )

    print(
        f"                  "
        f"Y {Y_RANGE[0]:.0f} to {Y_RANGE[1]:.0f} m"
    )

    print(
        f"Resolution: "
        f"{UNIFORM_RESOLUTION * 100:.0f} cm"
    )

    print(
        f"Grid dimensions: "
        f"{uniform_grid.height} x "
        f"{uniform_grid.width}"
    )

    print(
        f"Total allocated cells: "
        f"{uniform_total_cells:,}"
    )

    print(
        f"Active cells: "
        f"{uniform_active_cells:,}"
    )

    print()
    print("LATENCY")

    print(
        f"  Mean:   "
        f"{uniform_stats['mean_ms']:.3f} ms"
    )

    print(
        f"  Median: "
        f"{uniform_stats['median_ms']:.3f} ms"
    )

    print(
        f"  Min:    "
        f"{uniform_stats['min_ms']:.3f} ms"
    )

    print(
        f"  Max:    "
        f"{uniform_stats['max_ms']:.3f} ms"
    )

    print(
        f"  Std:    "
        f"{uniform_stats['std_ms']:.3f} ms"
    )

    print()
    print("MEMORY")

    print(
        f"  Dense array payload: "
        f"{format_bytes(uniform_dense_memory)}"
    )

    print(
        f"  Compact active-cell equivalent: "
        f"{format_bytes(uniform_compact_memory)}"
    )

    # =================================================================
    # 2. DISTANCE ADAPTIVE
    # =================================================================

    print()
    print("-" * 70)
    print("2. DISTANCE-ADAPTIVE MAP")
    print("-" * 70)

    adaptive_map, adaptive_stats = benchmark_function(
        build_distance_adaptive_map,
        benchmark_xyz,
    )

    adaptive_cells = (
        adaptive_map.cell_count
    )

    adaptive_memory = (
        estimate_compact_cell_memory_bytes(
            adaptive_cells
        )
    )

    print()
    print("DISTANCE POLICY")

    print(
        "  0–10 m   -> 5 cm"
    )

    print(
        "  10–25 m  -> 10 cm"
    )

    print(
        "  25–50 m  -> 20 cm"
    )

    print(
        "  50–100 m -> 50 cm"
    )

    print()
    print(
        f"Active cells: "
        f"{adaptive_cells:,}"
    )

    print()
    print("LATENCY")

    print(
        f"  Mean:   "
        f"{adaptive_stats['mean_ms']:.3f} ms"
    )

    print(
        f"  Median: "
        f"{adaptive_stats['median_ms']:.3f} ms"
    )

    print(
        f"  Min:    "
        f"{adaptive_stats['min_ms']:.3f} ms"
    )

    print(
        f"  Max:    "
        f"{adaptive_stats['max_ms']:.3f} ms"
    )

    print(
        f"  Std:    "
        f"{adaptive_stats['std_ms']:.3f} ms"
    )

    print()
    print("MEMORY")

    print(
        f"  Compact active-cell payload: "
        f"{format_bytes(adaptive_memory)}"
    )

    # =================================================================
    # RESOLUTION DISTRIBUTION
    # =================================================================

    if adaptive_cells > 0:

        resolutions = (
            adaptive_map.resolutions()
        )

        print()
        print("RESOLUTION DISTRIBUTION")

        for resolution in sorted(
            np.unique(resolutions)
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
                count
                / adaptive_cells
                * 100.0
            )

            print(
                f"  {resolution * 100:.0f} cm: "
                f"{count:,} cells "
                f"({percentage:.2f}%)"
            )

    # =================================================================
    # 3. A9 HIERARCHICAL INDEXING
    # =================================================================

    print()
    print("-" * 70)
    print("3. A9 HIERARCHICAL INDEXING")
    print("-" * 70)

    hierarchy_results = {}

    for level in range(4):

        cells, stats = benchmark_function(
            hierarchy_cells_for_points,
            benchmark_xyz,
            level,
        )

        cell_count = len(cells)

        memory = (
            estimate_hierarchy_memory_bytes(
                cell_count
            )
        )

        resolution = (
            0.05 * (2 ** level)
        )

        hierarchy_results[level] = {
            "cells": cell_count,
            "memory": memory,
            "stats": stats,
            "resolution": resolution,
        }

        print()
        print(
            f"Level {level}"
        )

        print(
            f"  Resolution: "
            f"{resolution:.2f} m"
        )

        print(
            f"  Unique cells: "
            f"{cell_count:,}"
        )

        print(
            f"  Mean latency: "
            f"{stats['mean_ms']:.3f} ms"
        )

        print(
            f"  Median latency: "
            f"{stats['median_ms']:.3f} ms"
        )

        print(
            f"  Compact index memory: "
            f"{format_bytes(memory)}"
        )

    # =================================================================
    # CALCULATIONS
    # =================================================================

    cell_reduction = (
        percentage_reduction(
            uniform_active_cells,
            adaptive_cells,
        )
    )

    compact_memory_reduction = (
        percentage_reduction(
            uniform_compact_memory,
            adaptive_memory,
        )
    )

    dense_memory_reduction = (
        percentage_reduction(
            uniform_dense_memory,
            adaptive_memory,
        )
    )

    latency_change = (
        percentage_change(
            uniform_stats["mean_ms"],
            adaptive_stats["mean_ms"],
        )
    )

    latency_ratio = (
        adaptive_stats["mean_ms"]
        / uniform_stats["mean_ms"]
    )

    # =================================================================
    # FINAL COMPARISON
    # =================================================================

    print()
    print("=" * 70)
    print("A10 REFINED COMPARISON")
    print("=" * 70)

    print()
    print("COMMON INPUT")

    print(
        f"  LiDAR points: "
        f"{len(benchmark_xyz):,}"
    )

    print(
        "  Horizontal radius: "
        "100 m"
    )

    print(
        "  Common square extent: "
        "[-100, +100] m"
    )

    print()
    print("ACTIVE CELLS")

    print(
        f"  Uniform 5 cm:       "
        f"{uniform_active_cells:,}"
    )

    print(
        f"  Distance-adaptive:  "
        f"{adaptive_cells:,}"
    )

    print(
        f"  Reduction:          "
        f"{cell_reduction:.2f}%"
    )

    print()
    print("MEMORY — DENSE STORAGE")

    print(
        f"  Uniform dense:      "
        f"{format_bytes(uniform_dense_memory)}"
    )

    print(
        f"  Adaptive compact:   "
        f"{format_bytes(adaptive_memory)}"
    )

    print(
        f"  Reduction:          "
        f"{dense_memory_reduction:.2f}%"
    )

    print()
    print("MEMORY — COMPACT ACTIVE-CELL STORAGE")

    print(
        f"  Uniform compact:    "
        f"{format_bytes(uniform_compact_memory)}"
    )

    print(
        f"  Adaptive compact:   "
        f"{format_bytes(adaptive_memory)}"
    )

    print(
        f"  Reduction:          "
        f"{compact_memory_reduction:.2f}%"
    )

    print()
    print("LATENCY")

    print(
        f"  Uniform mean:       "
        f"{uniform_stats['mean_ms']:.3f} ms"
    )

    print(
        f"  Adaptive mean:      "
        f"{adaptive_stats['mean_ms']:.3f} ms"
    )

    print(
        f"  Adaptive / uniform: "
        f"{latency_ratio:.2f}x"
    )

    print(
        f"  Latency change:     "
        f"{latency_change:+.2f}%"
    )

    # =================================================================
    # INTERPRETATION
    # =================================================================

    print()
    print("=" * 70)
    print("A10 INTERPRETATION")
    print("=" * 70)

    print()

    print(
        "1. CELL SPARSITY"
    )

    print(
        f"   Distance adaptation reduces active cells "
        f"by {cell_reduction:.2f}%."
    )

    print()

    print(
        "2. DENSE STORAGE BENEFIT"
    )

    print(
        f"   Compared with a dense 5 cm grid, the "
        f"adaptive representation reduces the"
    )

    print(
        f"   estimated numeric payload by "
        f"{dense_memory_reduction:.2f}%."
    )

    print()

    print(
        "3. REPRESENTATION-LEVEL BENEFIT"
    )

    print(
        f"   When both maps are represented compactly "
        f"using active cells only,"
    )

    print(
        f"   the adaptive representation requires "
        f"{compact_memory_reduction:.2f}% less"
    )

    print(
        "   numeric storage."
    )

    print()

    print(
        "4. LATENCY"
    )

    print(
        f"   The current adaptive prototype is "
        f"{latency_ratio:.2f}x slower"
    )

    print(
        "   than the vectorized uniform baseline."
    )

    print()

    print(
        "5. ENGINEERING INTERPRETATION"
    )

    print(
        "   This latency gap is currently an "
        "implementation bottleneck,"
    )

    print(
        "   not evidence that adaptive mapping "
        "is fundamentally slower."
    )

    print()

    print(
        "   The current adaptive implementation "
        "uses Python-level dictionary/list"
    )

    print(
        "   aggregation and should be optimized "
        "before making real-time claims."
    )

    print()
    print("=" * 70)
    print("A10 REFINED BENCHMARK COMPLETE")
    print("=" * 70)


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()