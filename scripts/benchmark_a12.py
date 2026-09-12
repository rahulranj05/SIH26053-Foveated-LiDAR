from pathlib import Path
import sys
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from mapping.basic_grid import build_uniform_grid
from mapping.distance_adaptive import build_distance_adaptive_map
from mapping.vectorized_adaptive import build_vectorized_adaptive_map


DATASET_FRAME = (
    PROJECT_ROOT
    / "datasets"
    / "TEST"
    / "000000.bin"
)


RADIUS_METERS = 100.0

WARMUP_RUNS = 3
TIMED_RUNS = 20


def load_kitti_xyz(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"SemanticKITTI test frame not found:\n{path}"
        )

    points = np.fromfile(
        path,
        dtype=np.float32,
    )

    if len(points) % 4 != 0:
        raise ValueError(
            "SemanticKITTI .bin file does not contain "
            "a multiple of 4 float32 values."
        )

    points = points.reshape(-1, 4)

    xyz = points[:, :3].astype(
        np.float64,
        copy=False,
    )

    if not np.all(np.isfinite(xyz)):
        raise ValueError(
            "LiDAR frame contains non-finite XYZ values."
        )

    return xyz


def filter_radius(
    xyz: np.ndarray,
    radius: float,
) -> np.ndarray:
    horizontal_distance = np.sqrt(
        xyz[:, 0] ** 2
        + xyz[:, 1] ** 2
    )

    return xyz[
        horizontal_distance <= radius
    ]


def benchmark(
    function,
    *args,
):
    for _ in range(WARMUP_RUNS):
        function(*args)

    timings = []

    for _ in range(TIMED_RUNS):
        start = time.perf_counter()

        result = function(*args)

        elapsed = (
            time.perf_counter() - start
        ) * 1000.0

        timings.append(elapsed)

    timings = np.asarray(
        timings,
        dtype=np.float64,
    )

    return result, {
        "mean": float(np.mean(timings)),
        "median": float(np.median(timings)),
        "minimum": float(np.min(timings)),
        "maximum": float(np.max(timings)),
        "std": float(np.std(timings)),
        "p95": float(np.percentile(timings, 95)),
    }


def uniform_mapper(
    xyz: np.ndarray,
):
    return build_uniform_grid(
        xyz,
        resolution=0.05,
        x_range=(-100.0, 100.0),
        y_range=(-100.0, 100.0),
    )


def count_uniform_active_cells(
    grid,
) -> int:
    return int(
        np.count_nonzero(
            grid.count > 0
        )
    )


def compact_numeric_memory_bytes(
    cell_count: int,
) -> int:
    """
    Theoretical compact numeric payload.

    Per active cell:
        3 int64 values
        5 float64 values

    Total:
        64 bytes per cell
    """

    bytes_per_cell = (
        3 * 8
        + 5 * 8
    )

    return (
        cell_count
        * bytes_per_cell
    )


def format_bytes(
    value: int,
) -> str:
    value = float(value)

    units = [
        "B",
        "KB",
        "MB",
        "GB",
    ]

    for unit in units:
        if value < 1024.0:
            return f"{value:.2f} {unit}"

        value /= 1024.0

    return f"{value:.2f} TB"


def print_timing(
    name: str,
    stats: dict,
):
    print(name)

    print(
        f"  Mean latency : "
        f"{stats['mean']:.3f} ms"
    )

    print(
        f"  Median       : "
        f"{stats['median']:.3f} ms"
    )

    print(
        f"  Minimum      : "
        f"{stats['minimum']:.3f} ms"
    )

    print(
        f"  Maximum      : "
        f"{stats['maximum']:.3f} ms"
    )

    print(
        f"  Std deviation: "
        f"{stats['std']:.3f} ms"
    )

    print(
        f"  P95 latency  : "
        f"{stats['p95']:.3f} ms"
    )

    print()


def main():
    print()
    print("=" * 70)
    print(
        "FOVEAMAP — A12 OPTIMIZED MAPPER BENCHMARK"
    )
    print("=" * 70)

    print()
    print(
        f"Dataset frame : {DATASET_FRAME}"
    )
    print(
        f"Radius        : {RADIUS_METERS:.0f} m"
    )
    print(
        f"Warmup runs   : {WARMUP_RUNS}"
    )
    print(
        f"Timed runs    : {TIMED_RUNS}"
    )

    print()
    print("Loading SemanticKITTI frame...")

    xyz = load_kitti_xyz(
        DATASET_FRAME
    )

    xyz = filter_radius(
        xyz,
        RADIUS_METERS,
    )

    print(
        f"Input points  : {len(xyz):,}"
    )

    print()
    print("-" * 70)
    print("1. UNIFORM 5 CM MAPPER")
    print("-" * 70)

    uniform_result, uniform_stats = benchmark(
        uniform_mapper,
        xyz,
    )

    uniform_active = (
        count_uniform_active_cells(
            uniform_result
        )
    )

    uniform_dense_cells = (
        uniform_result.count.size
    )

    uniform_dense_memory = (
        uniform_dense_cells
        * 8
        * 5
    )

    uniform_compact_memory = (
        compact_numeric_memory_bytes(
            uniform_active
        )
    )

    print_timing(
        "Uniform timing:",
        uniform_stats,
    )

    print(
        f"  Grid shape        : "
        f"{uniform_result.count.shape}"
    )

    print(
        f"  Dense cells       : "
        f"{uniform_dense_cells:,}"
    )

    print(
        f"  Active cells      : "
        f"{uniform_active:,}"
    )

    print(
        f"  Dense memory      : "
        f"{format_bytes(uniform_dense_memory)}"
    )

    print(
        f"  Compact equivalent: "
        f"{format_bytes(uniform_compact_memory)}"
    )

    print()
    print("-" * 70)
    print("2. A8 ORIGINAL DISTANCE-ADAPTIVE MAPPER")
    print("-" * 70)

    a8_result, a8_stats = benchmark(
        build_distance_adaptive_map,
        xyz,
    )

    a8_active = (
        a8_result.cell_count
    )

    a8_compact_memory = (
        compact_numeric_memory_bytes(
            a8_active
        )
    )

    print_timing(
        "A8 timing:",
        a8_stats,
    )

    print(
        f"  Active cells      : "
        f"{a8_active:,}"
    )

    print(
        f"  Compact equivalent: "
        f"{format_bytes(a8_compact_memory)}"
    )

    print()
    print("-" * 70)
    print("3. A11 VECTORIZED DISTANCE-ADAPTIVE MAPPER")
    print("-" * 70)

    a11_result, a11_stats = benchmark(
        build_vectorized_adaptive_map,
        xyz,
    )

    a11_active = (
        a11_result.cell_count
    )

    a11_compact_memory = (
        compact_numeric_memory_bytes(
            a11_active
        )
    )

    print_timing(
        "A11 timing:",
        a11_stats,
    )

    print(
        f"  Active cells      : "
        f"{a11_active:,}"
    )

    print(
        f"  Compact equivalent: "
        f"{format_bytes(a11_compact_memory)}"
    )

    print()
    print("=" * 70)
    print("A12 COMPARISON")
    print("=" * 70)

    print()

    print(
        f"Uniform active cells : "
        f"{uniform_active:,}"
    )

    print(
        f"A8 active cells      : "
        f"{a8_active:,}"
    )

    print(
        f"A11 active cells     : "
        f"{a11_active:,}"
    )

    print()

    print(
        f"A8/A11 cell match    : "
        f"{a8_active == a11_active}"
    )

    print()

    adaptive_reduction = (
        1.0
        - (
            a11_active
            / uniform_active
        )
    ) * 100.0

    print(
        f"Adaptive active-cell reduction "
        f"vs uniform: "
        f"{adaptive_reduction:.2f}%"
    )

    print()

    compact_memory_reduction = (
        1.0
        - (
            a11_compact_memory
            / uniform_compact_memory
        )
    ) * 100.0

    print(
        f"Compact memory reduction "
        f"vs uniform: "
        f"{compact_memory_reduction:.2f}%"
    )

    print()

    a8_to_a11_speedup = (
        a8_stats["median"]
        / a11_stats["median"]
    )

    uniform_to_a11_speedup = (
        uniform_stats["median"]
        / a11_stats["median"]
    )

    print(
        f"A11 speedup vs A8: "
        f"{a8_to_a11_speedup:.2f}x"
    )

    print(
        f"A11 speedup vs uniform: "
        f"{uniform_to_a11_speedup:.2f}x"
    )

    print()

    print(
        f"A11 median latency: "
        f"{a11_stats['median']:.3f} ms"
    )

    print(
        f"A11 P95 latency   : "
        f"{a11_stats['p95']:.3f} ms"
    )

    print()

    if a11_active == a8_active:
        print(
            "RESULT: A11 preserves A8 active-cell count."
        )
    else:
        print(
            "WARNING: A11 active-cell count differs "
            "from A8."
        )

    print()

    print("=" * 70)
    print("A12 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()