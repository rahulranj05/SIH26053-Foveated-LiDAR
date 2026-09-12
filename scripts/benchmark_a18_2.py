from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

from mapping.foveated_mapper import (
    LEVEL_RESOLUTIONS,
    FoveatedCell,
    FoveatedMap,
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


# ============================================================
# A18.2 COMPACT NUMERIC MAP
# ============================================================

class CompactFoveatedMap:
    """
    Compact NumPy representation of the foveated map.

    Unlike A18/A18.1, this does NOT create one Python
    dataclass object per cell.

    Each cell attribute is stored in a contiguous NumPy array.
    """

    def __init__(
        self,
        level: np.ndarray,
        ix: np.ndarray,
        iy: np.ndarray,
        resolution: np.ndarray,
        z_min: np.ndarray,
        z_max: np.ndarray,
        z_mean: np.ndarray,
        z_variance: np.ndarray,
        point_count: np.ndarray,
        dominant_reason: np.ndarray,
        input_points: int,
    ) -> None:

        self.level = level
        self.ix = ix
        self.iy = iy
        self.resolution = resolution
        self.z_min = z_min
        self.z_max = z_max
        self.z_mean = z_mean
        self.z_variance = z_variance
        self.point_count = point_count
        self.dominant_reason = dominant_reason

        self.input_points = input_points
        self.active_cells = len(level)

    def to_foveated_map(self) -> FoveatedMap:
        """
        Materialize the compact representation into the
        original A18 FoveatedMap dataclass representation.

        This is intentionally NOT used during benchmarking.
        """

        cells: list[FoveatedCell] = []

        for i in range(
            self.active_cells
        ):
            cells.append(
                FoveatedCell(
                    level=int(
                        self.level[i]
                    ),
                    ix=int(
                        self.ix[i]
                    ),
                    iy=int(
                        self.iy[i]
                    ),
                    resolution=float(
                        self.resolution[i]
                    ),
                    z_min=float(
                        self.z_min[i]
                    ),
                    z_max=float(
                        self.z_max[i]
                    ),
                    z_mean=float(
                        self.z_mean[i]
                    ),
                    z_variance=float(
                        self.z_variance[i]
                    ),
                    point_count=int(
                        self.point_count[i]
                    ),
                    dominant_reason=str(
                        self.dominant_reason[i]
                    ),
                )
            )

        return FoveatedMap(
            cells=cells,
            input_points=self.input_points,
            active_cells=self.active_cells,
        )


# ============================================================
# DATA LOADING
# ============================================================

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


# ============================================================
# SYNTHETIC A17-STYLE INPUT
# ============================================================

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


# ============================================================
# RESOLUTION → LEVEL
# ============================================================

def resolution_to_level(
    resolution: np.ndarray,
) -> np.ndarray:

    levels = np.full(
        len(resolution),
        -1,
        dtype=np.int8,
    )

    levels[
        np.isclose(
            resolution,
            0.05,
        )
    ] = 0

    levels[
        np.isclose(
            resolution,
            0.10,
        )
    ] = 1

    levels[
        np.isclose(
            resolution,
            0.20,
        )
    ] = 2

    levels[
        np.isclose(
            resolution,
            0.40,
        )
    ] = 3

    if np.any(
        levels < 0
    ):
        raise ValueError(
            "Resolution contains values outside "
            "the A9 hierarchy."
        )

    return levels


# ============================================================
# CELL INDEXING
# ============================================================

def compute_cell_indices(
    xyz: np.ndarray,
    resolution: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    x_scaled = (
        xyz[:, 0]
        / resolution
    )

    y_scaled = (
        xyz[:, 1]
        / resolution
    )

    tolerance = (
        1e-10
        * np.maximum(
            1.0,
            np.maximum(
                np.abs(x_scaled),
                np.abs(y_scaled),
            ),
        )
    )

    ix = np.floor(
        x_scaled + tolerance
    ).astype(
        np.int64
    )

    iy = np.floor(
        y_scaled + tolerance
    ).astype(
        np.int64
    )

    return (
        ix,
        iy,
    )


# ============================================================
# A18.2 VECTORISED MAP BUILDER
# ============================================================

def build_compact_foveated_map(
    xyz: np.ndarray,
    resolution: np.ndarray,
    dominant_reason: np.ndarray,
) -> CompactFoveatedMap:

    xyz = np.asarray(
        xyz,
        dtype=np.float64,
    )

    resolution = np.asarray(
        resolution,
        dtype=np.float64,
    )

    dominant_reason = np.asarray(
        dominant_reason,
        dtype="U16",
    )

    n = len(xyz)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(
            "xyz must have shape (N, 3)."
        )

    if resolution.ndim != 1:
        raise ValueError(
            "resolution must be 1D."
        )

    if dominant_reason.ndim != 1:
        raise ValueError(
            "dominant_reason must be 1D."
        )

    if len(resolution) != n:
        raise ValueError(
            "resolution length must match xyz."
        )

    if len(dominant_reason) != n:
        raise ValueError(
            "dominant_reason length must match xyz."
        )

    if n == 0:
        empty_i8 = np.empty(
            0,
            dtype=np.int8,
        )

        empty_i64 = np.empty(
            0,
            dtype=np.int64,
        )

        empty_f64 = np.empty(
            0,
            dtype=np.float64,
        )

        empty_reason = np.empty(
            0,
            dtype="U16",
        )

        return CompactFoveatedMap(
            level=empty_i8,
            ix=empty_i64,
            iy=empty_i64,
            resolution=empty_f64,
            z_min=empty_f64,
            z_max=empty_f64,
            z_mean=empty_f64,
            z_variance=empty_f64,
            point_count=empty_i64,
            dominant_reason=empty_reason,
            input_points=0,
        )

    if not np.all(
        np.isfinite(xyz)
    ):
        raise ValueError(
            "xyz must contain finite values."
        )

    # --------------------------------------------------------
    # Resolution → hierarchy level
    # --------------------------------------------------------

    levels = resolution_to_level(
        resolution
    )

    # --------------------------------------------------------
    # Resolution-aware cell coordinates
    # --------------------------------------------------------

    ix, iy = compute_cell_indices(
        xyz,
        resolution,
    )

    # --------------------------------------------------------
    # Sort points by hierarchical cell
    # --------------------------------------------------------

    order = np.lexsort(
        (
            iy,
            ix,
            levels,
        )
    )

    sorted_levels = levels[
        order
    ]

    sorted_ix = ix[
        order
    ]

    sorted_iy = iy[
        order
    ]

    sorted_z = xyz[:, 2][
        order
    ]

    sorted_reasons = (
        dominant_reason[order]
    )

    # --------------------------------------------------------
    # Detect group boundaries
    # --------------------------------------------------------

    if n == 1:

        starts = np.array(
            [0],
            dtype=np.int64,
        )

        ends = np.array(
            [1],
            dtype=np.int64,
        )

    else:

        different = (
            (sorted_levels[1:] != sorted_levels[:-1])
            | (sorted_ix[1:] != sorted_ix[:-1])
            | (sorted_iy[1:] != sorted_iy[:-1])
        )

        boundaries = (
            np.flatnonzero(
                different
            )
            + 1
        )

        starts = np.concatenate(
            (
                np.array(
                    [0],
                    dtype=np.int64,
                ),
                boundaries,
            )
        )

        ends = np.concatenate(
            (
                boundaries,
                np.array(
                    [n],
                    dtype=np.int64,
                ),
            )
        )

    cell_count = len(starts)

    # --------------------------------------------------------
    # Point counts
    # --------------------------------------------------------

    point_count = (
        ends - starts
    )

    # --------------------------------------------------------
    # Z statistics
    # --------------------------------------------------------

    z_min = np.minimum.reduceat(
        sorted_z,
        starts,
    )

    z_max = np.maximum.reduceat(
        sorted_z,
        starts,
    )

    z_sum = np.add.reduceat(
        sorted_z,
        starts,
    )

    z_squared_sum = np.add.reduceat(
        sorted_z * sorted_z,
        starts,
    )

    counts_float = (
        point_count.astype(
            np.float64
        )
    )

    z_mean = (
        z_sum
        / counts_float
    )

    z_variance = (
        z_squared_sum
        / counts_float
        - z_mean * z_mean
    )

    z_variance = np.maximum(
        z_variance,
        0.0,
    )

    # --------------------------------------------------------
    # Dominant reason
    # --------------------------------------------------------

    reason_names = np.array(
        [
            "DISTANCE",
            "KINEMATIC",
            "PREDICTED_PATH",
            "SEMANTIC",
            "DYNAMIC",
        ],
        dtype="U16",
    )

    reason_codes = np.full(
        n,
        -1,
        dtype=np.int8,
    )

    for code, name in enumerate(
        reason_names
    ):

        reason_codes[
            sorted_reasons == name
        ] = code

    reason_counts = np.zeros(
        (
            len(reason_names),
            cell_count,
        ),
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # Count each reason inside each cell
    # --------------------------------------------------------

    for code in range(
        len(reason_names)
    ):

        indicator = (
            reason_codes == code
        ).astype(
            np.int64
        )

        cumulative = np.cumsum(
            indicator
        )

        cumulative = np.concatenate(
            (
                np.array(
                    [0],
                    dtype=np.int64,
                ),
                cumulative,
            )
        )

        reason_counts[
            code
        ] = (
            cumulative[ends]
            - cumulative[starts]
        )

    dominant_reason_codes = (
        np.argmax(
            reason_counts,
            axis=0,
        )
    )

    dominant_reason_values = (
        reason_names[
            dominant_reason_codes
        ]
    )

    # --------------------------------------------------------
    # Compact output
    # --------------------------------------------------------

    return CompactFoveatedMap(
        level=sorted_levels[
            starts
        ].copy(),

        ix=sorted_ix[
            starts
        ].copy(),

        iy=sorted_iy[
            starts
        ].copy(),

        resolution=np.asarray(
            [
                LEVEL_RESOLUTIONS[
                    int(level)
                ]
                for level in sorted_levels[
                    starts
                ]
            ],
            dtype=np.float64,
        ),

        z_min=z_min,

        z_max=z_max,

        z_mean=z_mean,

        z_variance=z_variance,

        point_count=point_count,

        dominant_reason=(
            dominant_reason_values.copy()
        ),

        input_points=n,
    )


# ============================================================
# BENCHMARK
# ============================================================

def benchmark(
    xyz: np.ndarray,
    resolution: np.ndarray,
    dominant_reason: np.ndarray,
) -> tuple[
    CompactFoveatedMap,
    np.ndarray,
]:

    for _ in range(
        WARMUP_RUNS
    ):

        build_compact_foveated_map(
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
            build_compact_foveated_map(
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


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)

    print(
        "FOVEAMAP A18.2 — COMPACT FOVEATED 2.5D MAPPER"
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

    result, timings = benchmark(
        xyz,
        resolution,
        dominant_reason,
    )

    print()

    print("-" * 70)

    print("A18.2 RESULTS")

    print("-" * 70)

    print(
        f"Input points       : "
        f"{result.input_points:,}"
    )

    print(
        f"Active map cells   : "
        f"{result.active_cells:,}"
    )

    print()

    print(
        "Resolution distribution:"
    )

    distribution = {
        0.05: 0,
        0.10: 0,
        0.20: 0,
        0.40: 0,
    }

    for value in result.resolution:

        distribution[
            float(value)
        ] += 1

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

    # --------------------------------------------------------
    # Compact numeric memory
    # --------------------------------------------------------

    numeric_bytes = (
        result.level.nbytes
        + result.ix.nbytes
        + result.iy.nbytes
        + result.resolution.nbytes
        + result.z_min.nbytes
        + result.z_max.nbytes
        + result.z_mean.nbytes
        + result.z_variance.nbytes
        + result.point_count.nbytes
        + result.dominant_reason.nbytes
    )

    print()

    print(
        "Compact numeric payload:"
    )

    print(
        f"  {numeric_bytes / (1024 ** 2):.3f} MB"
    )

    print()

    print("=" * 70)

    print(
        "A18.2 BENCHMARK COMPLETE"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()