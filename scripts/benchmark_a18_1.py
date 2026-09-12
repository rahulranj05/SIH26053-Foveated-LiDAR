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
    reason_distribution,
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
        dtype=object,
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


def build_foveated_map_vectorized(
    xyz: np.ndarray,
    resolution: np.ndarray,
    dominant_reason: np.ndarray,
) -> FoveatedMap:
    """
    A18.1 optimized mapper.

    Same logical output as A18, but performs the
    expensive numerical aggregation in vectorized
    NumPy operations.

    The only remaining Python loop is the final
    construction of FoveatedCell objects.
    """

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
        dtype=object,
    )

    n = len(xyz)

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
        return FoveatedMap(
            cells=[],
            input_points=0,
            active_cells=0,
        )

    if not np.all(
        np.isfinite(xyz)
    ):
        raise ValueError(
            "xyz must contain finite values."
        )

    # ---------------------------------------------------------
    # Convert resolution to hierarchy level.
    # ---------------------------------------------------------

    levels = np.empty(
        n,
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

    valid = np.isin(
        levels,
        [0, 1, 2, 3],
    )

    if not np.all(valid):
        raise ValueError(
            "Resolution contains values outside "
            "the A9 hierarchy."
        )

    # ---------------------------------------------------------
    # Numerically stable cell indexing.
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Sort all points by hierarchical cell.
    # ---------------------------------------------------------

    order = np.lexsort(
        (
            iy,
            ix,
            levels,
        )
    )

    sorted_levels = levels[order]
    sorted_ix = ix[order]
    sorted_iy = iy[order]
    sorted_z = xyz[:, 2][order]

    sorted_reasons = (
        dominant_reason[order]
    )

    # ---------------------------------------------------------
    # Find cell boundaries.
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Vectorized point counts.
    # ---------------------------------------------------------

    point_counts = (
        ends - starts
    )

    # ---------------------------------------------------------
    # Vectorized z minimum / maximum.
    #
    # reduceat performs one reduction per group.
    # ---------------------------------------------------------

    z_min = np.minimum.reduceat(
        sorted_z,
        starts,
    )

    z_max = np.maximum.reduceat(
        sorted_z,
        starts,
    )

    # ---------------------------------------------------------
    # Vectorized sums for mean / variance.
    # ---------------------------------------------------------

    z_sum = np.add.reduceat(
        sorted_z,
        starts,
    )

    z_squared_sum = np.add.reduceat(
        sorted_z * sorted_z,
        starts,
    )

    counts_float = (
        point_counts.astype(
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

    # Floating-point noise can create tiny negative
    # variance values such as -1e-16.
    z_variance = np.maximum(
        z_variance,
        0.0,
    )

    # ---------------------------------------------------------
    # Efficient dominant-reason calculation.
    #
    # There are only five possible A17 reasons, so compare
    # against each reason globally instead of calling
    # np.unique() separately for every cell.
    # ---------------------------------------------------------

    reason_names = np.array(
        [
            "DISTANCE",
            "KINEMATIC",
            "PREDICTED_PATH",
            "SEMANTIC",
            "DYNAMIC",
        ],
        dtype=object,
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

    # Prefix counts for each reason.
    reason_counts = np.empty(
        (
            len(reason_names),
            cell_count,
        ),
        dtype=np.int64,
    )

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

        cumulative_with_zero = (
            np.concatenate(
                (
                    np.array(
                        [0],
                        dtype=np.int64,
                    ),
                    cumulative,
                )
            )
        )

        reason_counts[
            code
        ] = (
            cumulative_with_zero[
                ends
            ]
            - cumulative_with_zero[
                starts
            ]
        )

    dominant_reason_codes = (
        np.argmax(
            reason_counts,
            axis=0,
        )
    )

    # ---------------------------------------------------------
    # Construct final cells.
    # ---------------------------------------------------------

    cells: list[FoveatedCell] = []

    for i in range(
        cell_count
    ):
        level = int(
            sorted_levels[
                starts[i]
            ]
        )

        cells.append(
            FoveatedCell(
                level=level,
                ix=int(
                    sorted_ix[
                        starts[i]
                    ]
                ),
                iy=int(
                    sorted_iy[
                        starts[i]
                    ]
                ),
                resolution=LEVEL_RESOLUTIONS[
                    level
                ],
                z_min=float(
                    z_min[i]
                ),
                z_max=float(
                    z_max[i]
                ),
                z_mean=float(
                    z_mean[i]
                ),
                z_variance=float(
                    z_variance[i]
                ),
                point_count=int(
                    point_counts[i]
                ),
                dominant_reason=str(
                    reason_names[
                        dominant_reason_codes[i]
                    ]
                ),
            )
        )

    return FoveatedMap(
        cells=cells,
        input_points=n,
        active_cells=cell_count,
    )


def benchmark(
    xyz: np.ndarray,
    resolution: np.ndarray,
    dominant_reason: np.ndarray,
) -> tuple[
    FoveatedMap,
    np.ndarray,
]:
    for _ in range(
        WARMUP_RUNS
    ):
        build_foveated_map_vectorized(
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
            build_foveated_map_vectorized(
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


def main() -> None:
    print("=" * 70)
    print(
        "FOVEAMAP A18.1 — VECTORIZED FOVEATED 2.5D MAPPER"
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
    print("A18.1 RESULTS")
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
    print("Resolution distribution:")

    distribution = (
        resolution_distribution(
            result
        )
    )

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
    print("Latency:")

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

    print()
    print("=" * 70)
    print("A18.1 BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()