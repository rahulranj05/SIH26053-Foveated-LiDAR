from __future__ import annotations

from dataclasses import dataclass

import numpy as np


LEVEL_RESOLUTIONS = {
    0: 0.05,
    1: 0.10,
    2: 0.20,
    3: 0.40,
}

RESOLUTION_TO_LEVEL = {
    0.05: 0,
    0.10: 1,
    0.20: 2,
    0.40: 3,
}


@dataclass(frozen=True)
class FoveatedCell:
    """
    One adaptive 2.5D map cell.
    """

    level: int
    ix: int
    iy: int

    resolution: float

    z_min: float
    z_max: float
    z_mean: float
    z_variance: float

    point_count: int

    dominant_reason: str


@dataclass
class FoveatedMap:
    """
    Sparse hierarchical 2.5D map.

    Cells are indexed by:

        (level, ix, iy)

    where level determines the spatial resolution.
    """

    cells: list[FoveatedCell]

    input_points: int
    active_cells: int


def level_from_resolution(
    resolution: np.ndarray,
) -> np.ndarray:
    """
    Convert resolution values into hierarchy levels.
    """

    resolution = np.asarray(
        resolution,
        dtype=np.float64,
    )

    levels = np.full(
        resolution.shape,
        -1,
        dtype=np.int8,
    )

    for value, level in RESOLUTION_TO_LEVEL.items():
        mask = np.isclose(
            resolution,
            value,
        )

        levels[mask] = level

    if np.any(levels < 0):
        raise ValueError(
            "Resolution contains values outside "
            "the A9 hierarchy."
        )

    return levels


def cell_indices(
    x: np.ndarray,
    y: np.ndarray,
    resolution: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert world coordinates into resolution-aware cell indices.

    A tiny scale-aware numerical tolerance is added before floor()
    so mathematically exact grid boundaries such as:

        1.20 / 0.05 = 24

    are not incorrectly represented as:

        23.999999999...

    The tolerance is far smaller than any meaningful LiDAR
    spatial measurement.
    """

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    resolution = np.asarray(
        resolution,
        dtype=np.float64,
    )

    x_scaled = x / resolution
    y_scaled = y / resolution

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
    ).astype(np.int64)

    iy = np.floor(
        y_scaled + tolerance
    ).astype(np.int64)

    return ix, iy


def _aggregate_group(
    z: np.ndarray,
    start: int,
    end: int,
) -> tuple[float, float, float, float]:
    """
    Compute z statistics for one sorted group.
    """

    values = z[start:end]

    z_min = float(
        np.min(values)
    )

    z_max = float(
        np.max(values)
    )

    z_mean = float(
        np.mean(values)
    )

    z_variance = float(
        np.var(values)
    )

    return (
        z_min,
        z_max,
        z_mean,
        z_variance,
    )


def build_foveated_map(
    xyz: np.ndarray,
    resolution: np.ndarray,
    dominant_reason: np.ndarray,
) -> FoveatedMap:
    """
    Build a sparse hierarchical 2.5D map from point-wise
    foveation decisions.

    Args:
        xyz:
            Nx3 point cloud.

        resolution:
            N point-wise resolutions. Must contain only
            0.05, 0.10, 0.20, or 0.40 metres.

        dominant_reason:
            N dominant foveation-reason strings.

    Returns:
        FoveatedMap containing adaptive cells.
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

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(
            "xyz must have shape (N, 3)."
        )

    if resolution.ndim != 1:
        raise ValueError(
            "resolution must be a 1D array."
        )

    if dominant_reason.ndim != 1:
        raise ValueError(
            "dominant_reason must be a 1D array."
        )

    n = len(xyz)

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

    levels = level_from_resolution(
        resolution
    )

    ix, iy = cell_indices(
        xyz[:, 0],
        xyz[:, 1],
        resolution,
    )

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
    sorted_reasons = dominant_reason[order]

    same_as_previous = (
        (sorted_levels[1:] == sorted_levels[:-1])
        & (sorted_ix[1:] == sorted_ix[:-1])
        & (sorted_iy[1:] == sorted_iy[:-1])
    )

    boundaries = np.flatnonzero(
        ~same_as_previous
    ) + 1

    starts = np.concatenate(
        (
            np.array([0]),
            boundaries,
        )
    )

    ends = np.concatenate(
        (
            boundaries,
            np.array([n]),
        )
    )

    cells: list[FoveatedCell] = []

    for start, end in zip(
        starts,
        ends,
    ):
        level = int(
            sorted_levels[start]
        )

        cell_x = int(
            sorted_ix[start]
        )

        cell_y = int(
            sorted_iy[start]
        )

        resolution_value = (
            LEVEL_RESOLUTIONS[level]
        )

        (
            z_min,
            z_max,
            z_mean,
            z_variance,
        ) = _aggregate_group(
            sorted_z,
            start,
            end,
        )

        group_reasons = (
            sorted_reasons[start:end]
        )

        unique_reasons, counts = np.unique(
            group_reasons,
            return_counts=True,
        )

        dominant_reason_value = str(
            unique_reasons[
                np.argmax(counts)
            ]
        )

        cells.append(
            FoveatedCell(
                level=level,
                ix=cell_x,
                iy=cell_y,
                resolution=resolution_value,
                z_min=z_min,
                z_max=z_max,
                z_mean=z_mean,
                z_variance=z_variance,
                point_count=end - start,
                dominant_reason=(
                    dominant_reason_value
                ),
            )
        )

    return FoveatedMap(
        cells=cells,
        input_points=n,
        active_cells=len(cells),
    )


def resolution_distribution(
    foveated_map: FoveatedMap,
) -> dict[float, int]:
    """
    Count active cells at each hierarchy resolution.
    """

    distribution = {
        0.05: 0,
        0.10: 0,
        0.20: 0,
        0.40: 0,
    }

    for cell in foveated_map.cells:
        distribution[
            cell.resolution
        ] += 1

    return distribution


def level_distribution(
    foveated_map: FoveatedMap,
) -> dict[int, int]:
    """
    Count active cells at each hierarchy level.
    """

    distribution = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
    }

    for cell in foveated_map.cells:
        distribution[
            cell.level
        ] += 1

    return distribution


def reason_distribution(
    foveated_map: FoveatedMap,
) -> dict[str, int]:
    """
    Count active cells by dominant foveation reason.
    """

    distribution: dict[str, int] = {}

    for cell in foveated_map.cells:
        reason = cell.dominant_reason

        distribution[reason] = (
            distribution.get(
                reason,
                0,
            )
            + 1
        )

    return distribution