from dataclasses import dataclass

import numpy as np


@dataclass
class VectorizedAdaptiveCell:
    x_index: int
    y_index: int
    resolution: float
    point_count: int
    z_min: float
    z_max: float
    z_mean: float
    z_var: float

    @property
    def x_min(self) -> float:
        return self.x_index * self.resolution

    @property
    def y_min(self) -> float:
        return self.y_index * self.resolution

    @property
    def x_max(self) -> float:
        return (self.x_index + 1) * self.resolution

    @property
    def y_max(self) -> float:
        return (self.y_index + 1) * self.resolution

    @property
    def x_center(self) -> float:
        return self.x_min + self.resolution / 2.0

    @property
    def y_center(self) -> float:
        return self.y_min + self.resolution / 2.0


@dataclass
class VectorizedAdaptiveMap:
    cells: list[VectorizedAdaptiveCell]

    @property
    def cell_count(self) -> int:
        return len(self.cells)

    def resolutions(self) -> np.ndarray:
        if not self.cells:
            return np.array([], dtype=np.float64)

        return np.array(
            [cell.resolution for cell in self.cells],
            dtype=np.float64,
        )


def resolution_for_distance(distance: np.ndarray) -> np.ndarray:
    """
    Assign mapping resolution based only on horizontal distance.

    Distance bands:

        0–10 m    -> 5 cm
        10–25 m   -> 10 cm
        25–50 m   -> 20 cm
        50–100 m  -> 50 cm
        >100 m    -> ignored
    """
    distance = np.asarray(distance, dtype=np.float64)

    resolution = np.full(
        distance.shape,
        np.nan,
        dtype=np.float64,
    )

    valid = np.isfinite(distance)

    resolution[
        valid & (distance < 10.0)
    ] = 0.05

    resolution[
        valid & (distance >= 10.0) & (distance < 25.0)
    ] = 0.10

    resolution[
        valid & (distance >= 25.0) & (distance < 50.0)
    ] = 0.20

    resolution[
        valid & (distance >= 50.0) & (distance <= 100.0)
    ] = 0.50

    return resolution


def _aggregate_sorted_groups(
    x_indices: np.ndarray,
    y_indices: np.ndarray,
    resolutions: np.ndarray,
    z_values: np.ndarray,
) -> list[VectorizedAdaptiveCell]:
    """
    Aggregate points that belong to the same adaptive cell.

    Inputs must already be sorted by:

        resolution
        x_index
        y_index
    """

    if len(z_values) == 0:
        return []

    new_group = np.empty(
        len(z_values),
        dtype=bool,
    )

    new_group[0] = True

    if len(z_values) > 1:
        new_group[1:] = (
            (resolutions[1:] != resolutions[:-1])
            | (x_indices[1:] != x_indices[:-1])
            | (y_indices[1:] != y_indices[:-1])
        )

    group_starts = np.flatnonzero(new_group)

    group_ends = np.r_[
        group_starts[1:],
        len(z_values),
    ]

    point_counts = group_ends - group_starts

    z_min = np.minimum.reduceat(
        z_values,
        group_starts,
    )

    z_max = np.maximum.reduceat(
        z_values,
        group_starts,
    )

    z_sum = np.add.reduceat(
        z_values,
        group_starts,
    )

    z_squared_sum = np.add.reduceat(
        z_values * z_values,
        group_starts,
    )

    z_mean = z_sum / point_counts

    z_var = (
        z_squared_sum / point_counts
        - z_mean * z_mean
    )

    # Floating-point roundoff can produce tiny
    # negative variance values.
    z_var = np.maximum(
        z_var,
        0.0,
    )

    group_resolutions = resolutions[group_starts]
    group_x_indices = x_indices[group_starts]
    group_y_indices = y_indices[group_starts]

    cells = [
        VectorizedAdaptiveCell(
            x_index=int(x_index),
            y_index=int(y_index),
            resolution=float(resolution),
            point_count=int(point_count),
            z_min=float(z_min_value),
            z_max=float(z_max_value),
            z_mean=float(z_mean_value),
            z_var=float(z_var_value),
        )
        for (
            resolution,
            x_index,
            y_index,
            point_count,
            z_min_value,
            z_max_value,
            z_mean_value,
            z_var_value,
        ) in zip(
            group_resolutions,
            group_x_indices,
            group_y_indices,
            point_counts,
            z_min,
            z_max,
            z_mean,
            z_var,
        )
    ]

    return cells


def build_vectorized_adaptive_map(
    xyz: np.ndarray,
) -> VectorizedAdaptiveMap:
    """
    Build the A11 vectorized distance-adaptive 2.5D map.

    This preserves the A8 distance policy:

        0–10 m    -> 5 cm
        10–25 m   -> 10 cm
        25–50 m   -> 20 cm
        50–100 m  -> 50 cm
        >100 m    -> ignored

    A8 used:

        Python loop
        dictionary
        Python lists

    A11 uses:

        NumPy vectorized indexing
        sorting
        vectorized group aggregation

    The spatial policy itself is unchanged.
    """

    xyz = np.asarray(
        xyz,
        dtype=np.float64,
    )

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(
            "xyz must have shape (N, 3)."
        )

    if not np.all(np.isfinite(xyz)):
        raise ValueError(
            "xyz must contain only finite values."
        )

    if len(xyz) == 0:
        return VectorizedAdaptiveMap(
            cells=[]
        )

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    horizontal_distance = np.sqrt(
        x * x + y * y
    )

    resolutions = resolution_for_distance(
        horizontal_distance
    )

    valid = np.isfinite(resolutions)

    if not np.any(valid):
        return VectorizedAdaptiveMap(
            cells=[]
        )

    x = x[valid]
    y = y[valid]
    z = z[valid]
    resolutions = resolutions[valid]

    x_indices = np.floor(
        x / resolutions
    ).astype(np.int64)

    y_indices = np.floor(
        y / resolutions
    ).astype(np.int64)

    # Sort by:
    # resolution -> x_index -> y_index
    #
    # This gives deterministic output.
    order = np.lexsort(
        (
            y_indices,
            x_indices,
            resolutions,
        )
    )

    sorted_x_indices = x_indices[order]
    sorted_y_indices = y_indices[order]
    sorted_resolutions = resolutions[order]
    sorted_z = z[order]

    cells = _aggregate_sorted_groups(
        x_indices=sorted_x_indices,
        y_indices=sorted_y_indices,
        resolutions=sorted_resolutions,
        z_values=sorted_z,
    )

    return VectorizedAdaptiveMap(
        cells=cells
    )