from dataclasses import dataclass

import numpy as np


@dataclass
class AdaptiveCell:
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
class DistanceAdaptiveMap:
    cells: list[AdaptiveCell]

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

    resolution[valid & (distance < 10.0)] = 0.05

    resolution[
        valid
        & (distance >= 10.0)
        & (distance < 25.0)
    ] = 0.10

    resolution[
        valid
        & (distance >= 25.0)
        & (distance < 50.0)
    ] = 0.20

    resolution[
        valid
        & (distance >= 50.0)
        & (distance <= 100.0)
    ] = 0.50

    return resolution


def build_distance_adaptive_map(
    xyz: np.ndarray,
) -> DistanceAdaptiveMap:
    """
    Build a sparse distance-adaptive 2.5D map.

    This is the A8 distance-only prototype.

    Important:
    This version intentionally does NOT implement the
    alignment-safe hierarchy. That is A9.
    """

    xyz = np.asarray(xyz, dtype=np.float64)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(
            "xyz must have shape (N, 3)."
        )

    if not np.all(np.isfinite(xyz)):
        raise ValueError(
            "xyz must contain only finite values."
        )

    if len(xyz) == 0:
        return DistanceAdaptiveMap(cells=[])

    x = xyz[:, 0]
    y = xyz[:, 1]

    horizontal_distance = np.sqrt(
        x**2 + y**2
    )

    resolutions = resolution_for_distance(
        horizontal_distance
    )

    valid = np.isfinite(resolutions)

    if not np.any(valid):
        return DistanceAdaptiveMap(cells=[])

    points = xyz[valid]
    point_resolutions = resolutions[valid]

    groups: dict[
        tuple[float, int, int],
        list[float],
    ] = {}

    for point, resolution in zip(
        points,
        point_resolutions,
    ):
        point_x = point[0]
        point_y = point[1]
        point_z = point[2]

        x_index = int(
            np.floor(point_x / resolution)
        )

        y_index = int(
            np.floor(point_y / resolution)
        )

        key = (
            float(resolution),
            x_index,
            y_index,
        )

        if key not in groups:
            groups[key] = []

        groups[key].append(
            float(point_z)
        )

    cells: list[AdaptiveCell] = []

    for (
        resolution,
        x_index,
        y_index,
    ), z_values in groups.items():

        z_values = np.asarray(
            z_values,
            dtype=np.float64,
        )

        cells.append(
            AdaptiveCell(
                x_index=x_index,
                y_index=y_index,
                resolution=resolution,
                point_count=len(z_values),
                z_min=float(np.min(z_values)),
                z_max=float(np.max(z_values)),
                z_mean=float(np.mean(z_values)),
                z_var=float(np.var(z_values)),
            )
        )

    cells.sort(
        key=lambda cell: (
            cell.resolution,
            cell.x_index,
            cell.y_index,
        )
    )

    return DistanceAdaptiveMap(
        cells=cells
    )