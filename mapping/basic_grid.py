from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class UniformGrid2D:
    """
    Uniform 2.5D elevation grid.

    Each XY cell stores:
        - point count
        - minimum elevation
        - maximum elevation
        - mean elevation
        - elevation variance
    """

    resolution: float
    x_range: tuple[float, float]
    y_range: tuple[float, float]

    count: np.ndarray
    z_min: np.ndarray
    z_max: np.ndarray
    z_mean: np.ndarray
    z_var: np.ndarray

    @property
    def height(self) -> int:
        """Number of cells along Y."""
        return self.count.shape[0]

    @property
    def width(self) -> int:
        """Number of cells along X."""
        return self.count.shape[1]

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        """Return world-coordinate centers of all grid cells."""

        x_centers = (
            self.x_range[0]
            + (np.arange(self.width) + 0.5) * self.resolution
        )

        y_centers = (
            self.y_range[0]
            + (np.arange(self.height) + 0.5) * self.resolution
        )

        return x_centers, y_centers


def build_uniform_grid(
    xyz: np.ndarray,
    resolution: float = 0.10,
    x_range: tuple[float, float] = (-50.0, 50.0),
    y_range: tuple[float, float] = (-50.0, 50.0),
) -> UniformGrid2D:
    """
    Build a uniform 2.5D elevation grid from XYZ LiDAR points.

    Args:
        xyz: NumPy array with shape (N, 3).
        resolution: Cell size in metres.
        x_range: Minimum and maximum X coordinates.
        y_range: Minimum and maximum Y coordinates.

    Returns:
        UniformGrid2D containing elevation statistics.
    """

    xyz = np.asarray(xyz, dtype=np.float64)

    # Validate input
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(
            f"xyz must have shape (N, 3), got {xyz.shape}"
        )

    if resolution <= 0:
        raise ValueError("resolution must be greater than zero")

    if x_range[1] <= x_range[0]:
        raise ValueError("x_range must be (min, max)")

    if y_range[1] <= y_range[0]:
        raise ValueError("y_range must be (min, max)")

    if not np.isfinite(xyz).all():
        raise ValueError("xyz contains NaN or infinite values")

    # Calculate grid dimensions
    width = int(
        np.ceil((x_range[1] - x_range[0]) / resolution)
    )

    height = int(
        np.ceil((y_range[1] - y_range[0]) / resolution)
    )

    # Allocate grid
    count = np.zeros(
        (height, width),
        dtype=np.int32,
    )

    z_min = np.full(
        (height, width),
        np.nan,
        dtype=np.float64,
    )

    z_max = np.full(
        (height, width),
        np.nan,
        dtype=np.float64,
    )

    z_mean = np.full(
        (height, width),
        np.nan,
        dtype=np.float64,
    )

    z_var = np.full(
        (height, width),
        np.nan,
        dtype=np.float64,
    )

    # Empty input
    if xyz.shape[0] == 0:
        return UniformGrid2D(
            resolution=resolution,
            x_range=x_range,
            y_range=y_range,
            count=count,
            z_min=z_min,
            z_max=z_max,
            z_mean=z_mean,
            z_var=z_var,
        )

    # Convert world coordinates to grid indices
    ix = np.floor(
        (xyz[:, 0] - x_range[0]) / resolution
    ).astype(np.int64)

    iy = np.floor(
        (xyz[:, 1] - y_range[0]) / resolution
    ).astype(np.int64)

    # Keep only points inside the requested map
    valid = (
        (ix >= 0)
        & (ix < width)
        & (iy >= 0)
        & (iy < height)
    )

    ix = ix[valid]
    iy = iy[valid]
    z = xyz[valid, 2]

    # No points inside the map
    if z.size == 0:
        return UniformGrid2D(
            resolution=resolution,
            x_range=x_range,
            y_range=y_range,
            count=count,
            z_min=z_min,
            z_max=z_max,
            z_mean=z_mean,
            z_var=z_var,
        )

    # Flatten 2D indices
    flat_index = iy * width + ix

    # Count
    flat_count = np.bincount(
        flat_index,
        minlength=height * width,
    )

    count = flat_count.reshape(
        height,
        width,
    ).astype(np.int32)

    # Sum and sum of squares
    flat_sum = np.bincount(
        flat_index,
        weights=z,
        minlength=height * width,
    )

    flat_sum_sq = np.bincount(
        flat_index,
        weights=z * z,
        minlength=height * width,
    )

    populated = flat_count > 0

    # Mean
    flat_mean = np.full(
        height * width,
        np.nan,
        dtype=np.float64,
    )

    flat_mean[populated] = (
        flat_sum[populated]
        / flat_count[populated]
    )

    # Variance
    flat_var = np.full(
        height * width,
        np.nan,
        dtype=np.float64,
    )

    flat_var[populated] = (
        flat_sum_sq[populated]
        / flat_count[populated]
        - flat_mean[populated] ** 2
    )

    # Remove tiny negative floating-point errors
    flat_var[populated] = np.maximum(
        flat_var[populated],
        0.0,
    )

    # Minimum and maximum elevation
    #
    # Start with +/- infinity so np.minimum.at and
    # np.maximum.at can correctly accumulate values.
    flat_min = np.full(
        height * width,
        np.inf,
        dtype=np.float64,
    )

    flat_max = np.full(
        height * width,
        -np.inf,
        dtype=np.float64,
    )

    np.minimum.at(
        flat_min,
        flat_index,
        z,
    )

    np.maximum.at(
        flat_max,
        flat_index,
        z,
    )

    # Empty cells should remain explicitly represented as NaN.
    flat_min[~populated] = np.nan
    flat_max[~populated] = np.nan

    # Convert back to 2D
    z_min = flat_min.reshape(
        height,
        width,
    )

    z_max = flat_max.reshape(
        height,
        width,
    )

    z_mean = flat_mean.reshape(
        height,
        width,
    )

    z_var = flat_var.reshape(
        height,
        width,
    )

    return UniformGrid2D(
        resolution=resolution,
        x_range=x_range,
        y_range=y_range,
        count=count,
        z_min=z_min,
        z_max=z_max,
        z_mean=z_mean,
        z_var=z_var,
    )