import numpy as np

from mapping.basic_grid import UniformGrid2D


def compute_height_span(grid: UniformGrid2D) -> np.ndarray:
    """
    Compute vertical height variation inside each grid cell.

    Height span = z_max - z_min.

    Empty cells are returned as NaN.
    """
    height_span = np.full_like(
        grid.z_min,
        np.nan,
        dtype=np.float64,
    )

    populated = grid.count > 0

    height_span[populated] = (
        grid.z_max[populated] - grid.z_min[populated]
    )

    return height_span


def compute_slope(grid: UniformGrid2D) -> np.ndarray:
    """
    Estimate local terrain slope from the grid mean elevation.

    Slope is returned in degrees.

    Empty cells are returned as NaN.

    Note:
    At this stage the grid contains raw LiDAR returns, so vertical
    objects such as vehicles, buildings, vegetation, and poles can
    influence the slope estimate. Semantic terrain filtering will
    be introduced later.
    """
    z = grid.z_mean.astype(np.float64)

    valid = np.isfinite(z)

    if not np.any(valid):
        return np.full_like(
            z,
            np.nan,
            dtype=np.float64,
        )

    # Fill empty cells temporarily so np.gradient does not propagate
    # NaNs throughout the grid.
    filled = z.copy()

    filled[~valid] = np.nanmean(z)

    dz_dy, dz_dx = np.gradient(
        filled,
        grid.resolution,
        grid.resolution,
    )

    slope_radians = np.arctan(
        np.sqrt(
            dz_dx**2 +
            dz_dy**2
        )
    )

    slope_degrees = np.degrees(slope_radians)

    # Restore empty cells.
    slope_degrees[~valid] = np.nan

    return slope_degrees


def compute_roughness(grid: UniformGrid2D) -> np.ndarray:
    """
    Compute local vertical roughness for each grid cell.

    Roughness is defined as the population standard deviation
    of the points' Z values inside the cell.

    Empty cells are returned as NaN.
    """
    roughness = np.full_like(
        grid.z_var,
        np.nan,
        dtype=np.float64,
    )

    populated = grid.count > 0

    roughness[populated] = np.sqrt(
        np.maximum(
            grid.z_var[populated],
            0.0,
        )
    )

    return roughness


def compute_height_discontinuity(
    grid: UniformGrid2D,
) -> np.ndarray:
    """
    Compute the maximum absolute elevation difference between
    each populated cell and its 4-connected populated neighbors.

    Neighbors:
        - above
        - below
        - left
        - right

    Cells with no valid populated neighbor are returned as NaN.

    This implementation explicitly checks neighbor validity so
    np.nanmax never receives an all-NaN slice.
    """
    z = grid.z_mean.astype(np.float64)

    valid = np.isfinite(z)

    discontinuity = np.full_like(
        z,
        np.nan,
        dtype=np.float64,
    )

    if not np.any(valid):
        return discontinuity

    neighbor_differences = []

    # ------------------------------------------------------------
    # Neighbor: above
    # ------------------------------------------------------------

    diff = np.full_like(
        z,
        np.nan,
        dtype=np.float64,
    )

    both_valid = (
        valid[1:, :] &
        valid[:-1, :]
    )

    diff[1:, :][both_valid] = np.abs(
        z[1:, :][both_valid] -
        z[:-1, :][both_valid]
    )

    neighbor_differences.append(diff)

    # ------------------------------------------------------------
    # Neighbor: below
    # ------------------------------------------------------------

    diff = np.full_like(
        z,
        np.nan,
        dtype=np.float64,
    )

    both_valid = (
        valid[:-1, :] &
        valid[1:, :]
    )

    diff[:-1, :][both_valid] = np.abs(
        z[:-1, :][both_valid] -
        z[1:, :][both_valid]
    )

    neighbor_differences.append(diff)

    # ------------------------------------------------------------
    # Neighbor: right
    # ------------------------------------------------------------

    diff = np.full_like(
        z,
        np.nan,
        dtype=np.float64,
    )

    both_valid = (
        valid[:, 1:] &
        valid[:, :-1]
    )

    diff[:, 1:][both_valid] = np.abs(
        z[:, 1:][both_valid] -
        z[:, :-1][both_valid]
    )

    neighbor_differences.append(diff)

    # ------------------------------------------------------------
    # Neighbor: left
    # ------------------------------------------------------------

    diff = np.full_like(
        z,
        np.nan,
        dtype=np.float64,
    )

    both_valid = (
        valid[:, :-1] &
        valid[:, 1:]
    )

    diff[:, :-1][both_valid] = np.abs(
        z[:, :-1][both_valid] -
        z[:, 1:][both_valid]
    )

    neighbor_differences.append(diff)

    # ------------------------------------------------------------
    # Combine the four neighbor differences.
    # ------------------------------------------------------------

    stacked = np.stack(
        neighbor_differences
    )

    # A cell must have at least one valid neighboring cell.
    has_neighbor = np.any(
        np.isfinite(stacked),
        axis=0,
    )

    discontinuity[has_neighbor] = np.nanmax(
        stacked[:, has_neighbor],
        axis=0,
    )

    return discontinuity