import numpy as np

from mapping.basic_grid import UniformGrid2D
from mapping.terrain_features import (
    compute_height_span,
    compute_height_discontinuity,
    compute_roughness,
    compute_slope,
)


def _normalize_penalty(
    values: np.ndarray,
    safe_limit: float,
    unsafe_limit: float,
) -> np.ndarray:
    """
    Convert a terrain feature into a penalty between 0 and 1.

    0 = completely safe according to the configured threshold
    1 = completely unsafe according to the configured threshold

    Values between the safe and unsafe limits are linearly interpolated.
    """

    if unsafe_limit <= safe_limit:
        raise ValueError(
            "unsafe_limit must be greater than safe_limit."
        )

    penalty = np.full_like(
        values,
        np.nan,
        dtype=np.float64,
    )

    valid = np.isfinite(values)

    penalty[valid] = (
        values[valid] - safe_limit
    ) / (
        unsafe_limit - safe_limit
    )

    penalty[valid] = np.clip(
        penalty[valid],
        0.0,
        1.0,
    )

    return penalty


def compute_traversability(
    grid: UniformGrid2D,
    slope_safe: float = 5.0,
    slope_unsafe: float = 25.0,
    roughness_safe: float = 0.02,
    roughness_unsafe: float = 0.20,
    height_span_safe: float = 0.05,
    height_span_unsafe: float = 0.50,
    discontinuity_safe: float = 0.05,
    discontinuity_unsafe: float = 0.30,
) -> np.ndarray:
    """
    Compute a geometry-only traversability score.

    Output:
        1.0 = highly traversable
        0.0 = highly non-traversable
        NaN = no usable LiDAR information

    Current version uses:
        - slope
        - roughness
        - height span
        - height discontinuity

    Semantic information and occupancy will be incorporated later.
    """

    # ------------------------------------------------------------
    # Compute terrain features
    # ------------------------------------------------------------

    slope = compute_slope(grid)

    roughness = compute_roughness(grid)

    height_span = compute_height_span(grid)

    discontinuity = compute_height_discontinuity(grid)

    # ------------------------------------------------------------
    # Convert each feature into a 0-1 penalty
    # ------------------------------------------------------------

    slope_penalty = _normalize_penalty(
        slope,
        safe_limit=slope_safe,
        unsafe_limit=slope_unsafe,
    )

    roughness_penalty = _normalize_penalty(
        roughness,
        safe_limit=roughness_safe,
        unsafe_limit=roughness_unsafe,
    )

    height_span_penalty = _normalize_penalty(
        height_span,
        safe_limit=height_span_safe,
        unsafe_limit=height_span_unsafe,
    )

    discontinuity_penalty = _normalize_penalty(
        discontinuity,
        safe_limit=discontinuity_safe,
        unsafe_limit=discontinuity_unsafe,
    )

    # ------------------------------------------------------------
    # Stack penalties
    # ------------------------------------------------------------

    penalties = np.stack(
        [
            slope_penalty,
            roughness_penalty,
            height_span_penalty,
            discontinuity_penalty,
        ],
        axis=0,
    )

    # ------------------------------------------------------------
    # Calculate mean penalty without warnings
    #
    # Empty cells contain only NaN values.
    # np.nanmean() would produce "Mean of empty slice".
    # Instead, explicitly count valid penalties.
    # ------------------------------------------------------------

    finite_penalties = np.isfinite(penalties)

    penalty_sum = np.nansum(
        penalties,
        axis=0,
    )

    penalty_count = np.sum(
        finite_penalties,
        axis=0,
    )

    valid = penalty_count > 0

    mean_penalty = np.full_like(
        grid.z_mean,
        np.nan,
        dtype=np.float64,
    )

    mean_penalty[valid] = (
        penalty_sum[valid]
        / penalty_count[valid]
    )

    # ------------------------------------------------------------
    # Convert penalty into traversability
    # ------------------------------------------------------------

    traversability = np.full_like(
        grid.z_mean,
        np.nan,
        dtype=np.float64,
    )

    traversability[valid] = (
        1.0 - mean_penalty[valid]
    )

    traversability[valid] = np.clip(
        traversability[valid],
        0.0,
        1.0,
    )

    return traversability