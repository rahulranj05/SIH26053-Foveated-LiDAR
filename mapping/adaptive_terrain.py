from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mapping.hierarchical_foveated_mapper import HierarchicalLeafMap


@dataclass(frozen=True)
class AdaptiveTerrainFeatures:
    """Terrain and traversability features for an adaptive leaf map."""

    height_span: np.ndarray
    roughness: np.ndarray
    slope: np.ndarray
    height_discontinuity: np.ndarray
    traversability: np.ndarray

    @property
    def size(self) -> int:
        return int(self.height_span.size)


def _validate_leaf_map(
    leaf_map: HierarchicalLeafMap,
) -> None:
    """Validate structural and numeric invariants needed by terrain."""

    n = leaf_map.active_cells

    fields = (
        leaf_map.levels,
        leaf_map.resolutions,
        leaf_map.ix,
        leaf_map.iy,
        leaf_map.z_min,
        leaf_map.z_max,
        leaf_map.z_mean,
        leaf_map.z_variance,
        leaf_map.point_count,
        leaf_map.dominant_reason,
    )

    for field in fields:
        array = np.asarray(field)

        if array.ndim != 1:
            raise ValueError(
                "All leaf-map fields must be one-dimensional."
            )

        if array.size != n:
            raise ValueError(
                "All leaf-map fields must have the same length."
            )

    if leaf_map.input_points < 0:
        raise ValueError(
            "input_points must be non-negative."
        )

    resolutions = np.asarray(
        leaf_map.resolutions,
        dtype=np.float64,
    )

    if np.any(~np.isfinite(resolutions)):
        raise ValueError(
            "Leaf-map resolutions must be finite."
        )

    if np.any(resolutions <= 0.0):
        raise ValueError(
            "Leaf-map resolutions must be positive."
        )

    point_count = np.asarray(
        leaf_map.point_count,
        dtype=np.int64,
    )

    if np.any(point_count < 0):
        raise ValueError(
            "Leaf-map point counts must be non-negative."
        )

    z_min = np.asarray(
        leaf_map.z_min,
        dtype=np.float64,
    )

    z_max = np.asarray(
        leaf_map.z_max,
        dtype=np.float64,
    )

    z_mean = np.asarray(
        leaf_map.z_mean,
        dtype=np.float64,
    )

    z_variance = np.asarray(
        leaf_map.z_variance,
        dtype=np.float64,
    )

    if np.any(~np.isfinite(z_min)):
        raise ValueError(
            "Leaf-map z_min must be finite."
        )

    if np.any(~np.isfinite(z_max)):
        raise ValueError(
            "Leaf-map z_max must be finite."
        )

    if np.any(~np.isfinite(z_mean)):
        raise ValueError(
            "Leaf-map z_mean must be finite."
        )

    if np.any(~np.isfinite(z_variance)):
        raise ValueError(
            "Leaf-map z_variance must be finite."
        )

    if np.any(z_max < z_min):
        raise ValueError(
            "Leaf-map z_max cannot be below z_min."
        )


def compute_height_span(
    leaf_map: HierarchicalLeafMap,
) -> np.ndarray:
    """Return vertical height span for every adaptive cell."""

    _validate_leaf_map(leaf_map)

    return np.maximum(
        np.asarray(
            leaf_map.z_max,
            dtype=np.float64,
        )
        - np.asarray(
            leaf_map.z_min,
            dtype=np.float64,
        ),
        0.0,
    )


def compute_roughness(
    leaf_map: HierarchicalLeafMap,
) -> np.ndarray:
    """Return terrain roughness as height standard deviation."""

    _validate_leaf_map(leaf_map)

    variance = np.maximum(
        np.asarray(
            leaf_map.z_variance,
            dtype=np.float64,
        ),
        0.0,
    )

    return np.sqrt(variance)


def _cell_bounds(
    leaf_map: HierarchicalLeafMap,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Return physical XY bounds for every adaptive cell.

    ix and iy are interpreted in the coordinate system of the
    cell's own resolution.
    """

    resolution = np.asarray(
        leaf_map.resolutions,
        dtype=np.float64,
    )

    ix = np.asarray(
        leaf_map.ix,
        dtype=np.float64,
    )

    iy = np.asarray(
        leaf_map.iy,
        dtype=np.float64,
    )

    x_min = ix * resolution
    x_max = x_min + resolution

    y_min = iy * resolution
    y_max = y_min + resolution

    return (
        x_min,
        x_max,
        y_min,
        y_max,
    )


def _intervals_overlap(
    a_min: float,
    a_max: float,
    b_min: float,
    b_max: float,
    tolerance: float = 1e-9,
) -> bool:
    """Return True when two intervals overlap with positive length."""

    overlap = min(
        a_max,
        b_max,
    ) - max(
        a_min,
        b_min,
    )

    return overlap > tolerance


def _coordinate_key(
    value: float,
    tolerance: float = 1e-9,
) -> int:
    """
    Convert a physical coordinate into a stable integer key.

    The finest production resolution is 0.05 m, so coordinates in
    the hierarchy are exact multiples of that resolution in normal
    operation. Rounding protects against floating-point noise.
    """

    return int(
        round(
            value / tolerance
        )
    )


def _collect_neighbor_pairs(
    leaf_map: HierarchicalLeafMap,
) -> list[tuple[int, int]]:
    """
    Find physically adjacent adaptive cells.

    Two cells are neighbors only when:
      - their physical boundaries lie on the same X/Y coordinate;
      - their boundary intervals overlap with positive length.

    This avoids level-local coordinate collisions.
    """

    (
        x_min,
        x_max,
        y_min,
        y_max,
    ) = _cell_bounds(
        leaf_map
    )

    n = leaf_map.active_cells

    vertical_left: dict[
        int,
        list[tuple[float, float, int]],
    ] = {}

    vertical_right: dict[
        int,
        list[tuple[float, float, int]],
    ] = {}

    horizontal_bottom: dict[
        int,
        list[tuple[float, float, int]],
    ] = {}

    horizontal_top: dict[
        int,
        list[tuple[float, float, int]],
    ] = {}

    for i in range(n):
        left_key = _coordinate_key(
            float(x_min[i])
        )

        right_key = _coordinate_key(
            float(x_max[i])
        )

        bottom_key = _coordinate_key(
            float(y_min[i])
        )

        top_key = _coordinate_key(
            float(y_max[i])
        )

        vertical_left.setdefault(
            left_key,
            [],
        ).append(
            (
                float(y_min[i]),
                float(y_max[i]),
                i,
            )
        )

        vertical_right.setdefault(
            right_key,
            [],
        ).append(
            (
                float(y_min[i]),
                float(y_max[i]),
                i,
            )
        )

        horizontal_bottom.setdefault(
            bottom_key,
            [],
        ).append(
            (
                float(x_min[i]),
                float(x_max[i]),
                i,
            )
        )

        horizontal_top.setdefault(
            top_key,
            [],
        ).append(
            (
                float(x_min[i]),
                float(x_max[i]),
                i,
            )
        )

    pairs: set[tuple[int, int]] = set()

    def match(
        first: list[tuple[float, float, int]],
        second: list[tuple[float, float, int]],
    ) -> None:
        for (
            first_min,
            first_max,
            first_index,
        ) in first:
            for (
                second_min,
                second_max,
                second_index,
            ) in second:
                if first_index == second_index:
                    continue

                if not _intervals_overlap(
                    first_min,
                    first_max,
                    second_min,
                    second_max,
                ):
                    continue

                pair = (
                    min(
                        first_index,
                        second_index,
                    ),
                    max(
                        first_index,
                        second_index,
                    ),
                )

                pairs.add(pair)

    for coordinate, right_cells in vertical_right.items():
        left_cells = vertical_left.get(
            coordinate
        )

        if left_cells is not None:
            match(
                right_cells,
                left_cells,
            )

    for coordinate, top_cells in horizontal_top.items():
        bottom_cells = horizontal_bottom.get(
            coordinate
        )

        if bottom_cells is not None:
            match(
                top_cells,
                bottom_cells,
            )

    return list(pairs)


def _neighbor_statistics(
    leaf_map: HierarchicalLeafMap,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Compute local slope and height discontinuity."""

    _validate_leaf_map(
        leaf_map
    )

    n = leaf_map.active_cells

    slope = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    discontinuity = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    if n == 0:
        return (
            slope,
            discontinuity,
        )

    (
        x_min,
        x_max,
        y_min,
        y_max,
    ) = _cell_bounds(
        leaf_map
    )

    center_x = (
        x_min + x_max
    ) * 0.5

    center_y = (
        y_min + y_max
    ) * 0.5

    z_mean = np.asarray(
        leaf_map.z_mean,
        dtype=np.float64,
    )

    pairs = _collect_neighbor_pairs(
        leaf_map
    )

    for i, j in pairs:
        dx = (
            center_x[j]
            - center_x[i]
        )

        dy = (
            center_y[j]
            - center_y[i]
        )

        horizontal_distance = float(
            np.hypot(
                dx,
                dy,
            )
        )

        if horizontal_distance <= 0.0:
            continue

        height_difference = abs(
            float(z_mean[j])
            - float(z_mean[i])
        )

        local_slope = float(
            np.degrees(
                np.arctan2(
                    height_difference,
                    horizontal_distance,
                )
            )
        )

        if (
            np.isnan(slope[i])
            or local_slope > slope[i]
        ):
            slope[i] = local_slope

        if (
            np.isnan(slope[j])
            or local_slope > slope[j]
        ):
            slope[j] = local_slope

        if (
            np.isnan(discontinuity[i])
            or height_difference > discontinuity[i]
        ):
            discontinuity[i] = (
                height_difference
            )

        if (
            np.isnan(discontinuity[j])
            or height_difference > discontinuity[j]
        ):
            discontinuity[j] = (
                height_difference
            )

    return (
        slope,
        discontinuity,
    )


def compute_slope(
    leaf_map: HierarchicalLeafMap,
) -> np.ndarray:
    """Return steepest local slope in degrees."""

    slope, _ = _neighbor_statistics(
        leaf_map
    )

    return slope


def compute_height_discontinuity(
    leaf_map: HierarchicalLeafMap,
) -> np.ndarray:
    """Return maximum neighboring height discontinuity."""

    _, discontinuity = _neighbor_statistics(
        leaf_map
    )

    return discontinuity


def _normalize_penalty(
    values: np.ndarray,
    safe_limit: float,
    unsafe_limit: float,
) -> np.ndarray:
    """Convert a terrain quantity into a [0, 1] penalty."""

    if safe_limit < 0.0:
        raise ValueError(
            "safe_limit must be non-negative."
        )

    if unsafe_limit <= safe_limit:
        raise ValueError(
            "unsafe_limit must be greater than safe_limit."
        )

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    penalty = np.full(
        values.shape,
        np.nan,
        dtype=np.float64,
    )

    finite = np.isfinite(
        values
    )

    penalty[finite] = np.clip(
        (
            values[finite]
            - safe_limit
        )
        / (
            unsafe_limit
            - safe_limit
        ),
        0.0,
        1.0,
    )

    return penalty


def compute_traversability(
    terrain: AdaptiveTerrainFeatures,
    *,
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
    Compute geometry-only traversability.

    1.0 = highly traversable
    0.0 = highly non-traversable
    NaN = insufficient terrain information
    """

    slope_penalty = _normalize_penalty(
        terrain.slope,
        slope_safe,
        slope_unsafe,
    )

    roughness_penalty = _normalize_penalty(
        terrain.roughness,
        roughness_safe,
        roughness_unsafe,
    )

    height_span_penalty = _normalize_penalty(
        terrain.height_span,
        height_span_safe,
        height_span_unsafe,
    )

    discontinuity_penalty = _normalize_penalty(
        terrain.height_discontinuity,
        discontinuity_safe,
        discontinuity_unsafe,
    )

    penalties = np.stack(
        (
            slope_penalty,
            roughness_penalty,
            height_span_penalty,
            discontinuity_penalty,
        ),
        axis=0,
    )

    valid_count = np.sum(
        np.isfinite(
            penalties
        ),
        axis=0,
    )

    penalty_sum = np.nansum(
        penalties,
        axis=0,
    )

    traversability = np.full(
        terrain.size,
        np.nan,
        dtype=np.float64,
    )

    valid = valid_count > 0

    traversability[valid] = (
        1.0
        - (
            penalty_sum[valid]
            / valid_count[valid]
        )
    )

    return np.clip(
        traversability,
        0.0,
        1.0,
    )


def compute_adaptive_terrain_features(
    leaf_map: HierarchicalLeafMap,
    *,
    slope_safe: float = 5.0,
    slope_unsafe: float = 25.0,
    roughness_safe: float = 0.02,
    roughness_unsafe: float = 0.20,
    height_span_safe: float = 0.05,
    height_span_unsafe: float = 0.50,
    discontinuity_safe: float = 0.05,
    discontinuity_unsafe: float = 0.30,
) -> AdaptiveTerrainFeatures:
    """Compute the complete adaptive terrain feature layer."""

    height_span = compute_height_span(
        leaf_map
    )

    roughness = compute_roughness(
        leaf_map
    )

    slope, height_discontinuity = (
        _neighbor_statistics(
            leaf_map
        )
    )

    terrain = AdaptiveTerrainFeatures(
        height_span=height_span,
        roughness=roughness,
        slope=slope,
        height_discontinuity=height_discontinuity,
        traversability=np.full(
            leaf_map.active_cells,
            np.nan,
            dtype=np.float64,
        ),
    )

    traversability = compute_traversability(
        terrain,
        slope_safe=slope_safe,
        slope_unsafe=slope_unsafe,
        roughness_safe=roughness_safe,
        roughness_unsafe=roughness_unsafe,
        height_span_safe=height_span_safe,
        height_span_unsafe=height_span_unsafe,
        discontinuity_safe=discontinuity_safe,
        discontinuity_unsafe=discontinuity_unsafe,
    )

    return AdaptiveTerrainFeatures(
        height_span=height_span,
        roughness=roughness,
        slope=slope,
        height_discontinuity=height_discontinuity,
        traversability=traversability,
    )


__all__ = [
    "AdaptiveTerrainFeatures",
    "compute_adaptive_terrain_features",
    "compute_height_discontinuity",
    "compute_height_span",
    "compute_roughness",
    "compute_slope",
    "compute_traversability",
]