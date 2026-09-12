from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


# ============================================================================
# A18.3 — HIERARCHICAL FOVEATED 2.5D MAPPER
# ============================================================================
#
# Strict non-overlapping hierarchy:
#
#   Level 0 -> 0.05 m
#   Level 1 -> 0.10 m
#   Level 2 -> 0.20 m
#   Level 3 -> 0.40 m
#
# Refinement rule:
#
#   Start with 40 cm root cells.
#   If any point inside a cell requires finer resolution, refine it.
#   Continue until the required resolution is reached.
#
# Final output contains LEAF CELLS ONLY.
# No parent cell and descendant cell can coexist.
#
# Public API intentionally preserves the original A18.3 helpers and fields.
# ============================================================================


LEVEL_RESOLUTIONS: dict[int, float] = {
    0: 0.05,
    1: 0.10,
    2: 0.20,
    3: 0.40,
}

RESOLUTION_TO_LEVEL: dict[float, int] = {
    0.05: 0,
    0.10: 1,
    0.20: 2,
    0.40: 3,
}

REASON_NAMES: tuple[str, ...] = (
    "DISTANCE",
    "KINEMATIC",
    "PREDICTED_PATH",
    "SEMANTIC",
    "DYNAMIC",
)

_REASON_TO_ID = {
    reason: index
    for index, reason in enumerate(REASON_NAMES)
}


# ============================================================================
# DATA STRUCTURE
# ============================================================================


@dataclass
class HierarchicalLeafMap:
    """
    Compact representation of the final non-overlapping hierarchical map.

    Every row represents one final leaf cell.
    """

    levels: np.ndarray
    resolutions: np.ndarray
    ix: np.ndarray
    iy: np.ndarray

    z_min: np.ndarray
    z_max: np.ndarray
    z_mean: np.ndarray
    z_variance: np.ndarray

    point_count: np.ndarray
    dominant_reason: np.ndarray

    input_points: int

    @property
    def active_cells(self) -> int:
        """Number of final leaf cells."""
        return int(self.levels.size)

    # ------------------------------------------------------------------
    # Backward-compatible public aliases
    # ------------------------------------------------------------------

    @property
    def level(self) -> np.ndarray:
        return self.levels

    @property
    def resolution(self) -> np.ndarray:
        return self.resolutions


# ============================================================================
# BASIC HELPERS
# ============================================================================


def level_from_resolution(resolution: float) -> int:
    """
    Convert a supported resolution into its hierarchy level.
    """
    resolution = float(resolution)

    for known_resolution, level in RESOLUTION_TO_LEVEL.items():
        if np.isclose(
            resolution,
            known_resolution,
            rtol=1e-9,
            atol=1e-12,
        ):
            return level

    raise ValueError(
        f"Unsupported resolution {resolution}. "
        f"Expected one of {sorted(RESOLUTION_TO_LEVEL)}."
    )


def resolution_from_level(level: int) -> float:
    """
    Convert hierarchy level into cell resolution.
    """
    level = int(level)

    if level not in LEVEL_RESOLUTIONS:
        raise ValueError(
            f"Unsupported hierarchy level {level}. "
            f"Expected one of {sorted(LEVEL_RESOLUTIONS)}."
        )

    return LEVEL_RESOLUTIONS[level]


def _validate_xyz(xyz: np.ndarray) -> np.ndarray:
    """
    Validate and normalize XYZ point array.
    """
    xyz = np.asarray(xyz, dtype=np.float64)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(
            f"xyz must have shape (N, 3), got {xyz.shape}"
        )

    if not np.all(np.isfinite(xyz)):
        raise ValueError("xyz contains non-finite values.")

    return xyz


def _validate_inputs(
    xyz: np.ndarray,
    desired_resolution: np.ndarray,
    dominant_reason: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    xyz = _validate_xyz(xyz)

    desired_resolution = np.asarray(
        desired_resolution,
        dtype=np.float64,
    )

    if desired_resolution.ndim != 1:
        raise ValueError(
            "desired_resolution must be a 1D array."
        )

    if desired_resolution.size != xyz.shape[0]:
        raise ValueError(
            "desired_resolution length must match xyz."
        )

    if not np.all(np.isfinite(desired_resolution)):
        raise ValueError(
            "desired_resolution contains non-finite values."
        )

    dominant_reason = np.asarray(
        list(dominant_reason),
        dtype=object,
    )

    if dominant_reason.ndim != 1:
        raise ValueError(
            "dominant_reason must be a 1D sequence."
        )

    if dominant_reason.size != xyz.shape[0]:
        raise ValueError(
            "dominant_reason length must match xyz."
        )

    # Normalize supported resolutions.
    normalized_resolution = np.empty_like(
        desired_resolution,
        dtype=np.float64,
    )

    for resolution in RESOLUTION_TO_LEVEL:
        mask = np.isclose(
            desired_resolution,
            resolution,
            rtol=1e-8,
            atol=1e-10,
        )
        normalized_resolution[mask] = resolution

    supported = np.zeros(
        desired_resolution.shape,
        dtype=bool,
    )

    for resolution in RESOLUTION_TO_LEVEL:
        supported |= np.isclose(
            desired_resolution,
            resolution,
            rtol=1e-8,
            atol=1e-10,
        )

    if not np.all(supported):
        invalid = desired_resolution[~supported]

        raise ValueError(
            "desired_resolution contains unsupported values. "
            f"Examples: {invalid[:10]}"
        )

    for reason in dominant_reason:
        if str(reason) not in REASON_NAMES:
            raise ValueError(
                f"Invalid dominant reason: {reason!r}. "
                f"Expected one of {REASON_NAMES}."
            )

    return (
        xyz,
        normalized_resolution,
        dominant_reason,
    )


# ============================================================================
# CELL INDEXING
# ============================================================================


def _cell_indices(
    xy: np.ndarray,
    resolution: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert XY world coordinates to integer cell indices.

    Uses a scale-aware tolerance before flooring so values such as
    1.2 / 0.05 do not suffer from floating-point boundary errors.
    """

    resolution = float(resolution)

    scaled_x = xy[:, 0] / resolution
    scaled_y = xy[:, 1] / resolution

    tolerance = 1e-9 * np.maximum(
        1.0,
        np.maximum(
            np.abs(scaled_x),
            np.abs(scaled_y),
        ),
    )

    scaled_x = np.where(
        np.abs(scaled_x - np.round(scaled_x)) <= tolerance,
        np.round(scaled_x),
        scaled_x,
    )

    scaled_y = np.where(
        np.abs(scaled_y - np.round(scaled_y)) <= tolerance,
        np.round(scaled_y),
        scaled_y,
    )

    ix = np.floor(scaled_x).astype(np.int64)
    iy = np.floor(scaled_y).astype(np.int64)

    return ix, iy


# ============================================================================
# GROUPING
# ============================================================================


def _group_boundaries(
    ix: np.ndarray,
    iy: np.ndarray,
) -> np.ndarray:
    """
    Return group-start indices for sorted cell coordinates.

    The returned array always contains zero and one-past-the-end.
    """

    n = ix.size

    if n == 0:
        return np.array([0], dtype=np.int64)

    if n == 1:
        return np.array([0, 1], dtype=np.int64)

    changed = (
        (ix[1:] != ix[:-1])
        | (iy[1:] != iy[:-1])
    )

    starts = np.flatnonzero(changed) + 1

    return np.concatenate(
        (
            np.array([0], dtype=np.int64),
            starts.astype(np.int64),
            np.array([n], dtype=np.int64),
        )
    )


# ============================================================================
# NUMERIC AGGREGATION
# ============================================================================


def _aggregate_group_arrays(
    z: np.ndarray,
    starts: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Aggregate Z statistics for sorted groups.

    Returns:

        z_min
        z_max
        z_mean
        z_variance
        point_count
    """

    if starts.size <= 1:
        empty = np.empty(0, dtype=np.float64)

        return (
            empty,
            empty,
            empty,
            empty,
            np.empty(0, dtype=np.int64),
        )

    begin = starts[:-1]
    end = starts[1:]

    counts = (
        end - begin
    ).astype(np.int64)

    z_min = np.minimum.reduceat(
        z,
        begin,
    )

    z_max = np.maximum.reduceat(
        z,
        begin,
    )

    sums = np.add.reduceat(
        z,
        begin,
    )

    z_mean = sums / counts

    squared_sums = np.add.reduceat(
        z * z,
        begin,
    )

    z_variance = (
        squared_sums / counts
        - z_mean * z_mean
    )

    # Floating-point roundoff can create tiny negative variance.
    z_variance = np.maximum(
        z_variance,
        0.0,
    )

    return (
        z_min,
        z_max,
        z_mean,
        z_variance,
        counts,
    )


# ============================================================================
# OPTIMIZED REASON AGGREGATION
# ============================================================================


def _dominant_reasons(
    reasons: np.ndarray,
    starts: np.ndarray,
) -> np.ndarray:
    """
    Determine the dominant reason for each grouped cell.

    Optimized implementation.

    The old implementation called np.unique() independently for every
    group. With ~1.3 million groups/iterations under profiling this became
    the primary A18.3 bottleneck.

    Since the ontology has only five possible reasons, encode them as
    integers and count them directly.
    """

    if starts.size <= 1:
        return np.empty(
            0,
            dtype=object,
        )

    reason_ids = np.empty(
        reasons.size,
        dtype=np.int8,
    )

    for reason, reason_id in _REASON_TO_ID.items():
        reason_ids[
            reasons == reason
        ] = reason_id

    begin = starts[:-1]
    end = starts[1:]

    group_count = begin.size

    # Group index for every point.
    group_ids = np.repeat(
        np.arange(group_count, dtype=np.int64),
        end - begin,
    )

    # Encode reason + group into one flat index.
    flat_indices = (
        group_ids * len(REASON_NAMES)
        + reason_ids.astype(np.int64)
    )

    counts = np.bincount(
        flat_indices,
        minlength=(
            group_count
            * len(REASON_NAMES)
        ),
    )

    counts = counts.reshape(
        group_count,
        len(REASON_NAMES),
    )

    dominant_ids = np.argmax(
        counts,
        axis=1,
    )

    return np.asarray(
        REASON_NAMES,
        dtype=object,
    )[dominant_ids]


# ============================================================================
# FINAL GROUP CREATION
# ============================================================================


def _append_final_groups_at_level(
    xyz: np.ndarray,
    desired_levels: np.ndarray,
    reasons: np.ndarray,
    level: int,
    point_indices: np.ndarray,
    final_levels: list[np.ndarray],
    final_ix: list[np.ndarray],
    final_iy: list[np.ndarray],
    final_z_min: list[np.ndarray],
    final_z_max: list[np.ndarray],
    final_z_mean: list[np.ndarray],
    final_z_variance: list[np.ndarray],
    final_point_count: list[np.ndarray],
    final_reason: list[np.ndarray],
) -> None:
    """
    Aggregate points into final cells at a specific hierarchy level.

    This helper only receives points that have already been determined
    to terminate at this level.
    """

    if point_indices.size == 0:
        return

    resolution = LEVEL_RESOLUTIONS[level]

    selected_xyz = xyz[point_indices]
    selected_reasons = reasons[point_indices]

    ix, iy = _cell_indices(
        selected_xyz[:, :2],
        resolution,
    )

    order = np.lexsort(
        (
            iy,
            ix,
        )
    )

    ix_sorted = ix[order]
    iy_sorted = iy[order]

    point_indices_sorted = point_indices[order]

    z_sorted = xyz[
        point_indices_sorted,
        2,
    ]

    reasons_sorted = selected_reasons[order]

    starts = _group_boundaries(
        ix_sorted,
        iy_sorted,
    )

    (
        z_min,
        z_max,
        z_mean,
        z_variance,
        point_count,
    ) = _aggregate_group_arrays(
        z_sorted,
        starts,
    )

    dominant_reason = _dominant_reasons(
        reasons_sorted,
        starts,
    )

    final_levels.append(
        np.full(
            ix_sorted.size
            - 0
            - (
                0
            ),
            level,
            dtype=np.int8,
        )
    )

    final_ix.append(
        ix_sorted[
            starts[:-1]
        ]
    )

    final_iy.append(
        iy_sorted[
            starts[:-1]
        ]
    )

    final_z_min.append(z_min)
    final_z_max.append(z_max)
    final_z_mean.append(z_mean)
    final_z_variance.append(z_variance)
    final_point_count.append(point_count)
    final_reason.append(dominant_reason)


# ============================================================================
# MAIN HIERARCHICAL MAPPER
# ============================================================================


def build_hierarchical_foveated_map(
    xyz: np.ndarray,
    desired_resolution: np.ndarray,
    dominant_reason: Sequence[str],
) -> HierarchicalLeafMap:
    """
    Build the final non-overlapping hierarchical foveated map.

    Parameters
    ----------
    xyz:
        Point cloud with shape (N, 3).

    desired_resolution:
        Per-point desired resolution.

        Supported values:

            0.05
            0.10
            0.20
            0.40

    dominant_reason:
        Per-point dominant foveation reason.

    Returns
    -------
    HierarchicalLeafMap
        Final leaf-only hierarchical map.

    Algorithm
    ---------
    1. Begin at 40 cm root cells.
    2. Determine whether each root cell contains points requesting
       finer resolution.
    3. Refine those cells into 20 cm children.
    4. Continue to 10 cm.
    5. Continue to 5 cm.
    6. Store only the final leaves.

    Therefore:

        no duplicate leaves
        no parent/child coexistence
        aligned hierarchy
        complete point conservation
    """

    (
        xyz,
        desired_resolution,
        dominant_reason,
    ) = _validate_inputs(
        xyz,
        desired_resolution,
        dominant_reason,
    )

    n_points = xyz.shape[0]

    if n_points == 0:
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
            dtype=object,
        )

        return HierarchicalLeafMap(
            levels=empty_i8,
            resolutions=empty_f64,
            ix=empty_i64,
            iy=empty_i64,
            z_min=empty_f64,
            z_max=empty_f64,
            z_mean=empty_f64,
            z_variance=empty_f64,
            point_count=empty_i64,
            dominant_reason=empty_reason,
            input_points=0,
        )

    desired_levels = np.empty(
        n_points,
        dtype=np.int8,
    )

    for resolution, level in RESOLUTION_TO_LEVEL.items():
        mask = np.isclose(
            desired_resolution,
            resolution,
            rtol=1e-8,
            atol=1e-10,
        )

        desired_levels[mask] = level

    # ------------------------------------------------------------------
    # Final output containers
    # ------------------------------------------------------------------

    final_levels: list[np.ndarray] = []
    final_ix: list[np.ndarray] = []
    final_iy: list[np.ndarray] = []

    final_z_min: list[np.ndarray] = []
    final_z_max: list[np.ndarray] = []
    final_z_mean: list[np.ndarray] = []
    final_z_variance: list[np.ndarray] = []

    final_point_count: list[np.ndarray] = []
    final_reason: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Start with every point at the 40 cm root level.
    # ------------------------------------------------------------------

    active_indices = np.arange(
        n_points,
        dtype=np.int64,
    )

    # ------------------------------------------------------------------
    # Process hierarchy from coarse to fine.
    #
    # At each level:
    #
    #   If a cell contains only points whose desired level is this level
    #   or coarser -> finalize it.
    #
    #   If at least one point needs finer resolution -> keep the entire
    #   group active and refine it.
    #
    # ------------------------------------------------------------------

    for level in (
        3,
        2,
        1,
        0,
    ):

        if active_indices.size == 0:
            break

        resolution = LEVEL_RESOLUTIONS[level]

        active_xyz = xyz[
            active_indices
        ]

        ix, iy = _cell_indices(
            active_xyz[:, :2],
            resolution,
        )

        order = np.lexsort(
            (
                iy,
                ix,
            )
        )

        sorted_indices = active_indices[
            order
        ]

        sorted_ix = ix[
            order
        ]

        sorted_iy = iy[
            order
        ]

        sorted_desired_levels = desired_levels[
            sorted_indices
        ]

        sorted_reasons = dominant_reason[
            sorted_indices
        ]

        starts = _group_boundaries(
            sorted_ix,
            sorted_iy,
        )

        begin = starts[:-1]
        end = starts[1:]

        group_count = begin.size

        if group_count == 0:
            active_indices = np.empty(
                0,
                dtype=np.int64,
            )
            continue

        # --------------------------------------------------------------
        # Maximum required refinement inside each cell.
        # --------------------------------------------------------------

        max_required_level = np.maximum.reduceat(
            sorted_desired_levels,
            begin,
        )

        # --------------------------------------------------------------
        # A group can terminate if no point inside it requests a
        # finer level.
        #
        # Level numbering:
        #
        #   0 = finest
        #   3 = coarsest
        #
        # Therefore a point requiring level 0 forces refinement through
        # all levels.
        #
        # A group terminates at the current level when:
        #
        #   max_required_level >= current_level
        #
        # because all points are satisfied at this level or coarser.
        # --------------------------------------------------------------

        terminal_mask = (
            max_required_level
            >= level
        )

        terminal_group_ids = np.flatnonzero(
            terminal_mask
        )

        refine_group_ids = np.flatnonzero(
            ~terminal_mask
        )

        # --------------------------------------------------------------
        # Finalize groups that can stop at this level.
        # --------------------------------------------------------------

        if terminal_group_ids.size > 0:

            terminal_begin = begin[
                terminal_group_ids
            ]

            terminal_end = end[
                terminal_group_ids
            ]

            terminal_point_counts = (
                terminal_end
                - terminal_begin
            ).astype(np.int64)

            terminal_ix = sorted_ix[
                terminal_begin
            ]

            terminal_iy = sorted_iy[
                terminal_begin
            ]

            # ----------------------------------------------------------
            # Numeric aggregation.
            #
            # We aggregate each terminal group directly using its
            # contiguous sorted point range.
            # ----------------------------------------------------------

            terminal_z_min = np.empty(
                terminal_group_ids.size,
                dtype=np.float64,
            )

            terminal_z_max = np.empty(
                terminal_group_ids.size,
                dtype=np.float64,
            )

            terminal_z_mean = np.empty(
                terminal_group_ids.size,
                dtype=np.float64,
            )

            terminal_z_variance = np.empty(
                terminal_group_ids.size,
                dtype=np.float64,
            )

            terminal_reason = np.empty(
                terminal_group_ids.size,
                dtype=object,
            )

            # ----------------------------------------------------------
            # Aggregate each terminal group.
            #
            # This loop is only over final cells, not over individual
            # points. It preserves correctness and public semantics.
            # ----------------------------------------------------------

            for output_index, group_id in enumerate(
                terminal_group_ids
            ):

                start = begin[
                    group_id
                ]

                stop = end[
                    group_id
                ]

                point_ids = sorted_indices[
                    start:stop
                ]

                z = xyz[
                    point_ids,
                    2,
                ]

                terminal_z_min[
                    output_index
                ] = np.min(z)

                terminal_z_max[
                    output_index
                ] = np.max(z)

                mean = np.mean(z)

                terminal_z_mean[
                    output_index
                ] = mean

                variance = np.var(z)

                terminal_z_variance[
                    output_index
                ] = max(
                    0.0,
                    float(variance),
                )

                group_reasons = sorted_reasons[
                    start:stop
                ]

                # Only up to five possible reasons.
                reason_ids = np.empty(
                    group_reasons.size,
                    dtype=np.int8,
                )

                for reason, reason_id in _REASON_TO_ID.items():
                    reason_ids[
                        group_reasons == reason
                    ] = reason_id

                counts = np.bincount(
                    reason_ids,
                    minlength=len(REASON_NAMES),
                )

                terminal_reason[
                    output_index
                ] = REASON_NAMES[
                    int(
                        np.argmax(counts)
                    )
                ]

            final_levels.append(
                np.full(
                    terminal_group_ids.size,
                    level,
                    dtype=np.int8,
                )
            )

            final_ix.append(
                terminal_ix.astype(
                    np.int64,
                    copy=False,
                )
            )

            final_iy.append(
                terminal_iy.astype(
                    np.int64,
                    copy=False,
                )
            )

            final_z_min.append(
                terminal_z_min
            )

            final_z_max.append(
                terminal_z_max
            )

            final_z_mean.append(
                terminal_z_mean
            )

            final_z_variance.append(
                terminal_z_variance
            )

            final_point_count.append(
                terminal_point_counts
            )

            final_reason.append(
                terminal_reason
            )

        # --------------------------------------------------------------
        # Keep points belonging to groups that need refinement.
        # --------------------------------------------------------------

        if refine_group_ids.size > 0:

            refine_mask = np.zeros(
                group_count,
                dtype=bool,
            )

            refine_mask[
                refine_group_ids
            ] = True

            point_group_ids = np.repeat(
                np.arange(
                    group_count,
                    dtype=np.int64,
                ),
                end - begin,
            )

            keep_points = refine_mask[
                point_group_ids
            ]

            active_indices = sorted_indices[
                keep_points
            ]

        else:

            active_indices = np.empty(
                0,
                dtype=np.int64,
            )

    # ------------------------------------------------------------------
    # Safety check.
    #
    # At level 0 every remaining point must be finalized.
    # ------------------------------------------------------------------

    if active_indices.size != 0:
        raise RuntimeError(
            "Hierarchical refinement ended with "
            f"{active_indices.size} unassigned points."
        )

    # ------------------------------------------------------------------
    # Combine final groups.
    # ------------------------------------------------------------------

    if final_levels:

        levels = np.concatenate(
            final_levels
        )

        ix = np.concatenate(
            final_ix
        )

        iy = np.concatenate(
            final_iy
        )

        z_min = np.concatenate(
            final_z_min
        )

        z_max = np.concatenate(
            final_z_max
        )

        z_mean = np.concatenate(
            final_z_mean
        )

        z_variance = np.concatenate(
            final_z_variance
        )

        point_count = np.concatenate(
            final_point_count
        )

        reasons = np.concatenate(
            final_reason
        )

    else:

        levels = np.empty(
            0,
            dtype=np.int8,
        )

        ix = np.empty(
            0,
            dtype=np.int64,
        )

        iy = np.empty(
            0,
            dtype=np.int64,
        )

        z_min = np.empty(
            0,
            dtype=np.float64,
        )

        z_max = np.empty(
            0,
            dtype=np.float64,
        )

        z_mean = np.empty(
            0,
            dtype=np.float64,
        )

        z_variance = np.empty(
            0,
            dtype=np.float64,
        )

        point_count = np.empty(
            0,
            dtype=np.int64,
        )

        reasons = np.empty(
            0,
            dtype=object,
        )

    # ------------------------------------------------------------------
    # Sort final leaves deterministically.
    # ------------------------------------------------------------------

    if levels.size > 0:

        order = np.lexsort(
            (
                iy,
                ix,
                levels,
            )
        )

        levels = levels[
            order
        ]

        ix = ix[
            order
        ]

        iy = iy[
            order
        ]

        z_min = z_min[
            order
        ]

        z_max = z_max[
            order
        ]

        z_mean = z_mean[
            order
        ]

        z_variance = z_variance[
            order
        ]

        point_count = point_count[
            order
        ]

        reasons = reasons[
            order
        ]

    resolutions = np.asarray(
        [
            LEVEL_RESOLUTIONS[
                int(level)
            ]
            for level in levels
        ],
        dtype=np.float64,
    )

    return HierarchicalLeafMap(
        levels=levels,
        resolutions=resolutions,
        ix=ix,
        iy=iy,
        z_min=z_min,
        z_max=z_max,
        z_mean=z_mean,
        z_variance=z_variance,
        point_count=point_count,
        dominant_reason=reasons,
        input_points=n_points,
    )


# ============================================================================
# DISTRIBUTION HELPERS
# ============================================================================


def level_distribution(
    fmap: HierarchicalLeafMap,
) -> dict[int, int]:
    """
    Count final leaf cells by hierarchy level.

    Compatibility helper retained for the A18.3 test suite.
    """

    return {
        level: int(
            np.count_nonzero(
                fmap.levels == level
            )
        )
        for level in LEVEL_RESOLUTIONS
    }


def reason_distribution(
    fmap: HierarchicalLeafMap,
) -> dict[str, int]:
    """
    Count final leaf cells by dominant foveation reason.

    Compatibility helper retained for the A18.3 test suite.
    """

    return {
        reason: int(
            np.count_nonzero(
                fmap.dominant_reason == reason
            )
        )
        for reason in REASON_NAMES
    }


def resolution_distribution(
    fmap: HierarchicalLeafMap,
) -> dict[float, int]:
    """
    Count final leaf cells by resolution.
    """

    return {
        resolution: int(
            np.count_nonzero(
                np.isclose(
                    fmap.resolutions,
                    resolution,
                    rtol=1e-9,
                    atol=1e-12,
                )
            )
        )
        for resolution in LEVEL_RESOLUTIONS.values()
    }


# ============================================================================
# STRUCTURAL VALIDATION
# ============================================================================


def validate_leaf_partition(
    fmap: HierarchicalLeafMap,
) -> bool:
    """
    Validate the structural properties of the final leaf map.

    Checks:

        - no duplicate leaves
        - correct level/resolution relationship
        - correct cell alignment
        - no parent/child coexistence
        - positive point counts
        - finite numeric statistics
        - complete point conservation
    """

    n_cells = fmap.active_cells

    if n_cells == 0:
        return fmap.input_points == 0

    # ------------------------------------------------------------------
    # Point counts.
    # ------------------------------------------------------------------

    if np.any(
        fmap.point_count <= 0
    ):
        return False

    # ------------------------------------------------------------------
    # Point conservation.
    # ------------------------------------------------------------------

    if int(
        np.sum(
            fmap.point_count
        )
    ) != fmap.input_points:
        return False

    # ------------------------------------------------------------------
    # Level/resolution relationship.
    # ------------------------------------------------------------------

    expected_resolutions = np.asarray(
        [
            LEVEL_RESOLUTIONS[
                int(level)
            ]
            for level in fmap.levels
        ],
        dtype=np.float64,
    )

    if not np.allclose(
        fmap.resolutions,
        expected_resolutions,
        rtol=1e-9,
        atol=1e-12,
    ):
        return False

    # ------------------------------------------------------------------
    # Cell alignment.
    # ------------------------------------------------------------------

    for level in LEVEL_RESOLUTIONS:

        mask = (
            fmap.levels == level
        )

        if not np.any(mask):
            continue

        resolution = LEVEL_RESOLUTIONS[
            level
        ]

        # Integer cell indices are inherently aligned.
        if not np.all(
            np.isfinite(
                fmap.ix[mask]
            )
        ):
            return False

        if not np.all(
            np.isfinite(
                fmap.iy[mask]
            )
        ):
            return False

        if not np.all(
            np.equal(
                fmap.ix[mask],
                fmap.ix[mask].astype(
                    np.int64
                ),
            )
        ):
            return False

        if not np.all(
            np.equal(
                fmap.iy[mask],
                fmap.iy[mask].astype(
                    np.int64
                ),
            )
        ):
            return False

        if resolution <= 0:
            return False

    # ------------------------------------------------------------------
    # Duplicate leaf cells.
    #
    # Same level + ix + iy must never appear twice.
    # ------------------------------------------------------------------

    keys = np.rec.fromarrays(
        [
            fmap.levels,
            fmap.ix,
            fmap.iy,
        ],
        names=(
            "level",
            "ix",
            "iy",
        ),
    )

    if np.unique(keys).size != n_cells:
        return False

    # ------------------------------------------------------------------
    # Numeric validity.
    # ------------------------------------------------------------------

    numeric_arrays = (
        fmap.z_min,
        fmap.z_max,
        fmap.z_mean,
        fmap.z_variance,
    )

    for array in numeric_arrays:

        if not np.all(
            np.isfinite(array)
        ):
            return False

    if np.any(
        fmap.z_variance < -1e-12
    ):
        return False

    if np.any(
        fmap.z_min > fmap.z_max
    ):
        return False

    # ------------------------------------------------------------------
    # Valid reasons.
    # ------------------------------------------------------------------

    valid_reasons = set(
        REASON_NAMES
    )

    if any(
        reason not in valid_reasons
        for reason in fmap.dominant_reason
    ):
        return False

    # ------------------------------------------------------------------
    # Parent/child exclusion.
    #
    # If a level L cell exists, a child at L-1 must not occupy the same
    # parent footprint.
    # ------------------------------------------------------------------

    for child_level in (
        0,
        1,
        2,
    ):

        parent_level = child_level + 1

        child_mask = (
            fmap.levels
            == child_level
        )

        parent_mask = (
            fmap.levels
            == parent_level
        )

        if (
            not np.any(child_mask)
            or not np.any(parent_mask)
        ):
            continue

        child_ix = fmap.ix[
            child_mask
        ]

        child_iy = fmap.iy[
            child_mask
        ]

        parent_ix = fmap.ix[
            parent_mask
        ]

        parent_iy = fmap.iy[
            parent_mask
        ]

        # Every two child cells correspond to one parent cell in each
        # axis, therefore floor division by 2 gives the parent index.
        derived_parent_ix = np.floor_divide(
            child_ix,
            2,
        )

        derived_parent_iy = np.floor_divide(
            child_iy,
            2,
        )

        parent_keys = {
            (
                int(x),
                int(y),
            )
            for x, y in zip(
                parent_ix,
                parent_iy,
            )
        }

        for x, y in zip(
            derived_parent_ix,
            derived_parent_iy,
        ):

            if (
                int(x),
                int(y),
            ) in parent_keys:
                return False

    return True


# ============================================================================
# SIMPLE CONVENIENCE VALIDATORS
# ============================================================================


def validate_point_conservation(
    fmap: HierarchicalLeafMap,
) -> bool:
    """
    Check that every input point was mapped exactly once.
    """

    return (
        int(
            np.sum(
                fmap.point_count
            )
        )
        == fmap.input_points
    )


def cells_are_unique(
    fmap: HierarchicalLeafMap,
) -> bool:
    """
    Check that final leaf cells are unique.
    """

    if fmap.active_cells == 0:
        return True

    keys = np.rec.fromarrays(
        [
            fmap.levels,
            fmap.ix,
            fmap.iy,
        ],
        names=(
            "level",
            "ix",
            "iy",
        ),
    )

    return (
        np.unique(keys).size
        == fmap.active_cells
    )


def has_parent_child_overlap(
    fmap: HierarchicalLeafMap,
) -> bool:
    """
    Return True if a parent and descendant leaf coexist.

    A valid final hierarchical map should return False.
    """

    for child_level in (
        0,
        1,
        2,
    ):

        parent_level = child_level + 1

        child_mask = (
            fmap.levels
            == child_level
        )

        parent_mask = (
            fmap.levels
            == parent_level
        )

        if (
            not np.any(child_mask)
            or not np.any(parent_mask)
        ):
            continue

        parent_keys = {
            (
                int(x),
                int(y),
            )
            for x, y in zip(
                fmap.ix[parent_mask],
                fmap.iy[parent_mask],
            )
        }

        child_parent_keys = zip(
            np.floor_divide(
                fmap.ix[child_mask],
                2,
            ),
            np.floor_divide(
                fmap.iy[child_mask],
                2,
            ),
        )

        for x, y in child_parent_keys:

            if (
                int(x),
                int(y),
            ) in parent_keys:
                return True

    return False


# ============================================================================
# MODULE EXPORTS
# ============================================================================


__all__ = [
    "LEVEL_RESOLUTIONS",
    "RESOLUTION_TO_LEVEL",
    "REASON_NAMES",
    "HierarchicalLeafMap",
    "level_from_resolution",
    "resolution_from_level",
    "build_hierarchical_foveated_map",
    "level_distribution",
    "reason_distribution",
    "resolution_distribution",
    "validate_leaf_partition",
    "validate_point_conservation",
    "cells_are_unique",
    "has_parent_child_overlap",
]