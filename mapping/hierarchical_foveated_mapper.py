"""
A18.3 — Hierarchical Foveated 2.5D Mapper.

This module converts per-point desired resolutions into a compact,
non-overlapping hierarchical 2.5D leaf map.

Hierarchy:

    level 0 -> 0.05 m
    level 1 -> 0.10 m
    level 2 -> 0.20 m
    level 3 -> 0.40 m

Important invariant:

    A final map contains leaves only.

    A parent cell and one of its descendants must never coexist in the
    final representation.

Because of that invariant, if multiple points occupy the same parent
cell and even one point requires finer resolution, the entire active
group must continue refining. A coarse point therefore cannot block a
fine-resolution requirement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


# ============================================================================
# HIERARCHY CONSTANTS
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

_REASON_TO_ID: dict[str, int] = {
    reason: index
    for index, reason in enumerate(REASON_NAMES)
}


# ============================================================================
# DATA MODEL
# ============================================================================


@dataclass(frozen=True)
class HierarchicalLeafMap:
    """
    Compact representation of the final adaptive 2.5D leaf map.

    Each row represents one final leaf cell.

    Fields:

        levels:
            Hierarchy level of each cell.

        resolutions:
            Physical resolution of each cell.

        ix, iy:
            Integer horizontal cell coordinates.

        z_min:
            Minimum observed z in the cell.

        z_max:
            Maximum observed z in the cell.

        z_mean:
            Mean observed z in the cell.

        z_variance:
            Population variance of observed z.

        point_count:
            Number of input points represented by the cell.

        dominant_reason:
            Dominant foveation reason for the cell.

        input_points:
            Total number of input points supplied to the mapper.
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
        """Return the number of active final leaf cells."""
        return int(self.levels.size)

    @property
    def level(self) -> np.ndarray:
        """Compatibility alias for ``levels``."""
        return self.levels

    @property
    def resolution(self) -> np.ndarray:
        """Compatibility alias for ``resolutions``."""
        return self.resolutions


# ============================================================================
# BASIC CONVERSION HELPERS
# ============================================================================


def level_from_resolution(
    resolution: float,
) -> int:
    """
    Convert a supported physical resolution to its hierarchy level.

    Raises:
        ValueError:
            If the resolution is unsupported.
    """

    value = float(resolution)

    for known_resolution, level in RESOLUTION_TO_LEVEL.items():

        if np.isclose(
            value,
            known_resolution,
            rtol=1e-9,
            atol=1e-12,
        ):
            return level

    raise ValueError(
        "Unsupported resolution "
        f"{resolution!r}. Expected one of "
        f"{tuple(LEVEL_RESOLUTIONS.values())}."
    )


def resolution_from_level(
    level: int,
) -> float:
    """
    Convert a hierarchy level to its physical resolution.

    Raises:
        ValueError:
            If the level is unsupported.
    """

    level = int(level)

    if level not in LEVEL_RESOLUTIONS:
        raise ValueError(
            f"Unsupported hierarchy level {level!r}. "
            f"Expected one of {tuple(LEVEL_RESOLUTIONS)}."
        )

    return LEVEL_RESOLUTIONS[level]


# ============================================================================
# INPUT VALIDATION
# ============================================================================


def _validate_xyz(
    xyz: np.ndarray,
) -> np.ndarray:
    """
    Validate and normalize XYZ input.

    Returns:
        float64 array with shape (N, 3).
    """

    array = np.asarray(
        xyz,
        dtype=np.float64,
    )

    if array.ndim != 2:
        raise ValueError(
            "xyz must be a 2D array with shape (N, 3)."
        )

    if array.shape[1] < 3:
        raise ValueError(
            "xyz must contain at least three columns: x, y, z."
        )

    if not np.all(
        np.isfinite(array[:, :3])
    ):
        raise ValueError(
            "xyz contains non-finite values."
        )

    return array


def _validate_inputs(
    xyz: np.ndarray,
    desired_resolution: np.ndarray,
    dominant_reason: Iterable[str],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Validate all mapper inputs.

    Returns:
        validated_xyz,
        desired_levels,
        validated_reasons.
    """

    validated_xyz = _validate_xyz(
        xyz
    )

    n_points = validated_xyz.shape[0]

    resolutions = np.asarray(
        desired_resolution,
        dtype=np.float64,
    )

    if resolutions.ndim != 1:
        raise ValueError(
            "desired_resolution must be a 1D array."
        )

    if resolutions.size != n_points:
        raise ValueError(
            "desired_resolution length must match "
            f"xyz point count ({n_points})."
        )

    if not np.all(
        np.isfinite(resolutions)
    ):
        raise ValueError(
            "desired_resolution contains non-finite values."
        )

    # ------------------------------------------------------------------
    # Vectorized resolution -> hierarchy-level conversion.
    #
    # Production foveation produces only the supported hierarchy
    # resolutions. Matching the complete array in NumPy avoids doing
    # Python-level resolution checks once per LiDAR point.
    # ------------------------------------------------------------------

    known_resolutions = np.asarray(
        tuple(LEVEL_RESOLUTIONS.values()),
        dtype=np.float64,
    )

    known_levels = np.asarray(
        tuple(LEVEL_RESOLUTIONS.keys()),
        dtype=np.int8,
    )

    matches = np.isclose(
        resolutions[:, None],
        known_resolutions[None, :],
        rtol=1e-9,
        atol=1e-12,
    )

    matched = np.any(
        matches,
        axis=1,
    )

    if not np.all(matched):
        invalid_resolution = float(
            resolutions[np.flatnonzero(~matched)[0]]
        )

        raise ValueError(
            "Unsupported resolution "
            f"{invalid_resolution!r}. Expected one of "
            f"{tuple(LEVEL_RESOLUTIONS.values())}."
        )

    desired_levels = known_levels[
        np.argmax(
            matches,
            axis=1,
        )
    ]

    reasons = np.asarray(
        dominant_reason,
        dtype=object,
    )

    if reasons.ndim != 1:
        raise ValueError(
            "dominant_reason must be a 1D array."
        )

    if reasons.size != n_points:
        raise ValueError(
            "dominant_reason length must match "
            f"xyz point count ({n_points})."
        )

    invalid_reasons = sorted(
        {
            str(reason)
            for reason in reasons
            if str(reason) not in _REASON_TO_ID
        }
    )

    if invalid_reasons:
        raise ValueError(
            "dominant_reason contains unsupported values: "
            f"{invalid_reasons}. "
            f"Expected one of {REASON_NAMES}."
        )

    reasons = reasons.astype(
        object,
        copy=False,
    )

    return (
        validated_xyz,
        desired_levels,
        reasons,
    )


# ============================================================================
# CELL INDEXING
# ============================================================================


def _cell_indices(
    xy: np.ndarray,
    resolution: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert XY coordinates to integer cell indices.

    A small scale-aware tolerance is applied before flooring so that
    coordinates which are numerically extremely close to an exact cell
    boundary do not jump unpredictably between neighboring cells.

    The same global origin is used at every hierarchy level.
    """

    resolution = float(resolution)

    if not np.isfinite(resolution) or resolution <= 0.0:
        raise ValueError(
            f"resolution must be positive and finite, got {resolution!r}."
        )

    xy = np.asarray(
        xy,
        dtype=np.float64,
    )

    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(
            "xy must have shape (N, 2)."
        )

    scaled = xy / resolution

    # Tolerance is intentionally tiny relative to the coordinate scale.
    tolerance = (
        1e-10
        * np.maximum(
            1.0,
            np.abs(scaled),
        )
    )

    adjusted = np.where(
        np.abs(
            scaled
            - np.rint(scaled)
        )
        <= tolerance,
        np.rint(scaled),
        scaled,
    )

    ix = np.floor(
        adjusted[:, 0]
    ).astype(
        np.int64
    )

    iy = np.floor(
        adjusted[:, 1]
    ).astype(
        np.int64
    )

    return ix, iy


# ============================================================================
# GROUPING HELPERS
# ============================================================================


def _group_boundaries(
    ix: np.ndarray,
    iy: np.ndarray,
) -> np.ndarray:
    """
    Return boundaries for contiguous groups of equal ix/iy pairs.

    If there are N sorted points and G groups, the returned array has
    G + 1 entries:

        [0, ..., N]

    such that group k occupies:

        boundaries[k]:boundaries[k + 1]
    """

    if ix.size != iy.size:
        raise ValueError(
            "ix and iy must have equal length."
        )

    if ix.size == 0:
        return np.array(
            [0],
            dtype=np.int64,
        )

    changes = (
        (ix[1:] != ix[:-1])
        | (iy[1:] != iy[:-1])
    )

    starts = np.concatenate(
        [
            np.array(
                [0],
                dtype=np.int64,
            ),
            np.flatnonzero(changes)
            .astype(np.int64)
            + 1,
        ]
    )

    return np.concatenate(
        [
            starts,
            np.array(
                [ix.size],
                dtype=np.int64,
            ),
        ]
    )


def _aggregate_group_arrays(
    xyz: np.ndarray,
    sorted_indices: np.ndarray,
    begin: np.ndarray,
    end: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Aggregate Z statistics for contiguous sorted groups.

    Returns:

        z_min,
        z_max,
        z_mean,
        z_variance,
        point_count
    """

    group_count = begin.size

    point_count = (
        end - begin
    ).astype(
        np.int64
    )

    if group_count == 0:
        empty = np.empty(
            0,
            dtype=np.float64,
        )

        return (
            empty.copy(),
            empty.copy(),
            empty.copy(),
            empty.copy(),
            np.empty(
                0,
                dtype=np.int64,
            ),
        )

    z = xyz[
        sorted_indices,
        2,
    ]

    z_min = np.minimum.reduceat(
        z,
        begin,
    )

    z_max = np.maximum.reduceat(
        z,
        begin,
    )

    z_sum = np.add.reduceat(
        z,
        begin,
    )

    z_squared_sum = np.add.reduceat(
        z * z,
        begin,
    )

    z_mean = (
        z_sum
        / point_count
    )

    z_variance = (
        z_squared_sum
        / point_count
        - z_mean * z_mean
    )

    # Protect against tiny negative floating-point roundoff.
    z_variance = np.maximum(
        z_variance,
        0.0,
    )

    return (
        z_min.astype(
            np.float64,
            copy=False,
        ),
        z_max.astype(
            np.float64,
            copy=False,
        ),
        z_mean.astype(
            np.float64,
            copy=False,
        ),
        z_variance.astype(
            np.float64,
            copy=False,
        ),
        point_count,
    )


def _dominant_reasons(
    reasons: np.ndarray,
    begin: np.ndarray,
    end: np.ndarray,
) -> np.ndarray:
    """
    Determine the dominant reason for every group.

    Ties are deterministic and follow REASON_NAMES ordering.
    """

    group_count = begin.size

    output = np.empty(
        group_count,
        dtype=object,
    )

    if group_count == 0:
        return output

    reason_ids = np.empty(
        reasons.size,
        dtype=np.int8,
    )

    for reason, reason_id in _REASON_TO_ID.items():

        reason_ids[
            reasons == reason
        ] = reason_id

    for group_index in range(
        group_count
    ):

        start = begin[
            group_index
        ]

        stop = end[
            group_index
        ]

        counts = np.bincount(
            reason_ids[
                start:stop
            ],
            minlength=len(
                REASON_NAMES
            ),
        )

        output[
            group_index
        ] = REASON_NAMES[
            int(
                np.argmax(counts)
            )
        ]

    return output


# ============================================================================
# INTERNAL FINALIZATION
# ============================================================================


def _append_final_groups_at_level(
    xyz: np.ndarray,
    sorted_indices: np.ndarray,
    sorted_ix: np.ndarray,
    sorted_iy: np.ndarray,
    sorted_reasons: np.ndarray,
    begin: np.ndarray,
    end: np.ndarray,
    group_ids: np.ndarray,
    level: int,
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
    Aggregate and append selected terminal groups.
    """

    if group_ids.size == 0:
        return

    selected_begin = begin[
        group_ids
    ]

    selected_end = end[
        group_ids
    ]

    selected_ix = sorted_ix[
        selected_begin
    ].astype(
        np.int64,
        copy=False,
    )

    selected_iy = sorted_iy[
        selected_begin
    ].astype(
        np.int64,
        copy=False,
    )

    point_counts = (
        selected_end
        - selected_begin
    ).astype(
        np.int64
    )

    (
        z_min,
        z_max,
        z_mean,
        z_variance,
        _,
    ) = _aggregate_group_arrays(
        xyz,
        sorted_indices,
        selected_begin,
        selected_end,
    )

    reasons = _dominant_reasons(
        sorted_reasons,
        selected_begin,
        selected_end,
    )

    final_levels.append(
        np.full(
            group_ids.size,
            level,
            dtype=np.int8,
        )
    )

    final_ix.append(
        selected_ix
    )

    final_iy.append(
        selected_iy
    )

    final_z_min.append(
        z_min
    )

    final_z_max.append(
        z_max
    )

    final_z_mean.append(
        z_mean
    )

    final_z_variance.append(
        z_variance
    )

    final_point_count.append(
        point_counts
    )

    final_reason.append(
        reasons
    )


# ============================================================================
# MAIN HIERARCHICAL MAPPER
# ============================================================================


def build_hierarchical_foveated_map(
    xyz: np.ndarray,
    desired_resolution: np.ndarray,
    dominant_reason: Iterable[str],
) -> HierarchicalLeafMap:
    """
    Build a non-overlapping hierarchical foveated 2.5D leaf map.

    Parameters
    ----------
    xyz:
        Point cloud with shape (N, 3) or (N, >=3).

    desired_resolution:
        Per-point desired resolution. Supported values are:

            0.05
            0.10
            0.20
            0.40

    dominant_reason:
        Per-point foveation reason.

    Returns
    -------
    HierarchicalLeafMap
        Final adaptive leaf map.

    Refinement semantics
    --------------------
    The hierarchy is processed from coarse to fine.

    At each level, points are grouped into cells.

    If all points in a cell are satisfied at the current level or a
    coarser level, that cell becomes a final leaf.

    If at least one point requires finer resolution, the complete active
    group is refined.

    This intentionally uses the MINIMUM required hierarchy level.

    Example:

        requirements = [level 0, level 3]

    means:

        5 cm requirement + 40 cm requirement.

    Since level 0 is finer, the parent must continue refining. The
    coarse point cannot terminate the parent and thereby block the fine
    requirement.

    This design maintains the stronger invariant that no parent and
    descendant leaves coexist in the final map.
    """

    (
        xyz,
        desired_levels,
        dominant_reason,
    ) = _validate_inputs(
        xyz,
        desired_resolution,
        dominant_reason,
    )

    n_points = xyz.shape[0]

    # ------------------------------------------------------------------
    # Empty input.
    # ------------------------------------------------------------------

    if n_points == 0:

        return HierarchicalLeafMap(
            levels=np.empty(
                0,
                dtype=np.int8,
            ),
            resolutions=np.empty(
                0,
                dtype=np.float64,
            ),
            ix=np.empty(
                0,
                dtype=np.int64,
            ),
            iy=np.empty(
                0,
                dtype=np.int64,
            ),
            z_min=np.empty(
                0,
                dtype=np.float64,
            ),
            z_max=np.empty(
                0,
                dtype=np.float64,
            ),
            z_mean=np.empty(
                0,
                dtype=np.float64,
            ),
            z_variance=np.empty(
                0,
                dtype=np.float64,
            ),
            point_count=np.empty(
                0,
                dtype=np.int64,
            ),
            dominant_reason=np.empty(
                0,
                dtype=object,
            ),
            input_points=0,
        )

    # ------------------------------------------------------------------
    # Final output buffers.
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
    # Start with every point active.
    #
    # The first grouping happens at the 40 cm root level.
    # ------------------------------------------------------------------

    active_indices = np.arange(
        n_points,
        dtype=np.int64,
    )

    # ------------------------------------------------------------------
    # Process hierarchy from coarse to fine.
    # ------------------------------------------------------------------

    for level in (
        3,
        2,
        1,
        0,
    ):

        if active_indices.size == 0:
            break

        resolution = LEVEL_RESOLUTIONS[
            level
        ]

        active_xyz = xyz[
            active_indices
        ]

        ix, iy = _cell_indices(
            active_xyz[:, :2],
            resolution,
        )

        # Deterministic spatial ordering.
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
        # IMPORTANT:
        #
        # The minimum required level determines whether a group must
        # refine.
        #
        # Level 0 = finest.
        # Level 3 = coarsest.
        #
        # Therefore:
        #
        #     min([0, 3]) = 0
        #
        # correctly identifies that the group contains a fine
        # requirement.
        # --------------------------------------------------------------

        min_required_level = np.minimum.reduceat(
            sorted_desired_levels,
            begin,
        )

        # --------------------------------------------------------------
        # A group terminates when its finest requirement is satisfied
        # at the current level.
        #
        # Example:
        #
        #     current level = 3
        #     requirements = [0, 3]
        #
        #     min_required_level = 0
        #
        #     0 >= 3 -> False
        #
        # Therefore the group refines.
        # --------------------------------------------------------------

        terminal_mask = (
            min_required_level
            >= level
        )

        terminal_group_ids = np.flatnonzero(
            terminal_mask
        )

        refine_group_ids = np.flatnonzero(
            ~terminal_mask
        )

        # --------------------------------------------------------------
        # Finalize groups that have reached their required resolution.
        # --------------------------------------------------------------

        _append_final_groups_at_level(
            xyz=xyz,
            sorted_indices=sorted_indices,
            sorted_ix=sorted_ix,
            sorted_iy=sorted_iy,
            sorted_reasons=sorted_reasons,
            begin=begin,
            end=end,
            group_ids=terminal_group_ids,
            level=level,
            final_levels=final_levels,
            final_ix=final_ix,
            final_iy=final_iy,
            final_z_min=final_z_min,
            final_z_max=final_z_max,
            final_z_mean=final_z_mean,
            final_z_variance=final_z_variance,
            final_point_count=final_point_count,
            final_reason=final_reason,
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
    Count final leaf cells by physical resolution.
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
    Validate structural properties of the final leaf map.

    Checks:

        - no duplicate leaves
        - correct level/resolution relationship
        - valid integer cell indices
        - no parent/child coexistence
        - positive point counts
        - finite numeric statistics
        - valid reasons
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
    # Array lengths.
    # ------------------------------------------------------------------

    arrays = (
        fmap.levels,
        fmap.resolutions,
        fmap.ix,
        fmap.iy,
        fmap.z_min,
        fmap.z_max,
        fmap.z_mean,
        fmap.z_variance,
        fmap.point_count,
        fmap.dominant_reason,
    )

    if any(
        array.size != n_cells
        for array in arrays
    ):
        return False

    # ------------------------------------------------------------------
    # Level/resolution relationship.
    # ------------------------------------------------------------------

    try:

        expected_resolutions = np.asarray(
            [
                LEVEL_RESOLUTIONS[
                    int(level)
                ]
                for level in fmap.levels
            ],
            dtype=np.float64,
        )

    except (KeyError, ValueError, TypeError):

        return False

    if not np.allclose(
        fmap.resolutions,
        expected_resolutions,
        rtol=1e-9,
        atol=1e-12,
    ):
        return False

    # ------------------------------------------------------------------
    # Cell alignment / integer validity.
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

        if resolution <= 0:
            return False

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

    # ------------------------------------------------------------------
    # Duplicate leaf cells.
    #
    # Same level + ix + iy must never occur twice.
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
    # A level L cell and a level L-1 child must never coexist if the
    # child lies inside that parent's footprint.
    # ------------------------------------------------------------------

    for child_level in (
        0,
        1,
        2,
    ):

        parent_level = (
            child_level + 1
        )

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
    Return True if a parent and child leaf coexist.

    A valid final hierarchical map should return False.
    """

    for child_level in (
        0,
        1,
        2,
    ):

        parent_level = (
            child_level + 1
        )

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