from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_VOXEL_SIZE_M = 0.10


@dataclass(frozen=True)
class VoxelizationResult:
    voxel_coordinates: np.ndarray
    point_to_voxel_inverse: np.ndarray
    voxel_to_representative_point: np.ndarray


def voxelize_points(
    xyz: np.ndarray,
    *,
    voxel_size_m: float = DEFAULT_VOXEL_SIZE_M,
) -> VoxelizationResult:
    """
    Frozen S7-E SPVCNN voxelization contract.

    - signed floor(xyz / voxel_size)
    - no origin shift
    - negative voxel coordinates allowed
    - every canonical point maps to one voxel
    - representative point = nearest Euclidean-range point
    - exact range tie = lowest canonical point index
    """

    xyz = np.asarray(xyz, dtype=np.float32)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(
            f"xyz must have shape (N, 3), got {xyz.shape}"
        )

    if not np.all(np.isfinite(xyz)):
        raise ValueError(
            "SPVCNN voxelization requires finite canonical XYZ"
        )

    if (
        not np.isfinite(voxel_size_m)
        or voxel_size_m <= 0.0
    ):
        raise ValueError(
            "voxel_size_m must be finite and > 0"
        )

    if len(xyz) == 0:
        return VoxelizationResult(
            voxel_coordinates=np.empty(
                (0, 3),
                dtype=np.int64,
            ),
            point_to_voxel_inverse=np.empty(
                (0,),
                dtype=np.int64,
            ),
            voxel_to_representative_point=np.empty(
                (0,),
                dtype=np.int64,
            ),
        )

    # Frozen signed floor rule. No coordinate offset/origin shift.
    point_voxel_coordinates = np.floor(
        xyz.astype(np.float64) / float(voxel_size_m)
    ).astype(np.int64)

    # np.unique gives deterministic lexicographic voxel ordering.
    voxel_coordinates, point_to_voxel_inverse = np.unique(
        point_voxel_coordinates,
        axis=0,
        return_inverse=True,
    )

    point_to_voxel_inverse = point_to_voxel_inverse.astype(
        np.int64,
        copy=False,
    )

    # Euclidean range in the same sensor-frame XYZ.
    ranges_squared = np.sum(
        xyz.astype(np.float64) ** 2,
        axis=1,
    )

    num_voxels = len(voxel_coordinates)

    representatives = np.empty(
        num_voxels,
        dtype=np.int64,
    )

    for voxel_id in range(num_voxels):
        point_indices = np.flatnonzero(
            point_to_voxel_inverse == voxel_id
        )

        local_ranges = ranges_squared[point_indices]

        minimum_range = np.min(local_ranges)

        tied_indices = point_indices[
            local_ranges == minimum_range
        ]

        # point_indices are canonical point indices in ascending order.
        representatives[voxel_id] = int(
            np.min(tied_indices)
        )

    return VoxelizationResult(
        voxel_coordinates=voxel_coordinates.astype(
            np.int64,
            copy=False,
        ),
        point_to_voxel_inverse=point_to_voxel_inverse,
        voxel_to_representative_point=representatives,
    )