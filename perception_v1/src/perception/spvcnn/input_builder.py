from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from perception_v1.src.perception.spvcnn.voxelization import (
    DEFAULT_VOXEL_SIZE_M,
    VoxelizationResult,
    voxelize_points,
)


@dataclass(frozen=True)
class SPVCNNInput:
    voxel_coordinates: np.ndarray
    voxel_features: np.ndarray
    voxel_targets: np.ndarray | None
    point_to_voxel_inverse: np.ndarray
    voxel_to_representative_point: np.ndarray


def build_spvcnn_input(
    xyz: np.ndarray,
    point_features: np.ndarray,
    *,
    point_targets: np.ndarray | None = None,
    voxel_size_m: float = DEFAULT_VOXEL_SIZE_M,
) -> SPVCNNInput:
    """
    Build sparse SPVCNN input from canonical valid points.

    Frozen S7-E rules:
    - 0.10 m default voxelization
    - representative = nearest-range canonical point
    - exact tie = lowest canonical point index
    - voxel features come from representative point
    - voxel target comes from representative point
    - NO semantic majority vote
    - preserve point<->voxel mappings
    """

    xyz = np.asarray(xyz, dtype=np.float32)
    point_features = np.asarray(point_features)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(
            f"xyz must have shape (N, 3), got {xyz.shape}"
        )

    if point_features.ndim != 2:
        raise ValueError(
            "point_features must have shape (N, F)"
        )

    if len(point_features) != len(xyz):
        raise ValueError(
            "xyz and point_features length mismatch"
        )

    if not np.all(np.isfinite(point_features)):
        raise ValueError(
            "point_features must be finite"
        )

    targets = None

    if point_targets is not None:
        targets = np.asarray(point_targets)

        if targets.ndim != 1:
            raise ValueError(
                "point_targets must have shape (N,)"
            )

        if len(targets) != len(xyz):
            raise ValueError(
                "xyz and point_targets length mismatch"
            )

    vox: VoxelizationResult = voxelize_points(
        xyz,
        voxel_size_m=voxel_size_m,
    )

    representatives = vox.voxel_to_representative_point

    voxel_features = point_features[
        representatives
    ].copy()

    voxel_targets = None

    if targets is not None:
        # Deliberately representative-point semantics.
        # No majority voting.
        voxel_targets = targets[
            representatives
        ].copy()

    return SPVCNNInput(
        voxel_coordinates=vox.voxel_coordinates,
        voxel_features=voxel_features,
        voxel_targets=voxel_targets,
        point_to_voxel_inverse=vox.point_to_voxel_inverse,
        voxel_to_representative_point=representatives,
    )


def backproject_voxel_values_to_points(
    voxel_values: np.ndarray,
    point_to_voxel_inverse: np.ndarray,
) -> np.ndarray:
    """
    Lift voxel-domain outputs back to every canonical point.
    """

    voxel_values = np.asarray(voxel_values)
    inverse = np.asarray(
        point_to_voxel_inverse,
        dtype=np.int64,
    )

    if inverse.ndim != 1:
        raise ValueError(
            "point_to_voxel_inverse must have shape (N,)"
        )

    if voxel_values.ndim < 1:
        raise ValueError(
            "voxel_values must have at least one dimension"
        )

    if len(inverse) == 0:
        return voxel_values[:0].copy()

    if np.any(inverse < 0):
        raise ValueError(
            "point_to_voxel_inverse contains negative indices"
        )

    if np.any(inverse >= len(voxel_values)):
        raise ValueError(
            "point_to_voxel_inverse references missing voxel"
        )

    return voxel_values[inverse].copy()