from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class VoxelizedSample:
    voxel_coords: np.ndarray
    voxel_features: np.ndarray
    voxel_labels: np.ndarray
    inverse_map: np.ndarray
    point_to_voxel_counts: np.ndarray


class VoxelAdapter:
    """
    CPU-side deterministic voxelization adapter.

    Responsibilities:
    - convert XYZ to discrete voxel coordinates
    - group points by voxel
    - aggregate point features by mean
    - assign one semantic label per voxel by majority vote
    - preserve point -> voxel inverse mapping

    This adapter is backend-agnostic. TorchSparse-specific tensor
    conversion should happen later in the SPVCNN adapter.
    """

    def __init__(
        self,
        voxel_size: float = 0.05,
        ignore_index: int = 0,
    ) -> None:
        if voxel_size <= 0:
            raise ValueError(
                "voxel_size must be > 0"
            )

        self.voxel_size = float(voxel_size)
        self.ignore_index = int(ignore_index)

    def _validate(
        self,
        xyz: np.ndarray,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        xyz = np.asarray(
            xyz,
            dtype=np.float32,
        )

        features = np.asarray(
            features,
            dtype=np.float32,
        )

        labels = np.asarray(
            labels,
            dtype=np.int64,
        )

        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(
                f"xyz must have shape [N,3], got {xyz.shape}"
            )

        if features.ndim != 2:
            raise ValueError(
                f"features must have shape [N,C], got {features.shape}"
            )

        if labels.ndim != 1:
            raise ValueError(
                f"labels must have shape [N], got {labels.shape}"
            )

        n = xyz.shape[0]

        if features.shape[0] != n:
            raise ValueError(
                "xyz/features point count mismatch"
            )

        if labels.shape[0] != n:
            raise ValueError(
                "xyz/labels point count mismatch"
            )

        if n == 0:
            raise ValueError(
                "Cannot voxelize empty point cloud"
            )

        if not np.isfinite(xyz).all():
            raise ValueError(
                "xyz contains NaN/Inf"
            )

        if not np.isfinite(features).all():
            raise ValueError(
                "features contain NaN/Inf"
            )

        return xyz, features, labels

    def _majority_label(
        self,
        labels: np.ndarray,
    ) -> int:

        valid = labels[
            labels != self.ignore_index
        ]

        if valid.size == 0:
            return self.ignore_index

        unique, counts = np.unique(
            valid,
            return_counts=True,
        )

        max_count = counts.max()

        winners = unique[
            counts == max_count
        ]

        # Deterministic tie-break:
        # choose lowest class ID.
        return int(winners.min())

    def __call__(
        self,
        xyz: np.ndarray,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> VoxelizedSample:

        xyz, features, labels = self._validate(
            xyz,
            features,
            labels,
        )

        # ----------------------------------------------------
        # Convert metric XYZ to integer voxel coordinates.
        # floor() is important for negative coordinates.
        # ----------------------------------------------------

        coords = np.floor(
            xyz / self.voxel_size
        ).astype(np.int32)

        # ----------------------------------------------------
        # Find unique voxels and inverse point -> voxel map.
        # ----------------------------------------------------

        unique_coords, inverse = np.unique(
            coords,
            axis=0,
            return_inverse=True,
        )

        num_voxels = unique_coords.shape[0]
        num_features = features.shape[1]

        # ----------------------------------------------------
        # Aggregate point features by arithmetic mean.
        # ----------------------------------------------------

        voxel_features = np.zeros(
            (num_voxels, num_features),
            dtype=np.float64,
        )

        counts = np.bincount(
            inverse,
            minlength=num_voxels,
        ).astype(np.int64)

        for channel in range(num_features):
            voxel_features[:, channel] = (
                np.bincount(
                    inverse,
                    weights=features[:, channel],
                    minlength=num_voxels,
                )
                / counts
            )

        voxel_features = voxel_features.astype(
            np.float32,
        )

        # ----------------------------------------------------
        # Majority semantic label per voxel.
        # ----------------------------------------------------

        voxel_labels = np.full(
            num_voxels,
            self.ignore_index,
            dtype=np.int64,
        )

        # Sort points by voxel once so we avoid repeatedly
        # scanning the entire point cloud.
        order = np.argsort(
            inverse,
            kind="stable",
        )

        sorted_inverse = inverse[order]
        sorted_labels = labels[order]

        boundaries = np.flatnonzero(
            np.diff(sorted_inverse)
        ) + 1

        voxel_groups = np.split(
            sorted_labels,
            boundaries,
        )

        for voxel_id, group_labels in enumerate(
            voxel_groups
        ):
            voxel_labels[voxel_id] = (
                self._majority_label(
                    group_labels
                )
            )

        return VoxelizedSample(
            voxel_coords=unique_coords,
            voxel_features=voxel_features,
            voxel_labels=voxel_labels,
            inverse_map=inverse.astype(
                np.int64,
                copy=False,
            ),
            point_to_voxel_counts=counts,
        )


def project_voxel_predictions_to_points(
    voxel_predictions: np.ndarray,
    inverse_map: np.ndarray,
) -> np.ndarray:
    """
    Expand voxel-level predictions back to original points.
    """

    voxel_predictions = np.asarray(
        voxel_predictions
    )

    inverse_map = np.asarray(
        inverse_map,
        dtype=np.int64,
    )

    if voxel_predictions.ndim != 1:
        raise ValueError(
            "voxel_predictions must be 1-D"
        )

    if inverse_map.ndim != 1:
        raise ValueError(
            "inverse_map must be 1-D"
        )

    if inverse_map.size == 0:
        return np.empty(
            0,
            dtype=voxel_predictions.dtype,
        )

    if inverse_map.min() < 0:
        raise ValueError(
            "inverse_map contains negative indices"
        )

    if inverse_map.max() >= voxel_predictions.shape[0]:
        raise ValueError(
            "inverse_map references missing voxel prediction"
        )

    return voxel_predictions[
        inverse_map
    ]