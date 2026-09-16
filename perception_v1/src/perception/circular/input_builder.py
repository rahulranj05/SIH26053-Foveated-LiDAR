from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from perception_v1.src.perception.circular.projection import (
    RangeProjection,
    project_range_view,
)


@dataclass(frozen=True)
class CircularInput:
    image_features: np.ndarray
    image_targets: np.ndarray | None
    pixel_valid_mask: np.ndarray
    point_to_pixel: np.ndarray
    pixel_to_source_point: np.ndarray
    in_fov_mask: np.ndarray
    winner_mask: np.ndarray
    source_point_id: np.ndarray


def build_circular_input(
    xyz: np.ndarray,
    point_features: np.ndarray,
    *,
    dataset_id: str,
    source_point_id: np.ndarray | None = None,
    point_targets: np.ndarray | None = None,
) -> CircularInput:
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
            "XYZ/feature length mismatch"
        )

    if not np.all(np.isfinite(point_features)):
        raise ValueError(
            "point_features must be finite"
        )

    n = len(xyz)

    if source_point_id is None:
        source_ids = np.arange(
            n,
            dtype=np.int64,
        )
    else:
        source_ids = np.asarray(
            source_point_id,
            dtype=np.int64,
        )

        if source_ids.ndim != 1:
            raise ValueError(
                "source_point_id must have shape (N,)"
            )

        if len(source_ids) != n:
            raise ValueError(
                "XYZ/source_point_id length mismatch"
            )

        if len(np.unique(source_ids)) != n:
            raise ValueError(
                "source_point_id must be unique"
            )

    targets = None

    if point_targets is not None:
        targets = np.asarray(point_targets)

        if targets.ndim != 1:
            raise ValueError(
                "point_targets must have shape (N,)"
            )

        if len(targets) != n:
            raise ValueError(
                "XYZ/target length mismatch"
            )

    projection: RangeProjection = project_range_view(
        xyz,
        dataset_id=dataset_id,
        source_point_id=source_ids,
    )

    feature_dim = point_features.shape[1]

    image_features = np.zeros(
        (
            projection.height,
            projection.width,
            feature_dim,
        ),
        dtype=point_features.dtype,
    )

    pixel_valid_mask = np.zeros(
        (
            projection.height,
            projection.width,
        ),
        dtype=bool,
    )

    image_targets = None

    if targets is not None:
        image_targets = np.zeros(
            (
                projection.height,
                projection.width,
            ),
            dtype=targets.dtype,
        )

    source_id_to_index = {
        int(source_id): point_index
        for point_index, source_id
        in enumerate(source_ids)
    }

    rows, cols = np.nonzero(
        projection.pixel_to_source_point >= 0
    )

    for row, col in zip(rows, cols):
        source_id = int(
            projection.pixel_to_source_point[
                row, col
            ]
        )

        point_index = source_id_to_index[
            source_id
        ]

        image_features[
            row, col
        ] = point_features[point_index]

        pixel_valid_mask[
            row, col
        ] = True

        if image_targets is not None:
            image_targets[
                row, col
            ] = targets[point_index]

    return CircularInput(
        image_features=image_features,
        image_targets=image_targets,
        pixel_valid_mask=pixel_valid_mask,
        point_to_pixel=projection.point_to_pixel,
        pixel_to_source_point=projection.pixel_to_source_point,
        in_fov_mask=projection.in_fov_mask,
        winner_mask=projection.winner_mask,
        source_point_id=source_ids.copy(),
    )


def lift_pixel_features_to_points(
    pixel_features: np.ndarray,
    circular_input: CircularInput,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Lift circular-branch features to canonical point domain.

    Frozen S7-H rule:
    - only pixel-owning z-buffer winner receives its pixel feature
    - z-buffer losers receive zero circular feature
    - out-of-FOV points receive zero circular feature
    - rv_feature_valid_mask marks only valid winners
    """

    pixel_features = np.asarray(
        pixel_features
    )

    if pixel_features.ndim != 3:
        raise ValueError(
            "pixel_features must have shape (H, W, C)"
        )

    expected_hw = (
        circular_input.pixel_valid_mask.shape
    )

    if pixel_features.shape[:2] != expected_hw:
        raise ValueError(
            "Pixel feature geometry mismatch"
        )

    n = len(circular_input.source_point_id)
    channels = pixel_features.shape[2]

    point_features = np.zeros(
        (n, channels),
        dtype=pixel_features.dtype,
    )

    rv_feature_valid_mask = np.zeros(
        n,
        dtype=bool,
    )

    winner_indices = np.flatnonzero(
        circular_input.winner_mask
    )

    for point_index in winner_indices:
        row, col = circular_input.point_to_pixel[
            point_index
        ]

        if row < 0 or col < 0:
            raise RuntimeError(
                "Winner has no valid pixel mapping"
            )

        point_features[
            point_index
        ] = pixel_features[row, col]

        rv_feature_valid_mask[
            point_index
        ] = True

    return (
        point_features,
        rv_feature_valid_mask,
    )