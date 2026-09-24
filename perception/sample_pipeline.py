from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from perception.geometry.augmentation import (
    AugmentationConfig,
    GeometricTransform,
    augment_xyz,
)
from perception.geometry.features import (
    compute_spherical_features,
)
from perception.circular.input_builder import (
    CircularInput,
    build_circular_input,
)
from perception.spvcnn.input_builder import (
    SPVCNNInput,
    build_spvcnn_input,
)


@dataclass(frozen=True)
class UnifiedPerceptionSample:
    dataset_id: str
    split_role: str

    xyz: np.ndarray
    intensity: np.ndarray

    range: np.ndarray
    azimuth: np.ndarray
    elevation: np.ndarray

    source_point_id: np.ndarray
    targets: np.ndarray | None

    point_features: np.ndarray

    spvcnn: SPVCNNInput
    circular: CircularInput

    augmentation_transform: GeometricTransform | None


def _validate_inputs(
    xyz: np.ndarray,
    intensity: np.ndarray,
    source_point_id: np.ndarray,
    targets: np.ndarray | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
]:
    xyz = np.asarray(
        xyz,
        dtype=np.float32,
    )

    intensity = np.asarray(
        intensity,
        dtype=np.float32,
    )

    source_point_id = np.asarray(
        source_point_id,
        dtype=np.int64,
    )

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(
            f"xyz must have shape (N, 3), got {xyz.shape}"
        )

    n = len(xyz)

    if not np.all(np.isfinite(xyz)):
        raise ValueError(
            "Canonical XYZ must be finite"
        )

    if np.any(
        np.all(xyz == 0.0, axis=1)
    ):
        raise ValueError(
            "Canonical XYZ must exclude zero XYZ"
        )

    if (
        intensity.ndim != 1
        or len(intensity) != n
    ):
        raise ValueError(
            "intensity must have shape (N,)"
        )

    if not np.all(np.isfinite(intensity)):
        raise ValueError(
            "intensity must be finite"
        )

    if (
        source_point_id.ndim != 1
        or len(source_point_id) != n
    ):
        raise ValueError(
            "source_point_id must have shape (N,)"
        )

    if len(np.unique(source_point_id)) != n:
        raise ValueError(
            "source_point_id must be unique"
        )

    target_array = None

    if targets is not None:
        target_array = np.asarray(
            targets,
            dtype=np.int64,
        )

        if (
            target_array.ndim != 1
            or len(target_array) != n
        ):
            raise ValueError(
                "targets must have shape (N,)"
            )

        if np.any(
            (target_array < 0)
            | (target_array > 24)
        ):
            raise ValueError(
                "targets must use frozen IDs 0..24"
            )

    return (
        xyz,
        intensity,
        source_point_id,
        target_array,
    )


def _build_point_features(
    xyz: np.ndarray,
    intensity: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    point_range, azimuth, elevation = (
        compute_spherical_features(xyz)
    )

    point_features = np.column_stack(
        (
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            intensity,
            point_range,
            azimuth,
            elevation,
        )
    ).astype(
        np.float32,
        copy=False,
    )

    return (
        point_features,
        point_range,
        azimuth,
        elevation,
    )


def build_unified_sample(
    *,
    dataset_id: str,
    xyz: np.ndarray,
    intensity: np.ndarray,
    source_point_id: np.ndarray,
    split_role: str,
    targets: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    augmentation_config: AugmentationConfig | None = None,
) -> UnifiedPerceptionSample:

    allowed_splits = {
        "train",
        "val",
        "test",
        "inference",
    }

    if split_role not in allowed_splits:
        raise ValueError(
            f"Unknown split_role: {split_role}"
        )

    (
        xyz,
        intensity,
        source_point_id,
        targets,
    ) = _validate_inputs(
        xyz,
        intensity,
        source_point_id,
        targets,
    )

    # --------------------------------------------------------
    # ONE shared geometry source BEFORE branch split.
    # --------------------------------------------------------

    if split_role == "train":
        if rng is None:
            raise ValueError(
                "Training requires explicit RNG"
            )

        config = (
            AugmentationConfig()
            if augmentation_config is None
            else augmentation_config
        )

        working_xyz, transform = augment_xyz(
            xyz,
            training=True,
            rng=rng,
            config=config,
        )

    else:
        # Use the committed M3 API for evaluation as well.
        working_xyz, transform = augment_xyz(
            xyz,
            training=False,
        )

    # --------------------------------------------------------
    # Derived geometry MUST be recomputed AFTER augmentation.
    # --------------------------------------------------------

    (
        point_features,
        point_range,
        azimuth,
        elevation,
    ) = _build_point_features(
        working_xyz,
        intensity,
    )

    # --------------------------------------------------------
    # Branch split occurs ONLY here.
    # Both branches consume the same working_xyz/features.
    # --------------------------------------------------------

    spvcnn = build_spvcnn_input(
        working_xyz,
        point_features,
        point_targets=targets,
    )

    circular = build_circular_input(
        working_xyz,
        point_features,
        dataset_id=dataset_id,
        source_point_id=source_point_id,
        point_targets=targets,
    )

    return UnifiedPerceptionSample(
        dataset_id=dataset_id,
        split_role=split_role,

        xyz=working_xyz.copy(),
        intensity=intensity.copy(),

        range=point_range.copy(),
        azimuth=azimuth.copy(),
        elevation=elevation.copy(),

        source_point_id=source_point_id.copy(),

        targets=(
            None
            if targets is None
            else targets.copy()
        ),

        point_features=point_features.copy(),

        spvcnn=spvcnn,
        circular=circular,

        augmentation_transform=transform,
    )
