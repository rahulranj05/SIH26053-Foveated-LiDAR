from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from perception_v1.src.data.normalization import normalize_intensity
from perception_v1.src.geometry.features import (
    compute_spherical_features,
    compute_validity_mask,
)
from perception_v1.src.ontology.mappings import NATIVE_TO_UNIFIED


@dataclass(frozen=True)
class CanonicalPointFrame:
    dataset_id: str
    xyz: np.ndarray
    intensity_raw: np.ndarray
    intensity_normalized: np.ndarray
    range: np.ndarray
    azimuth: np.ndarray
    elevation: np.ndarray
    validity_mask: np.ndarray
    source_point_id: np.ndarray
    unified_semantic_target: np.ndarray | None


def map_semantic_targets(
    dataset_id: str,
    native_labels: np.ndarray,
) -> np.ndarray:
    """
    Convert native dataset labels to frozen unified labels 0..24.

    Unknown native IDs are rejected rather than silently mapped.
    """
    labels = np.asarray(native_labels).reshape(-1)

    if dataset_id not in NATIVE_TO_UNIFIED:
        raise KeyError(f"Unsupported dataset: {dataset_id}")

    mapping = NATIVE_TO_UNIFIED[dataset_id]

    unique_labels = np.unique(labels)

    unknown = [
        int(label)
        for label in unique_labels
        if int(label) not in mapping
    ]

    if unknown:
        raise ValueError(
            f"{dataset_id} contains native labels absent from frozen mapping: "
            f"{unknown}"
        )

    target = np.empty(len(labels), dtype=np.int64)

    for native_id in unique_labels:
        target[labels == native_id] = mapping[int(native_id)]

    return target


def build_canonical_point_frame(
    dataset_id: str,
    xyz: np.ndarray,
    intensity_raw: np.ndarray,
    native_labels: np.ndarray | None = None,
) -> CanonicalPointFrame:

    xyz = np.asarray(xyz, dtype=np.float32)
    intensity_raw = np.asarray(
        intensity_raw,
        dtype=np.float32,
    ).reshape(-1)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(
            f"xyz must have shape (N, 3), got {xyz.shape}"
        )

    if len(intensity_raw) != len(xyz):
        raise ValueError(
            f"Intensity length {len(intensity_raw)} "
            f"!= point count {len(xyz)}"
        )

    validity_mask = compute_validity_mask(xyz)

    range_m, azimuth, elevation = compute_spherical_features(xyz)

    intensity_normalized = normalize_intensity(
        dataset_id,
        intensity_raw,
    )

    source_point_id = np.arange(
        len(xyz),
        dtype=np.int64,
    )

    unified_target = None

    if native_labels is not None:
        native_labels = np.asarray(native_labels).reshape(-1)

        if len(native_labels) != len(xyz):
            raise ValueError(
                f"Label length {len(native_labels)} "
                f"!= point count {len(xyz)}"
            )

        unified_target = map_semantic_targets(
            dataset_id,
            native_labels,
        )

    return CanonicalPointFrame(
        dataset_id=dataset_id,
        xyz=xyz,
        intensity_raw=intensity_raw,
        intensity_normalized=intensity_normalized,
        range=range_m,
        azimuth=azimuth,
        elevation=elevation,
        validity_mask=validity_mask,
        source_point_id=source_point_id,
        unified_semantic_target=unified_target,
    )
