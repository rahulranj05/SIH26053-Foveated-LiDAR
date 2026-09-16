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
    raw_source_index: np.ndarray
    unified_semantic_target: np.ndarray | None


def map_semantic_targets(
    dataset_id: str,
    native_labels: np.ndarray,
) -> np.ndarray:

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
            f"{dataset_id} contains native labels absent "
            f"from frozen mapping: {unknown}"
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

    raw_count = len(xyz)

    if len(intensity_raw) != raw_count:
        raise ValueError(
            f"Intensity length {len(intensity_raw)} "
            f"!= point count {raw_count}"
        )

    labels = None

    if native_labels is not None:
        labels = np.asarray(native_labels).reshape(-1)

        if len(labels) != raw_count:
            raise ValueError(
                f"Label length {len(labels)} "
                f"!= point count {raw_count}"
            )

    # -------------------------------------------------
    # S7-B / S7-H:
    # validity is determined on the raw point records.
    # -------------------------------------------------
    raw_validity_mask = compute_validity_mask(xyz)

    # Preserve provenance back to the original raw record.
    raw_source_index = np.flatnonzero(
        raw_validity_mask
    ).astype(np.int64)

    # -------------------------------------------------
    # Canonical representation contains VALID points only.
    # -------------------------------------------------
    xyz = xyz[raw_validity_mask]
    intensity_raw = intensity_raw[raw_validity_mask]

    if labels is not None:
        labels = labels[raw_validity_mask]

    # Canonical IDs are assigned AFTER validity filtering.
    source_point_id = np.arange(
        len(xyz),
        dtype=np.int64,
    )

    # All points in the canonical frame are valid by definition.
    validity_mask = np.ones(
        len(xyz),
        dtype=bool,
    )

    # Frozen normalization is applied to retained points.
    intensity_normalized = normalize_intensity(
        dataset_id,
        intensity_raw,
    )

    range_m, azimuth, elevation = (
        compute_spherical_features(xyz)
    )

    unified_target = None

    if labels is not None:
        unified_target = map_semantic_targets(
            dataset_id,
            labels,
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
        raw_source_index=raw_source_index,
        unified_semantic_target=unified_target,
    )
