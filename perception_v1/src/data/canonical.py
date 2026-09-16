from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from perception_v1.src.data.normalization import normalize_intensity
from perception_v1.src.geometry.features import (
    compute_spherical_features,
    compute_validity_mask,
)


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


def build_canonical_point_frame(
    dataset_id: str,
    xyz: np.ndarray,
    intensity_raw: np.ndarray,
) -> CanonicalPointFrame:

    xyz = np.asarray(xyz, dtype=np.float32)
    intensity_raw = np.asarray(intensity_raw, dtype=np.float32).reshape(-1)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape (N, 3), got {xyz.shape}")

    if len(intensity_raw) != len(xyz):
        raise ValueError(
            f"Intensity length {len(intensity_raw)} != point count {len(xyz)}"
        )

    validity_mask = compute_validity_mask(xyz)

    range_m, azimuth, elevation = compute_spherical_features(xyz)

    intensity_normalized = normalize_intensity(
        dataset_id,
        intensity_raw,
    )

    source_point_id = np.arange(len(xyz), dtype=np.int64)

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
    )
