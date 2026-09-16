from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from perception_v1.src.data.canonical import (
    CanonicalPointFrame,
    build_canonical_point_frame,
)


@dataclass(frozen=True)
class NuScenesRawScan:
    xyz: np.ndarray
    intensity: np.ndarray
    ring: np.ndarray


def load_nuscenes_scan(
    scan_path: str | Path,
) -> NuScenesRawScan:
    """
    nuScenes LIDAR_TOP point format:
        [x, y, z, intensity, ring]
        5 x float32 per point.

    Unlike SemanticSTF, nuScenes ring provenance is proven
    and may be preserved as native sensor metadata.
    """
    path = Path(scan_path)

    raw = np.fromfile(path, dtype=np.float32)

    if raw.size % 5 != 0:
        raise ValueError(
            f"Invalid nuScenes scan {path}: "
            f"{raw.size} float32 values is not divisible by 5"
        )

    points = raw.reshape(-1, 5)

    ring_float = points[:, 4]

    if not np.all(np.isfinite(ring_float)):
        raise ValueError("nuScenes ring contains non-finite values")

    if not np.all(ring_float == np.floor(ring_float)):
        raise ValueError("nuScenes ring contains non-integer values")

    ring = ring_float.astype(np.int32)

    if np.any((ring < 0) | (ring > 31)):
        raise ValueError(
            "nuScenes native ring must be within [0, 31]"
        )

    return NuScenesRawScan(
        xyz=points[:, :3].copy(),
        intensity=points[:, 3].copy(),
        ring=ring,
    )


def load_nuscenes_native_labels(
    label_path: str | Path,
) -> np.ndarray:
    """
    nuScenes lidarseg labels are stored as uint8 native
    semantic category IDs.
    """
    return np.fromfile(
        Path(label_path),
        dtype=np.uint8,
    ).astype(np.int32)


def load_nuscenes_frame(
    scan_path: str | Path,
    label_path: str | Path,
) -> CanonicalPointFrame:

    raw_scan = load_nuscenes_scan(scan_path)
    labels = load_nuscenes_native_labels(label_path)

    if len(raw_scan.xyz) != len(labels):
        raise ValueError(
            f"Point/label mismatch: "
            f"{len(raw_scan.xyz)} points vs {len(labels)} labels"
        )

    return build_canonical_point_frame(
        dataset_id="nuScenes",
        xyz=raw_scan.xyz,
        intensity_raw=raw_scan.intensity,
        native_labels=labels,
    )
