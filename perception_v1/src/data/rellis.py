from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from perception_v1.src.data.canonical import (
    CanonicalPointFrame,
    build_canonical_point_frame,
)


@dataclass(frozen=True)
class RELLISFrameInfo:
    sequence: str
    frame_id: str
    scan_path: Path
    label_path: Path


def load_rellis_scan(
    scan_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one RELLIS-3D OS1 KITTI-format scan.

    Stored format:
        x, y, z, intensity
    as float32.

    Zero-XYZ records are preserved here.
    Canonical validity logic marks them invalid later.
    """
    path = Path(scan_path)

    raw = np.fromfile(path, dtype=np.float32)

    if raw.size % 4 != 0:
        raise ValueError(
            f"Invalid RELLIS scan {path}: "
            f"{raw.size} float32 values is not divisible by 4"
        )

    points = raw.reshape(-1, 4)

    xyz = points[:, :3].copy()
    intensity = points[:, 3].copy()

    return xyz, intensity


def load_rellis_native_labels(
    label_path: str | Path,
) -> np.ndarray:
    """
    RELLIS SemanticKITTI-style label-id files are stored
    as uint32 native semantic IDs.
    """
    path = Path(label_path)

    return np.fromfile(
        path,
        dtype=np.uint32,
    ).astype(np.int32)


def load_rellis_frame(
    scan_path: str | Path,
    label_path: str | Path,
) -> CanonicalPointFrame:

    xyz, intensity = load_rellis_scan(scan_path)
    native_labels = load_rellis_native_labels(label_path)

    if len(xyz) != len(native_labels):
        raise ValueError(
            f"Point/label mismatch: "
            f"{len(xyz)} points vs "
            f"{len(native_labels)} labels"
        )

    return build_canonical_point_frame(
        dataset_id="RELLIS-3D",
        xyz=xyz,
        intensity_raw=intensity,
        native_labels=native_labels,
    )


def frame_paths(
    dataset_root: str | Path,
    sequence: str,
    frame_id: str,
) -> RELLISFrameInfo:
    """
    dataset_root:
        .../RELLIS_3D/extracted/Rellis-3D
    """
    root = Path(dataset_root)

    sequence_root = root / sequence

    scan_path = (
        sequence_root
        / "os1_cloud_node_kitti_bin"
        / f"{frame_id}.bin"
    )

    label_path = (
        sequence_root
        / "os1_cloud_node_semantickitti_label_id"
        / f"{frame_id}.label"
    )

    return RELLISFrameInfo(
        sequence=sequence,
        frame_id=frame_id,
        scan_path=scan_path,
        label_path=label_path,
    )
