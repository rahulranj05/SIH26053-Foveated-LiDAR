from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from perception_v1.src.data.canonical import (
    CanonicalPointFrame,
    build_canonical_point_frame,
)


@dataclass(frozen=True)
class SemanticSTFFrameInfo:
    split: str
    frame_id: str
    scan_path: Path
    label_path: Path


@dataclass(frozen=True)
class SemanticSTFRawScan:
    xyz: np.ndarray
    intensity: np.ndarray
    unknown_sensor_channel: np.ndarray


def load_semanticstf_scan(
    scan_path: str | Path,
) -> SemanticSTFRawScan:
    """
    SemanticSTF raw scan format:
        [x, y, z, intensity, unknown_sensor_channel]
        5 x float32 per point.

    Frozen S7 policy:
    channel 5 is preserved for provenance but MUST NOT be
    interpreted or exposed as native LiDAR ring.
    """
    path = Path(scan_path)
    raw = np.fromfile(path, dtype=np.float32)

    if raw.size % 5 != 0:
        raise ValueError(
            f"Invalid SemanticSTF scan {path}: "
            f"{raw.size} float32 values is not divisible by 5"
        )

    points = raw.reshape(-1, 5)

    return SemanticSTFRawScan(
        xyz=points[:, :3].copy(),
        intensity=points[:, 3].copy(),
        unknown_sensor_channel=points[:, 4].copy(),
    )


def load_semanticstf_native_labels(
    label_path: str | Path,
) -> np.ndarray:
    return np.fromfile(
        Path(label_path),
        dtype=np.uint32,
    ).astype(np.int32)


def load_semanticstf_frame(
    scan_path: str | Path,
    label_path: str | Path,
) -> CanonicalPointFrame:

    raw_scan = load_semanticstf_scan(scan_path)
    labels = load_semanticstf_native_labels(label_path)

    if len(raw_scan.xyz) != len(labels):
        raise ValueError(
            f"Point/label mismatch: "
            f"{len(raw_scan.xyz)} points vs {len(labels)} labels"
        )

    return build_canonical_point_frame(
        dataset_id="SemanticSTF",
        xyz=raw_scan.xyz,
        intensity_raw=raw_scan.intensity,
        native_labels=labels,
    )


def frame_paths(
    dataset_root: str | Path,
    split: str,
    frame_id: str,
) -> SemanticSTFFrameInfo:

    if split not in {"train", "val", "test"}:
        raise ValueError(
            f"Invalid SemanticSTF split: {split}"
        )

    root = Path(dataset_root) / split

    return SemanticSTFFrameInfo(
        split=split,
        frame_id=frame_id,
        scan_path=root / "velodyne" / f"{frame_id}.bin",
        label_path=root / "labels" / f"{frame_id}.label",
    )
