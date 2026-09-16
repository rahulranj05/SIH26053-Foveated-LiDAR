from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from perception_v1.src.data.canonical import (
    CanonicalPointFrame,
    build_canonical_point_frame,
)


@dataclass(frozen=True)
class SemanticKITTIFrameInfo:
    sequence: str
    frame_id: str
    scan_path: Path
    label_path: Path | None


def decode_semantickitti_labels(
    raw_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    SemanticKITTI uint32 label format:

    lower 16 bits -> semantic ID
    upper 16 bits -> instance ID
    """
    raw = np.asarray(raw_labels, dtype=np.uint32).reshape(-1)

    semantic = (raw & np.uint32(0xFFFF)).astype(np.int32)
    instance = (raw >> np.uint32(16)).astype(np.int32)

    return semantic, instance


def load_semantickitti_scan(
    scan_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load SemanticKITTI .bin scan.

    Each point contains:
        x, y, z, intensity
    as four float32 values.
    """
    path = Path(scan_path)

    raw = np.fromfile(path, dtype=np.float32)

    if raw.size % 4 != 0:
        raise ValueError(
            f"Invalid SemanticKITTI scan {path}: "
            f"{raw.size} float32 values is not divisible by 4"
        )

    points = raw.reshape(-1, 4)

    xyz = points[:, :3].copy()
    intensity = points[:, 3].copy()

    return xyz, intensity


def load_semantickitti_native_labels(
    label_path: str | Path,
) -> np.ndarray:
    path = Path(label_path)

    raw = np.fromfile(path, dtype=np.uint32)

    semantic, _ = decode_semantickitti_labels(raw)

    return semantic


def load_semantickitti_frame(
    scan_path: str | Path,
    label_path: str | Path | None = None,
) -> CanonicalPointFrame:
    """
    Load one SemanticKITTI frame and convert it to the
    frozen canonical point representation.
    """
    xyz, intensity = load_semantickitti_scan(scan_path)

    native_labels = None

    if label_path is not None:
        native_labels = load_semantickitti_native_labels(label_path)

        if len(native_labels) != len(xyz):
            raise ValueError(
                f"Point/label mismatch: "
                f"{len(xyz)} points vs {len(native_labels)} labels"
            )

    return build_canonical_point_frame(
        dataset_id="SemanticKITTI",
        xyz=xyz,
        intensity_raw=intensity,
        native_labels=native_labels,
    )


def frame_paths(
    dataset_root: str | Path,
    sequence: str,
    frame_id: str,
    supervised: bool = True,
) -> SemanticKITTIFrameInfo:
    """
    Resolve one frame without recursively scanning the dataset.

    dataset_root must point to:
        .../SemanticKITTI/extracted/dataset
    """
    root = Path(dataset_root)

    scan_path = (
        root
        / "sequences"
        / sequence
        / "velodyne"
        / f"{frame_id}.bin"
    )

    label_path = None

    if supervised:
        label_path = (
            root
            / "sequences"
            / sequence
            / "labels"
            / f"{frame_id}.label"
        )

    return SemanticKITTIFrameInfo(
        sequence=sequence,
        frame_id=frame_id,
        scan_path=scan_path,
        label_path=label_path,
    )
