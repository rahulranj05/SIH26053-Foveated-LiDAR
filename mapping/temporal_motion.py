"""
A25 — Temporal Motion Estimation

Estimates per-point temporal motion evidence between consecutive LiDAR
SensorFrames.

A25 is simulator- and dataset-independent. It does not perform ego-motion
compensation; that is intentionally separated into A26.

The module compares spatial point occupancy between consecutive frames and
produces a deterministic per-point dynamic probability in [0, 1].

A25 does not modify A1–A24.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mapping.sensor_frame import SensorFrame


@dataclass(frozen=True)
class TemporalMotionConfig:
    """
    Configuration for temporal motion estimation.

    Parameters
    ----------
    voxel_size:
        Spatial voxel size in metres used to establish coarse correspondence.

    motion_threshold:
        Distance in metres above which unmatched displacement contributes
        strongly to motion evidence.

    max_correspondence_distance:
        Maximum distance between corresponding occupied voxels.

    min_probability:
        Minimum probability assigned to points with temporal evidence.

    """

    voxel_size: float = 0.20
    motion_threshold: float = 0.50
    max_correspondence_distance: float = 1.50
    min_probability: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.voxel_size) or self.voxel_size <= 0.0:
            raise ValueError("voxel_size must be finite and > 0")

        if (
            not np.isfinite(self.motion_threshold)
            or self.motion_threshold < 0.0
        ):
            raise ValueError(
                "motion_threshold must be finite and >= 0"
            )

        if (
            not np.isfinite(self.max_correspondence_distance)
            or self.max_correspondence_distance <= 0.0
        ):
            raise ValueError(
                "max_correspondence_distance must be finite and > 0"
            )

        if (
            not np.isfinite(self.min_probability)
            or not 0.0 <= self.min_probability <= 1.0
        ):
            raise ValueError(
                "min_probability must be finite and in [0, 1]"
            )


def _validate_xyz(xyz: np.ndarray) -> np.ndarray:
    """Validate and normalize an XYZ point cloud."""
    array = np.asarray(xyz, dtype=np.float64)

    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(
            f"xyz must have shape (N, 3), got {array.shape}"
        )

    if not np.all(np.isfinite(array)):
        raise ValueError("xyz must contain only finite values")

    return array


def _voxel_keys(
    xyz: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    """
    Convert XYZ coordinates into integer voxel coordinates.
    """
    return np.floor(xyz / voxel_size).astype(np.int64)


def _unique_voxel_centres(
    xyz: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return unique voxel keys and their representative point-centres.

    The representative centre is the mean of all points occupying each
    voxel. The returned arrays are deterministic.
    """
    keys = _voxel_keys(xyz, voxel_size)

    if keys.shape[0] == 0:
        return (
            np.empty((0, 3), dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
        )

    unique_keys, inverse = np.unique(
        keys,
        axis=0,
        return_inverse=True,
    )

    centres = np.zeros(
        (unique_keys.shape[0], 3),
        dtype=np.float64,
    )

    counts = np.bincount(inverse)

    for axis in range(3):
        centres[:, axis] = np.bincount(
            inverse,
            weights=xyz[:, axis],
        )

    centres /= counts[:, None]

    return unique_keys, centres


def _nearest_distances(
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """
    Calculate nearest-neighbour distances from source to target.

    Uses a vectorized squared-distance matrix. A25 is intended for
    deterministic moderate-size temporal validation; A26/A37 can replace
    this backend with spatial indexing/optimized correspondence.
    """
    if source.shape[0] == 0:
        return np.empty(0, dtype=np.float64)

    if target.shape[0] == 0:
        return np.full(
            source.shape[0],
            np.inf,
            dtype=np.float64,
        )

    distances_squared = np.sum(
        (
            source[:, None, :]
            - target[None, :, :]
        ) ** 2,
        axis=2,
    )

    return np.sqrt(
        np.min(distances_squared, axis=1)
    )


def temporal_motion_evidence(
    previous_xyz: np.ndarray,
    current_xyz: np.ndarray,
    config: TemporalMotionConfig | None = None,
) -> np.ndarray:
    """
    Estimate motion evidence for every point in the current frame.

    Evidence is based on the nearest occupied spatial voxel representation
    from the previous frame.

    Values are in [0, 1]:

    - 0.0 means strong temporal consistency.
    - 1.0 means the current point has no sufficiently close previous
      correspondence.

    Parameters
    ----------
    previous_xyz:
        Previous-frame XYZ points, shape (N, 3).

    current_xyz:
        Current-frame XYZ points, shape (M, 3).

    config:
        Temporal motion configuration.

    Returns
    -------
    np.ndarray
        Per-current-point motion evidence with shape (M,).
    """
    if config is None:
        config = TemporalMotionConfig()

    previous = _validate_xyz(previous_xyz)
    current = _validate_xyz(current_xyz)

    if current.shape[0] == 0:
        return np.empty(0, dtype=np.float64)

    if previous.shape[0] == 0:
        return np.ones(
            current.shape[0],
            dtype=np.float64,
        )

    _, previous_centres = _unique_voxel_centres(
        previous,
        config.voxel_size,
    )

    distances = _nearest_distances(
        current,
        previous_centres,
    )

    threshold = config.motion_threshold
    maximum = config.max_correspondence_distance

    if maximum <= threshold:
        normalized = distances > threshold
        evidence = normalized.astype(np.float64)
    else:
        evidence = np.clip(
            (distances - threshold)
            / (maximum - threshold),
            0.0,
            1.0,
        )

    evidence = np.maximum(
        evidence,
        config.min_probability,
    )

    return np.clip(
        evidence,
        0.0,
        1.0,
    )


def temporal_dynamic_probability(
    previous_frame: SensorFrame,
    current_frame: SensorFrame,
    config: TemporalMotionConfig | None = None,
) -> np.ndarray:
    """
    Estimate per-point dynamic probability for the current frame.

    The timestamp interval is used as a consistency check. A non-positive
    interval is rejected because temporal motion requires chronological
    frames.

    The current frame's XYZ coordinates receive the resulting probabilities.
    """
    if not isinstance(previous_frame, SensorFrame):
        raise TypeError(
            "previous_frame must be a SensorFrame"
        )

    if not isinstance(current_frame, SensorFrame):
        raise TypeError(
            "current_frame must be a SensorFrame"
        )

    delta_time = (
        current_frame.timestamp
        - previous_frame.timestamp
    )

    if delta_time <= 0.0:
        raise ValueError(
            "current_frame timestamp must be greater than "
            "previous_frame timestamp"
        )

    return temporal_motion_evidence(
        previous_frame.xyz,
        current_frame.xyz,
        config=config,
    )


def temporal_motion_signal(
    previous_frame: SensorFrame,
    current_frame: SensorFrame,
    config: TemporalMotionConfig | None = None,
) -> dict[str, np.ndarray]:
    """
    Return an A20-compatible dynamic signal dictionary.

    The returned dictionary uses the canonical ``dynamic`` signal namespace.
    """
    return {
        "dynamic": temporal_dynamic_probability(
            previous_frame,
            current_frame,
            config=config,
        )
    }