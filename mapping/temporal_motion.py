"""
A25/A37 — Temporal Motion Estimation

A25:
    Estimates per-point temporal motion evidence between consecutive
    LiDAR SensorFrames.

A37:
    Optimizes nearest-neighbour correspondence using a dependency-free
    uniform spatial hash.

The public A25 API and semantics are preserved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mapping.sensor_frame import SensorFrame


@dataclass(frozen=True)
class TemporalMotionConfig:
    """
    Configuration for temporal motion estimation.
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
    """Convert XYZ coordinates into integer voxel coordinates."""
    return np.floor(
        xyz / voxel_size
    ).astype(np.int64)


def _unique_voxel_centres(
    xyz: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return unique voxel keys and representative point centres.

    The representative centre is the mean of all points occupying
    each voxel.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(unique_keys, centres)``.
    """
    keys = _voxel_keys(
        xyz,
        voxel_size,
    )

    if keys.shape[0] == 0:
        return (
            np.empty(
                (0, 3),
                dtype=np.int64,
            ),
            np.empty(
                (0, 3),
                dtype=np.float64,
            ),
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

    counts = np.bincount(
        inverse
    ).astype(np.float64)

    for axis in range(3):
        centres[:, axis] = np.bincount(
            inverse,
            weights=xyz[:, axis],
        )

    centres /= counts[:, None]

    return unique_keys, centres


def _nearest_distances_bruteforce(
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """
    Exact O(N*M) reference nearest-neighbour implementation.

    Retained as a correctness reference for A37 and for compatibility
    with the original A25 implementation.
    """
    if source.shape[0] == 0:
        return np.empty(
            0,
            dtype=np.float64,
        )

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
        np.min(
            distances_squared,
            axis=1,
        )
    )


def _build_spatial_hash(
    target: np.ndarray,
    cell_size: float,
) -> dict[tuple[int, int, int], np.ndarray]:
    """
    Build a uniform spatial hash over target points.

    Each hash key identifies one cubic spatial cell and maps to the
    indices of target points inside that cell.
    """
    if target.shape[0] == 0:
        return {}

    if (
        not np.isfinite(cell_size)
        or cell_size <= 0.0
    ):
        raise ValueError(
            "cell_size must be finite and > 0"
        )

    keys = np.floor(
        target / cell_size
    ).astype(np.int64)

    order = np.lexsort(
        (
            keys[:, 2],
            keys[:, 1],
            keys[:, 0],
        )
    )

    sorted_keys = keys[order]

    boundaries = (
        np.flatnonzero(
            np.any(
                sorted_keys[1:]
                != sorted_keys[:-1],
                axis=1,
            )
        )
        + 1
    )

    starts = np.concatenate(
        (
            np.array(
                [0],
                dtype=np.int64,
            ),
            boundaries,
        )
    )

    ends = np.concatenate(
        (
            boundaries,
            np.array(
                [sorted_keys.shape[0]],
                dtype=np.int64,
            ),
        )
    )

    spatial_hash: dict[
        tuple[int, int, int],
        np.ndarray,
    ] = {}

    for start, end in zip(
        starts,
        ends,
    ):
        key_array = sorted_keys[start]

        key = (
            int(key_array[0]),
            int(key_array[1]),
            int(key_array[2]),
        )

        spatial_hash[key] = order[start:end]

    return spatial_hash


def _nearest_distances_within_radius(
    source: np.ndarray,
    target: np.ndarray,
    radius: float,
    *,
    cell_size: float | None = None,
) -> np.ndarray:
    """
    Calculate nearest target distances within a finite radius.

    A uniform spatial hash limits distance calculations to nearby
    target cells.

    Points with no target within ``radius`` receive ``np.inf``.
    """
    if source.shape[0] == 0:
        return np.empty(
            0,
            dtype=np.float64,
        )

    if target.shape[0] == 0:
        return np.full(
            source.shape[0],
            np.inf,
            dtype=np.float64,
        )

    if (
        not np.isfinite(radius)
        or radius <= 0.0
    ):
        raise ValueError(
            "radius must be finite and > 0"
        )

    if cell_size is None:
        cell_size = radius

    if (
        not np.isfinite(cell_size)
        or cell_size <= 0.0
    ):
        raise ValueError(
            "cell_size must be finite and > 0"
        )

    spatial_hash = _build_spatial_hash(
        target,
        cell_size,
    )

    source_keys = np.floor(
        source / cell_size
    ).astype(np.int64)

    radius_squared = radius * radius

    neighbour_range = int(
        np.ceil(
            radius / cell_size
        )
    )

    distances = np.full(
        source.shape[0],
        np.inf,
        dtype=np.float64,
    )

    unique_source_keys, inverse = np.unique(
        source_keys,
        axis=0,
        return_inverse=True,
    )

    for group_index, source_key in enumerate(
        unique_source_keys
    ):
        source_indices = np.flatnonzero(
            inverse == group_index
        )

        sx = int(source_key[0])
        sy = int(source_key[1])
        sz = int(source_key[2])

        candidate_chunks: list[np.ndarray] = []

        for dx in range(
            -neighbour_range,
            neighbour_range + 1,
        ):
            for dy in range(
                -neighbour_range,
                neighbour_range + 1,
            ):
                for dz in range(
                    -neighbour_range,
                    neighbour_range + 1,
                ):
                    candidate_indices = spatial_hash.get(
                        (
                            sx + dx,
                            sy + dy,
                            sz + dz,
                        )
                    )

                    if candidate_indices is not None:
                        candidate_chunks.append(
                            candidate_indices
                        )

        if not candidate_chunks:
            continue

        candidate_indices = np.concatenate(
            candidate_chunks
        )

        source_points = source[
            source_indices
        ]

        target_points = target[
            candidate_indices
        ]

        differences = (
            source_points[:, None, :]
            - target_points[None, :, :]
        )

        distances_squared = np.sum(
            differences * differences,
            axis=2,
        )

        local_min_squared = np.min(
            distances_squared,
            axis=1,
        )

        valid = (
            local_min_squared
            <= radius_squared
        )

        if np.any(valid):
            valid_indices = source_indices[
                valid
            ]

            distances[valid_indices] = np.sqrt(
                local_min_squared[valid]
            )

    return distances


def _nearest_distances(
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """
    Preserve the original A25 private helper semantics.

    The production temporal path uses the optimized spatial-hash backend.
    """
    return _nearest_distances_bruteforce(
        source,
        target,
    )


def temporal_motion_evidence(
    previous_xyz: np.ndarray,
    current_xyz: np.ndarray,
    config: TemporalMotionConfig | None = None,
) -> np.ndarray:
    """
    Estimate motion evidence for every point in the current frame.

    Values are in [0, 1]:

    - 0.0 means strong temporal consistency.
    - 1.0 means no sufficiently close previous correspondence.
    """
    if config is None:
        config = TemporalMotionConfig()

    previous = _validate_xyz(
        previous_xyz
    )

    current = _validate_xyz(
        current_xyz
    )

    if current.shape[0] == 0:
        return np.empty(
            0,
            dtype=np.float64,
        )

    if previous.shape[0] == 0:
        return np.ones(
            current.shape[0],
            dtype=np.float64,
        )

    _, previous_centres = _unique_voxel_centres(
        previous,
        config.voxel_size,
    )

    distances = _nearest_distances_within_radius(
        current,
        previous_centres,
        config.max_correspondence_distance,
        cell_size=config.max_correspondence_distance,
    )

    threshold = config.motion_threshold
    maximum = config.max_correspondence_distance

    if maximum <= threshold:
        evidence = (
            distances > threshold
        ).astype(np.float64)
    else:
        evidence = np.clip(
            (
                distances - threshold
            )
            / (
                maximum - threshold
            ),
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
    """
    if not isinstance(
        previous_frame,
        SensorFrame,
    ):
        raise TypeError(
            "previous_frame must be a SensorFrame"
        )

    if not isinstance(
        current_frame,
        SensorFrame,
    ):
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
    """
    return {
        "dynamic": temporal_dynamic_probability(
            previous_frame,
            current_frame,
            config=config,
        )
    }