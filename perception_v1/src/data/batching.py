from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from perception_v1.src.data.sample_pipeline import (
    UnifiedPerceptionSample,
)


@dataclass(frozen=True)
class UnifiedPerceptionBatch:
    """
    Whole-frame batch container.

    Frames remain independent because:
    - point counts vary
    - voxel counts vary
    - circular image heights vary by dataset

    No point padding/deletion/reordering is performed.
    """

    samples: tuple[UnifiedPerceptionSample, ...]

    point_batch_index: np.ndarray
    voxel_batch_index: np.ndarray

    point_offsets: np.ndarray
    voxel_offsets: np.ndarray

    total_points: int
    total_voxels: int


def collate_whole_frames(
    samples: list[UnifiedPerceptionSample]
    | tuple[UnifiedPerceptionSample, ...],
) -> UnifiedPerceptionBatch:

    samples = tuple(samples)

    if len(samples) == 0:
        raise ValueError(
            "Cannot collate an empty batch"
        )

    point_counts = np.asarray(
        [
            len(sample.xyz)
            for sample in samples
        ],
        dtype=np.int64,
    )

    voxel_counts = np.asarray(
        [
            len(
                sample.spvcnn.voxel_coordinates
            )
            for sample in samples
        ],
        dtype=np.int64,
    )

    if np.any(point_counts <= 0):
        raise ValueError(
            "Every batch sample must contain points"
        )

    if np.any(voxel_counts <= 0):
        raise ValueError(
            "Every batch sample must contain voxels"
        )

    # Offsets include final total:
    # [0, end_frame0, end_frame1, ...]
    point_offsets = np.concatenate(
        (
            np.array(
                [0],
                dtype=np.int64,
            ),
            np.cumsum(
                point_counts,
                dtype=np.int64,
            ),
        )
    )

    voxel_offsets = np.concatenate(
        (
            np.array(
                [0],
                dtype=np.int64,
            ),
            np.cumsum(
                voxel_counts,
                dtype=np.int64,
            ),
        )
    )

    point_batch_index = np.concatenate(
        [
            np.full(
                count,
                batch_index,
                dtype=np.int64,
            )
            for batch_index, count
            in enumerate(point_counts)
        ]
    )

    voxel_batch_index = np.concatenate(
        [
            np.full(
                count,
                batch_index,
                dtype=np.int64,
            )
            for batch_index, count
            in enumerate(voxel_counts)
        ]
    )

    return UnifiedPerceptionBatch(
        samples=samples,

        point_batch_index=point_batch_index,
        voxel_batch_index=voxel_batch_index,

        point_offsets=point_offsets,
        voxel_offsets=voxel_offsets,

        total_points=int(
            point_offsets[-1]
        ),
        total_voxels=int(
            voxel_offsets[-1]
        ),
    )