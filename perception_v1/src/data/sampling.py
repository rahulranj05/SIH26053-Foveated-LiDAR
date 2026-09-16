from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class SampleReference:
    dataset_id: str
    frame_id: str


class DatasetBalancedSampler:
    """
    Frozen S7-I training sampler.

    Sampling rule:
    1. uniformly choose one dataset
    2. uniformly choose one frame within that dataset

    Dataset size therefore does NOT determine dataset probability.
    """

    def __init__(
        self,
        frames_by_dataset: Mapping[
            str,
            Sequence[str],
        ],
        *,
        seed: int,
    ) -> None:

        if not frames_by_dataset:
            raise ValueError(
                "frames_by_dataset cannot be empty"
            )

        normalized: dict[str, tuple[str, ...]] = {}

        for dataset_id, frame_ids in (
            frames_by_dataset.items()
        ):
            if not isinstance(dataset_id, str):
                raise ValueError(
                    "dataset_id must be a string"
                )

            if not dataset_id:
                raise ValueError(
                    "dataset_id cannot be empty"
                )

            frames = tuple(frame_ids)

            if not frames:
                raise ValueError(
                    f"Dataset {dataset_id} has no frames"
                )

            if any(
                not isinstance(frame_id, str)
                or not frame_id
                for frame_id in frames
            ):
                raise ValueError(
                    f"Invalid frame ID in {dataset_id}"
                )

            if len(set(frames)) != len(frames):
                raise ValueError(
                    f"Duplicate frame ID in {dataset_id}"
                )

            normalized[dataset_id] = frames

        # Sort dataset IDs so RNG behavior is independent
        # of caller dictionary insertion order.
        self._dataset_ids = tuple(
            sorted(normalized)
        )

        self._frames_by_dataset = {
            dataset_id: normalized[dataset_id]
            for dataset_id in self._dataset_ids
        }

        self._seed = int(seed)

        self._rng = np.random.default_rng(
            self._seed
        )

        self._draw_count = 0

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def draw_count(self) -> int:
        return self._draw_count

    @property
    def dataset_ids(self) -> tuple[str, ...]:
        return self._dataset_ids

    def sample_one(self) -> SampleReference:
        dataset_index = int(
            self._rng.integers(
                0,
                len(self._dataset_ids),
            )
        )

        dataset_id = self._dataset_ids[
            dataset_index
        ]

        frames = self._frames_by_dataset[
            dataset_id
        ]

        frame_index = int(
            self._rng.integers(
                0,
                len(frames),
            )
        )

        frame_id = frames[frame_index]

        self._draw_count += 1

        return SampleReference(
            dataset_id=dataset_id,
            frame_id=frame_id,
        )

    def sample_many(
        self,
        count: int,
    ) -> tuple[SampleReference, ...]:

        if count < 0:
            raise ValueError(
                "count must be >= 0"
            )

        return tuple(
            self.sample_one()
            for _ in range(count)
        )

    def state_dict(self) -> dict:
        return {
            "seed": self._seed,
            "draw_count": self._draw_count,
            "rng_state": self._rng.bit_generator.state,
            "dataset_ids": self._dataset_ids,
            "frames_by_dataset": (
                self._frames_by_dataset.copy()
            ),
        }

    def load_state_dict(
        self,
        state: dict,
    ) -> None:

        if tuple(state["dataset_ids"]) != (
            self._dataset_ids
        ):
            raise ValueError(
                "Sampler dataset configuration mismatch"
            )

        state_frames = {
            dataset_id: tuple(frame_ids)
            for dataset_id, frame_ids
            in state[
                "frames_by_dataset"
            ].items()
        }

        if state_frames != self._frames_by_dataset:
            raise ValueError(
                "Sampler frame configuration mismatch"
            )

        if int(state["seed"]) != self._seed:
            raise ValueError(
                "Sampler seed mismatch"
            )

        self._rng.bit_generator.state = (
            state["rng_state"]
        )

        self._draw_count = int(
            state["draw_count"]
        )