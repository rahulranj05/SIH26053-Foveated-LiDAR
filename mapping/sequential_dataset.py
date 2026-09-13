from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np

from mapping.kinematic_foveation import VehicleState
from mapping.semantic_kitti_adapter import load_semantickitti_scan
from mapping.sensor_frame import SensorFrame


TimestampProvider = Callable[[int, Path], float]
VehicleStateProvider = Callable[[int, Path], VehicleState]
SemanticLabelProvider = Callable[[int, Path], np.ndarray | None]


@dataclass(frozen=True)
class SequentialDatasetConfig:
    """
    Configuration for a sequential LiDAR dataset directory.
    """

    scan_suffix: str = ".bin"
    recursive: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.scan_suffix, str):
            raise TypeError("scan_suffix must be a string")

        if not self.scan_suffix:
            raise ValueError("scan_suffix must not be empty")

        if not self.scan_suffix.startswith("."):
            raise ValueError("scan_suffix must start with '.'")

        if not isinstance(self.recursive, bool):
            raise TypeError("recursive must be a boolean")


@dataclass(frozen=True)
class DatasetFrameMetadata:
    """
    Metadata associated with one discovered sequential scan.
    """

    frame_id: str
    path: Path
    index: int
    timestamp: float


class SequentialLidarDataset:
    """
    Dataset-independent sequential LiDAR loader.

    The primary responsibility is discovering ordered scans and converting
    each scan into the frozen SensorFrame interface used by A23.

    This class intentionally does not fabricate semantic or dynamic signals.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        config: SequentialDatasetConfig | None = None,
        timestamp_provider: TimestampProvider | None = None,
        vehicle_state_provider: VehicleStateProvider | None = None,
        semantic_label_provider: SemanticLabelProvider | None = None,
    ) -> None:
        self.root = Path(root)
        self.config = config or SequentialDatasetConfig()
        self.timestamp_provider = timestamp_provider
        self.vehicle_state_provider = vehicle_state_provider
        self.semantic_label_provider = semantic_label_provider

        if not self.root.exists():
            raise FileNotFoundError(
                f"dataset root does not exist: {self.root}"
            )

        if not self.root.is_dir():
            raise NotADirectoryError(
                f"dataset root is not a directory: {self.root}"
            )

        self._scans = self._discover_scans()

    def _discover_scans(self) -> tuple[Path, ...]:
        if self.config.recursive:
            candidates = self.root.rglob(f"*{self.config.scan_suffix}")
        else:
            candidates = self.root.glob(f"*{self.config.scan_suffix}")

        scans = [path for path in candidates if path.is_file()]
        scans.sort(key=_scan_sort_key)

        return tuple(scans)

    @property
    def scans(self) -> tuple[Path, ...]:
        """Return discovered scans in deterministic order."""

        return self._scans

    @property
    def frame_count(self) -> int:
        """Return the number of discovered scans."""

        return len(self._scans)

    @property
    def frame_ids(self) -> tuple[str, ...]:
        """Return frame IDs in dataset order."""

        return tuple(path.stem for path in self._scans)

    def metadata(self, index: int) -> DatasetFrameMetadata:
        """Return metadata for one dataset frame."""

        path = self._path_at(index)

        return DatasetFrameMetadata(
            frame_id=path.stem,
            path=path,
            index=index,
            timestamp=self._timestamp_for(index, path),
        )

    def iter_metadata(self) -> Iterator[DatasetFrameMetadata]:
        """Iterate over frame metadata in deterministic dataset order."""

        for index, path in enumerate(self._scans):
            yield DatasetFrameMetadata(
                frame_id=path.stem,
                path=path,
                index=index,
                timestamp=self._timestamp_for(index, path),
            )

    def load_frame(self, index: int) -> SensorFrame:
        """
        Load one sequential LiDAR scan as a SensorFrame.

        A22 is used for the authoritative SemanticKITTI raw scan decoding.
        A29 then attaches optional dataset metadata to the existing frozen
        SensorFrame interface without modifying A22.
        """

        path = self._path_at(index)
        frame_id = path.stem
        timestamp = self._timestamp_for(index, path)
        vehicle_state = self._vehicle_state_for(index, path)

        xyz = load_semantickitti_scan(path)

        semantic_labels = None

        if self.semantic_label_provider is not None:
            semantic_labels = self.semantic_label_provider(
                index,
                path,
            )

            if semantic_labels is not None:
                semantic_labels = validate_semantic_labels(
                    semantic_labels,
                    xyz.shape[0],
                )

        return SensorFrame(
            xyz=xyz,
            vehicle_state=vehicle_state,
            timestamp=timestamp,
            frame_id=frame_id,
            semantic_labels=semantic_labels,
            dynamic_probability=None,
        )

    def iter_frames(self) -> Iterator[SensorFrame]:
        """Iterate over all scans as SensorFrame objects."""

        for index in range(self.frame_count):
            yield self.load_frame(index)

    def __iter__(self) -> Iterator[SensorFrame]:
        return self.iter_frames()

    def _path_at(self, index: int) -> Path:
        if not isinstance(index, (int, np.integer)):
            raise TypeError("index must be an integer")

        index = int(index)

        if index < 0 or index >= self.frame_count:
            raise IndexError(
                f"frame index {index} out of range "
                f"for dataset with {self.frame_count} frames"
            )

        return self._scans[index]

    def _timestamp_for(self, index: int, path: Path) -> float:
        if self.timestamp_provider is None:
            timestamp = float(index)
        else:
            timestamp = float(
                self.timestamp_provider(index, path)
            )

        if not np.isfinite(timestamp):
            raise ValueError(
                f"timestamp for frame {path.stem} must be finite"
            )

        return timestamp

    def _vehicle_state_for(
        self,
        index: int,
        path: Path,
    ) -> VehicleState:
        if self.vehicle_state_provider is None:
            return VehicleState(
                speed=0.0,
                yaw_rate=0.0,
                heading=0.0,
            )

        state = self.vehicle_state_provider(index, path)

        if not isinstance(state, VehicleState):
            raise TypeError(
                "vehicle_state_provider must return VehicleState"
            )

        return state


def _scan_sort_key(path: Path) -> tuple[int, int | str]:
    """
    Prefer numeric filename ordering.

    Examples:
        2.bin
        10.bin

    are ordered as 2, 10 rather than lexicographically as 10, 2.

    Non-numeric stems are ordered lexicographically after numeric stems.
    """

    stem = path.stem

    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem)


def discover_sequential_scans(
    root: str | Path,
    *,
    scan_suffix: str = ".bin",
    recursive: bool = False,
) -> tuple[Path, ...]:
    """
    Convenience function for deterministic scan discovery.
    """

    dataset = SequentialLidarDataset(
        root,
        config=SequentialDatasetConfig(
            scan_suffix=scan_suffix,
            recursive=recursive,
        ),
    )

    return dataset.scans


def load_sequential_frames(
    root: str | Path,
    *,
    config: SequentialDatasetConfig | None = None,
    timestamp_provider: TimestampProvider | None = None,
    vehicle_state_provider: VehicleStateProvider | None = None,
    semantic_label_provider: SemanticLabelProvider | None = None,
) -> tuple[SensorFrame, ...]:
    """
    Convenience function that loads an entire sequential dataset.
    """

    dataset = SequentialLidarDataset(
        root,
        config=config,
        timestamp_provider=timestamp_provider,
        vehicle_state_provider=vehicle_state_provider,
        semantic_label_provider=semantic_label_provider,
    )

    return tuple(dataset.iter_frames())


def validate_sequential_timestamps(
    frames: Sequence[SensorFrame],
) -> None:
    """
    Validate that SensorFrame timestamps are strictly increasing.
    """

    previous_timestamp: float | None = None

    for frame in frames:
        if not isinstance(frame, SensorFrame):
            raise TypeError(
                "frames must contain only SensorFrame instances"
            )

        if previous_timestamp is not None:
            if frame.timestamp <= previous_timestamp:
                raise ValueError(
                    "SensorFrame timestamps must be strictly increasing"
                )

        previous_timestamp = frame.timestamp


def validate_semantic_labels(
    labels: np.ndarray,
    point_count: int,
) -> np.ndarray:
    """
    Validate a semantic label vector against a LiDAR point count.

    Labels are preserved exactly; this function does not decode or remap
    dataset-specific semantic IDs.
    """

    labels_array = np.asarray(labels)

    if labels_array.ndim != 1:
        raise ValueError("semantic labels must be one-dimensional")

    if labels_array.shape[0] != point_count:
        raise ValueError(
            "semantic label count must match LiDAR point count"
        )

    if not np.issubdtype(labels_array.dtype, np.integer):
        raise ValueError(
            "semantic labels must use an integer dtype"
        )

    if not np.all(np.isfinite(labels_array)):
        raise ValueError(
            "semantic labels must be finite"
        )

    return labels_array.copy()