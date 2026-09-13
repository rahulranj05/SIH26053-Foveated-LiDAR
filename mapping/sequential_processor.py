"""
A23 — Sequential Sensor Frame Processing

Provides the sequence-level orchestration layer for FoveaMap.

A23 accepts canonical SensorFrame objects and processes them in temporal
order through the existing A20 pipeline.

A23 does not modify:
    - A1–A20 mapping implementations
    - A21 SensorFrame
    - A22 dataset adapters

The processor deliberately requires the caller to provide FoveaMap
importance signals for each frame. It never fabricates semantic,
dynamic, or other importance values.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from mapping.foveamap_pipeline import run_foveamap_pipeline
from mapping.sensor_frame import SensorFrame


SignalProvider = Callable[[SensorFrame], dict[str, np.ndarray]]


@dataclass(frozen=True)
class SequentialFrameResult:
    """
    Result of processing one sequential SensorFrame.

    Parameters
    ----------
    frame_id:
        Source-specific frame identifier.
    timestamp:
        Frame timestamp in seconds.
    delta_time:
        Time elapsed since the previous processed frame.

        ``None`` for the first frame.

    result:
        Complete A20 FoveaMap pipeline result.

    """

    frame_id: Any
    timestamp: float
    delta_time: float | None
    result: Any


@dataclass(frozen=True)
class SequenceProcessingResult:
    """
    Aggregate result for one processed sequence.

    Parameters
    ----------
    frames:
        Per-frame processing results in chronological order.
    total_frames:
        Number of processed frames.
    total_duration_seconds:
        Timestamp span from first to last frame.

    """

    frames: tuple[SequentialFrameResult, ...]
    total_frames: int
    total_duration_seconds: float

    @property
    def frame_ids(self) -> tuple[Any, ...]:
        """Return processed frame IDs in order."""
        return tuple(frame.frame_id for frame in self.frames)

    @property
    def timestamps(self) -> tuple[float, ...]:
        """Return processed timestamps in order."""
        return tuple(frame.timestamp for frame in self.frames)

    @property
    def delta_times(self) -> tuple[float | None, ...]:
        """Return inter-frame time deltas in order."""
        return tuple(frame.delta_time for frame in self.frames)

    @property
    def mean_delta_time(self) -> float:
        """
        Return the mean inter-frame interval.

        Returns 0.0 when fewer than two frames were processed.
        """
        deltas = [
            frame.delta_time
            for frame in self.frames
            if frame.delta_time is not None
        ]

        if not deltas:
            return 0.0

        return float(np.mean(deltas))


class SequentialFoveaMapProcessor:
    """
    Process canonical SensorFrame objects sequentially.

    The processor retains the previously processed frame so that future
    temporal stages can build on the same interface without changing
    callers.

    A23 itself does not alter the A20 pipeline.
    """

    def __init__(
        self,
        signal_provider: SignalProvider,
    ) -> None:
        if not callable(signal_provider):
            raise TypeError("signal_provider must be callable")

        self._signal_provider = signal_provider
        self._previous_frame: SensorFrame | None = None
        self._processed_frames = 0

    @property
    def previous_frame(self) -> SensorFrame | None:
        """Return the previously processed frame."""
        return self._previous_frame

    @property
    def processed_frames(self) -> int:
        """Return the number of successfully processed frames."""
        return self._processed_frames

    @staticmethod
    def _validate_frame(frame: SensorFrame) -> None:
        if not isinstance(frame, SensorFrame):
            raise TypeError("frame must be a SensorFrame")

    def _compute_delta_time(
        self,
        frame: SensorFrame,
    ) -> float | None:
        if self._previous_frame is None:
            return None

        delta_time = float(
            frame.timestamp - self._previous_frame.timestamp
        )

        if delta_time <= 0.0:
            raise ValueError(
                "SensorFrame timestamps must be strictly increasing"
            )

        return delta_time

    def process_frame(
        self,
        frame: SensorFrame,
    ) -> SequentialFrameResult:
        """
        Process one SensorFrame through the existing A20 pipeline.

        Parameters
        ----------
        frame:
            Canonical FoveaMap SensorFrame.

        Returns
        -------
        SequentialFrameResult
            A23 sequence metadata plus the A20 result.

        Raises
        ------
        TypeError
            If ``frame`` is not a SensorFrame.

        ValueError
            If timestamps are not strictly increasing.

        ValueError
            If the signal provider returns no importance signals.
            This error originates from A20 and is intentionally preserved.
        """
        self._validate_frame(frame)

        delta_time = self._compute_delta_time(frame)

        signals = self._signal_provider(frame)

        if not isinstance(signals, dict):
            raise TypeError(
                "signal_provider must return a dictionary of signals"
            )

        result = run_foveamap_pipeline(
            frame.xyz,
            signals,
        )

        self._previous_frame = frame
        self._processed_frames += 1

        return SequentialFrameResult(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            delta_time=delta_time,
            result=result,
        )

    def process_sequence(
        self,
        frames: Iterable[SensorFrame],
    ) -> SequenceProcessingResult:
        """
        Process a complete SensorFrame sequence.

        Frames must already be supplied in chronological order.

        Parameters
        ----------
        frames:
            Iterable of SensorFrame objects.

        Returns
        -------
        SequenceProcessingResult
            Ordered per-frame results and sequence statistics.
        """
        results: list[SequentialFrameResult] = []

        for frame in frames:
            results.append(self.process_frame(frame))

        if results:
            total_duration = (
                results[-1].timestamp - results[0].timestamp
            )
        else:
            total_duration = 0.0

        return SequenceProcessingResult(
            frames=tuple(results),
            total_frames=len(results),
            total_duration_seconds=float(total_duration),
        )

    def reset(self) -> None:
        """
        Reset temporal processor state.

        The next processed frame becomes the first frame of a new sequence.
        """
        self._previous_frame = None
        self._processed_frames = 0


def process_sensor_sequence(
    frames: Iterable[SensorFrame],
    signal_provider: SignalProvider,
) -> SequenceProcessingResult:
    """
    Convenience function for processing one complete sequence.

    This creates a fresh SequentialFoveaMapProcessor for the sequence.
    """
    processor = SequentialFoveaMapProcessor(
        signal_provider=signal_provider,
    )

    return processor.process_sequence(frames)


def iter_sensor_frames(
    frames: Iterable[SensorFrame],
) -> Iterator[SensorFrame]:
    """
    Validate and yield SensorFrames lazily.

    This helper is useful when sequence data is coming from a dataset
    adapter or filesystem iterator and should not be loaded entirely
    into memory.
    """
    for frame in frames:
        if not isinstance(frame, SensorFrame):
            raise TypeError(
                "iter_sensor_frames received a non-SensorFrame object"
            )

        yield frame