"""
A32 — Sequential Dataset → Temporal FoveaMap Runner

Connects the frozen A29 SequentialLidarDataset interface to the frozen
A31 TemporalFoveaMapProcessor.

Architecture
------------

    A29 SequentialLidarDataset
                |
                v
         SensorFrame stream
                |
                v
    A31 TemporalFoveaMapProcessor
                |
                v
       TemporalFoveaMapSequenceResult

A32 contains orchestration only.

It does NOT modify:
    - A20 FoveaMap pipeline
    - A21 SensorFrame
    - A25 temporal motion
    - A26 ego-motion
    - A29 SequentialLidarDataset
    - A30 signal integration
    - A31 temporal FoveaMap processing
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from mapping.ego_motion import EgoMotion
from mapping.sequential_dataset import SequentialLidarDataset
from mapping.sensor_frame import SensorFrame
from mapping.signal_integration import SignalIntegrationConfig
from mapping.temporal_foveamap_pipeline import (
    BaseSignalProvider,
    EgoMotionProvider,
    TemporalFoveaMapProcessor,
    TemporalFoveaMapSequenceResult,
)


# ============================================================================
# Dataset runner
# ============================================================================


class SequentialDatasetFoveaMapRunner:
    """
    Run the A31 temporal FoveaMap processor over an A29 dataset.

    Parameters
    ----------
    dataset:
        Existing A29 SequentialLidarDataset instance.

    base_signal_provider:
        Required A31 caller-supplied importance-signal provider.

    ego_motion_provider:
        Optional A31 ego-motion provider.

    signal_config:
        Optional A30 signal-integration configuration.
    """

    def __init__(
        self,
        dataset: SequentialLidarDataset,
        base_signal_provider: BaseSignalProvider,
        *,
        ego_motion_provider: EgoMotionProvider | None = None,
        signal_config: SignalIntegrationConfig | None = None,
    ) -> None:

        if not isinstance(
            dataset,
            SequentialLidarDataset,
        ):
            raise TypeError(
                "dataset must be a SequentialLidarDataset"
            )

        if not callable(base_signal_provider):
            raise TypeError(
                "base_signal_provider must be callable"
            )

        if (
            ego_motion_provider is not None
            and not callable(ego_motion_provider)
        ):
            raise TypeError(
                "ego_motion_provider must be callable or None"
            )

        self._dataset = dataset

        self._processor = TemporalFoveaMapProcessor(
            base_signal_provider,
            ego_motion_provider=ego_motion_provider,
            signal_config=signal_config,
        )

    # ------------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------------

    @property
    def dataset(self) -> SequentialLidarDataset:
        """Return the dataset being processed."""

        return self._dataset

    @property
    def processor(self) -> TemporalFoveaMapProcessor:
        """Return the underlying A31 processor."""

        return self._processor

    @property
    def processed_frames(self):
        """Return A31 results accumulated by the processor."""

        return self._processor.processed_frames

    # ------------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------------

    def run(self) -> TemporalFoveaMapSequenceResult:
        """
        Load all dataset frames and run the A31 temporal pipeline.

        The dataset's deterministic iteration order is preserved.
        """

        frames = self._dataset.iter_frames()

        result = self._processor.process_sequence(
            frames
        )

        if result.total_frames != self._dataset.frame_count:
            raise RuntimeError(
                "A32 processed frame count does not match dataset "
                "frame count"
            )

        if result.frame_ids != self._dataset.frame_ids:
            raise RuntimeError(
                "A32 processed frame IDs do not match dataset order"
            )

        return result

    def reset(self) -> None:
        """Reset the underlying A31 temporal processor."""

        self._processor.reset()


# ============================================================================
# Convenience function
# ============================================================================


def run_sequential_dataset_foveamap(
    dataset: SequentialLidarDataset,
    base_signal_provider: BaseSignalProvider,
    *,
    ego_motion_provider: EgoMotionProvider | None = None,
    signal_config: SignalIntegrationConfig | None = None,
) -> TemporalFoveaMapSequenceResult:
    """
    Run A31 over an existing A29 SequentialLidarDataset.
    """

    runner = SequentialDatasetFoveaMapRunner(
        dataset,
        base_signal_provider,
        ego_motion_provider=ego_motion_provider,
        signal_config=signal_config,
    )

    return runner.run()


# ============================================================================
# Path convenience function
# ============================================================================


def run_sequential_dataset_path(
    root: str | Path,
    base_signal_provider: BaseSignalProvider,
    *,
    ego_motion_provider: EgoMotionProvider | None = None,
    signal_config: SignalIntegrationConfig | None = None,
) -> TemporalFoveaMapSequenceResult:
    """
    Construct an A29 SequentialLidarDataset from a filesystem path and
    immediately run the A31 temporal FoveaMap pipeline.

    This convenience function intentionally uses A29 defaults.

    Dataset-specific timestamp, vehicle-state, and semantic-label providers
    should be configured by constructing SequentialLidarDataset explicitly
    and passing it to run_sequential_dataset_foveamap().
    """

    dataset = SequentialLidarDataset(
        root
    )

    return run_sequential_dataset_foveamap(
        dataset,
        base_signal_provider,
        ego_motion_provider=ego_motion_provider,
        signal_config=signal_config,
    )