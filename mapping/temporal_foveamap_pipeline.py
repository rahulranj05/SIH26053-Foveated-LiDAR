"""
A31 — Temporal FoveaMap Pipeline

Connects sequential SensorFrames to the frozen A20 FoveaMap pipeline.

A31 integrates:

    A21 SensorFrame
        |
        v
    Production Foveation Signals
        |
        +--> DISTANCE
        +--> KINEMATIC
        +--> PREDICTED_PATH
        |
        v
    A30 Semantic + Dynamic Signal Integration
        |
        v
    A20 FoveaMap Pipeline
        |
        +--> A17 Foveation Controller
        |
        +--> A18.3 Hierarchical Foveated Mapper

Temporal responsibilities
-------------------------
A31 owns:

    - ordered SensorFrame processing
    - previous-frame retention
    - timestamp / delta-time validation
    - optional ego-motion provider execution
    - production base signal construction
    - optional caller-supplied base signal override
    - A30 signal integration
    - adaptation of A30 signal names to A17/A18.3 canonical names
    - A20 execution
    - per-frame result retention
    - sequence statistics
    - processor reset

A31 does NOT modify the frozen A20, A17, A18.3, A21, A25, A26,
A29, or A30 implementations.

Production signal ownership
----------------------------
When no base_signal_provider is supplied, A31 constructs the production
foveation signals directly from the canonical SensorFrame:

    DISTANCE
    KINEMATIC
    PREDICTED_PATH

The production signal builder does not fuse signals or choose a
resolution. Those responsibilities remain with A17.

A caller-supplied base_signal_provider remains supported for tests,
experiments, ablations, and custom signal sources.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Hashable, Iterable

import numpy as np

from mapping.ego_motion import EgoMotion
from mapping.foveamap_pipeline import (
    FoveaMapPipelineResult,
    run_foveamap_pipeline,
)
from mapping.production_foveation_signals import (
    ProductionFoveationSignalConfig,
    build_production_foveation_signals,
)
from mapping.sensor_frame import SensorFrame
from mapping.signal_integration import (
    SignalIntegrationConfig,
    build_integrated_signals,
)


# ============================================================================
# Type aliases
# ============================================================================

BaseSignalProvider = Callable[
    [SensorFrame],
    dict[str, np.ndarray],
]

EgoMotionProvider = Callable[
    [SensorFrame, SensorFrame],
    EgoMotion,
]


# ============================================================================
# Per-frame result
# ============================================================================


@dataclass(frozen=True)
class TemporalFoveaMapFrameResult:
    """
    Result produced by A31 for one SensorFrame.

    Attributes
    ----------
    frame_id:
        Original SensorFrame frame identifier.

    timestamp:
        Original SensorFrame timestamp.

    delta_time:
        Time elapsed since the previous processed frame.
        None for the first frame.

    ego_motion:
        EgoMotion returned by the configured provider for this frame.
        None for the first frame or when no provider is configured.

    signals:
        Complete signal dictionary used by A31.

        A31 exposes canonical production signals as:

            DISTANCE / KINEMATIC / PREDICTED_PATH

        A30 semantic/dynamic signals remain:

            semantic
            dynamic

        Internally, all signals are adapted to the canonical uppercase
        A17/A18.3 reason names before entering A20.

    pipeline_result:
        Exact FoveaMapPipelineResult returned by A20.
    """

    frame_id: Hashable
    timestamp: float
    delta_time: float | None
    ego_motion: EgoMotion | None
    signals: dict[str, np.ndarray]
    pipeline_result: FoveaMapPipelineResult


# ============================================================================
# Sequence result
# ============================================================================


@dataclass(frozen=True)
class TemporalFoveaMapSequenceResult:
    """
    Aggregate result for a processed SensorFrame sequence.

    Attributes
    ----------
    frames:
        Per-frame A31 results in processing order.

    total_frames:
        Number of processed frames.

    frame_ids:
        Frame identifiers in processing order.

    timestamps:
        Frame timestamps in processing order.

    delta_times:
        Per-frame time deltas. The first entry is None.

    mean_delta_time:
        Mean inter-frame delta time, or None when fewer than two
        frames were processed.

    total_duration_seconds:
        Timestamp duration from the first processed frame to the last.
        Zero for an empty or single-frame sequence.
    """

    frames: tuple[TemporalFoveaMapFrameResult, ...]
    total_frames: int
    frame_ids: tuple[Hashable, ...]
    timestamps: tuple[float, ...]
    delta_times: tuple[float | None, ...]
    mean_delta_time: float | None
    total_duration_seconds: float


# ============================================================================
# Validation helpers
# ============================================================================


def _validate_base_signal_provider(
    provider: BaseSignalProvider | None,
) -> None:
    """
    Validate an optional caller-supplied base signal provider.

    None is valid because A31 now has a native production signal builder.
    """

    if provider is not None and not callable(provider):
        raise TypeError(
            "base_signal_provider must be callable or None"
        )


def _validate_production_signal_config(
    config: ProductionFoveationSignalConfig | None,
) -> None:
    """Validate the optional production foveation configuration."""

    if config is not None and not isinstance(
        config,
        ProductionFoveationSignalConfig,
    ):
        raise TypeError(
            "production_foveation_config must be a "
            "ProductionFoveationSignalConfig or None"
        )


def _validate_frame(
    frame: SensorFrame,
) -> None:
    """Validate that a sequence item is a canonical SensorFrame."""

    if not isinstance(frame, SensorFrame):
        raise TypeError(
            "sequence items must be SensorFrame instances"
        )


def _validate_base_signals(
    signals: dict[str, np.ndarray],
    num_points: int,
) -> dict[str, np.ndarray]:
    """
    Validate per-point importance signals.

    A31 accepts either production signals or caller-supplied signals.
    Every signal must contain one normalized importance value per point.
    """

    if not isinstance(signals, dict):
        raise TypeError(
            "base_signal_provider must return a dictionary"
        )

    validated: dict[str, np.ndarray] = {}

    for name, values in signals.items():

        if not isinstance(name, str):
            raise TypeError(
                "base signal names must be strings"
            )

        array = np.asarray(
            values,
            dtype=np.float64,
        )

        if array.ndim != 1:
            raise ValueError(
                f"base signal '{name}' must be one-dimensional"
            )

        if array.shape[0] != num_points:
            raise ValueError(
                f"base signal '{name}' length must match "
                "the current frame point count"
            )

        if not np.all(np.isfinite(array)):
            raise ValueError(
                f"base signal '{name}' must contain only finite values"
            )

        if np.any(array < 0.0) or np.any(array > 1.0):
            raise ValueError(
                f"base signal '{name}' values must be in [0, 1]"
            )

        validated[name] = array

    return validated


# ============================================================================
# A30 -> A20 signal-name adapter
# ============================================================================


_CANONICAL_REASON_NAMES = {
    "distance": "DISTANCE",
    "kinematic": "KINEMATIC",
    "predicted_path": "PREDICTED_PATH",
    "semantic": "SEMANTIC",
    "dynamic": "DYNAMIC",
}


def _canonical_reason_name(name: str) -> str:
    """
    Convert a signal name into the canonical A17/A18.3 reason name.

    A30 intentionally exposes lowercase semantic/dynamic names.
    A17/A18.3 require the canonical uppercase reason ontology.

    Existing canonical names are preserved.
    """

    if not isinstance(name, str):
        raise TypeError(
            "signal names must be strings"
        )

    if name in (
        "DISTANCE",
        "KINEMATIC",
        "PREDICTED_PATH",
        "SEMANTIC",
        "DYNAMIC",
    ):
        return name

    normalized = name.strip().lower()

    if normalized in _CANONICAL_REASON_NAMES:
        return _CANONICAL_REASON_NAMES[normalized]

    # Preserve unknown caller-defined signals rather than silently
    # changing their identity. A17 can still consume arbitrary
    # importance-signal names; A18.3 only requires the resulting
    # dominant reason to be canonical.
    return name


def _canonicalize_signals_for_a20(
    signals: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Adapt A31/A30 signal names to the A20/A17/A18.3 interface.

    The returned dictionary is only for the A20 call.

    The original A31-facing dictionary remains unchanged so callers
    can continue to inspect ``semantic`` and ``dynamic`` signals.
    """

    canonical: dict[str, np.ndarray] = {}

    for name, values in signals.items():

        canonical_name = _canonical_reason_name(name)

        if canonical_name in canonical:
            raise ValueError(
                "multiple signals resolve to the same canonical "
                f"reason name '{canonical_name}'"
            )

        canonical[canonical_name] = np.asarray(
            values,
            dtype=np.float64,
        )

    return canonical


# ============================================================================
# Processor
# ============================================================================


class TemporalFoveaMapProcessor:
    """
    Stateful A31 temporal FoveaMap processor.

    Parameters
    ----------
    base_signal_provider:
        Optional callable producing caller-supplied per-point importance
        signals for every SensorFrame.

        When omitted, A31 uses the native production foveation signal
        builder and constructs:

            DISTANCE
            KINEMATIC
            PREDICTED_PATH

        directly from the SensorFrame.

        A caller-supplied provider overrides the production builder.
        This preserves support for tests, experiments, ablations, and
        custom signal sources.

    ego_motion_provider:
        Optional callable receiving:

            current_frame, previous_frame

        and returning an EgoMotion object.

        It is called only when a previous frame exists.

    signal_config:
        Optional A30 SignalIntegrationConfig.

    production_foveation_config:
        Optional configuration for the native production foveation
        signal builder.
    """

    def __init__(
        self,
        base_signal_provider: BaseSignalProvider | None = None,
        *,
        ego_motion_provider: EgoMotionProvider | None = None,
        signal_config: SignalIntegrationConfig | None = None,
        production_foveation_config: (
            ProductionFoveationSignalConfig | None
        ) = None,
    ) -> None:

        _validate_base_signal_provider(
            base_signal_provider
        )

        if (
            ego_motion_provider is not None
            and not callable(ego_motion_provider)
        ):
            raise TypeError(
                "ego_motion_provider must be callable or None"
            )

        if signal_config is not None and not isinstance(
            signal_config,
            SignalIntegrationConfig,
        ):
            raise TypeError(
                "signal_config must be a SignalIntegrationConfig or None"
            )

        _validate_production_signal_config(
            production_foveation_config
        )

        self._base_signal_provider = (
            base_signal_provider
        )

        self._ego_motion_provider = (
            ego_motion_provider
        )

        self._signal_config = (
            signal_config
            if signal_config is not None
            else SignalIntegrationConfig()
        )

        self._production_foveation_config = (
            production_foveation_config
            if production_foveation_config is not None
            else ProductionFoveationSignalConfig()
        )

        self._previous_frame: SensorFrame | None = None

        self._processed_frames: list[
            TemporalFoveaMapFrameResult
        ] = []

    # ------------------------------------------------------------------------
    # Public state
    # ------------------------------------------------------------------------

    @property
    def previous_frame(self) -> SensorFrame | None:
        """Return the most recently processed SensorFrame."""

        return self._previous_frame

    @property
    def processed_frames(
        self,
    ) -> tuple[TemporalFoveaMapFrameResult, ...]:
        """Return all results processed since the last reset."""

        return tuple(
            self._processed_frames
        )

    # ------------------------------------------------------------------------
    # Signal construction
    # ------------------------------------------------------------------------

    def _build_base_signals(
        self,
        frame: SensorFrame,
    ) -> dict[str, np.ndarray]:
        """
        Build the A31 base foveation signals for one frame.

        A caller-supplied provider takes precedence. Otherwise the
        native production signal builder is used.
        """

        if self._base_signal_provider is not None:
            signals = self._base_signal_provider(
                frame
            )
        else:
            signals = build_production_foveation_signals(
                frame,
                config=self._production_foveation_config,
            )

        return _validate_base_signals(
            signals,
            frame.num_points,
        )

    # ------------------------------------------------------------------------
    # Single-frame processing
    # ------------------------------------------------------------------------

    def process_frame(
        self,
        frame: SensorFrame,
    ) -> TemporalFoveaMapFrameResult:
        """
        Process one SensorFrame through production/A30 signals and A20.

        Frames must have strictly increasing timestamps.
        """

        _validate_frame(frame)

        previous_frame = self._previous_frame

        # --------------------------------------------------------------------
        # Temporal ordering
        # --------------------------------------------------------------------

        delta_time: float | None = None

        if previous_frame is not None:

            delta_time = (
                float(frame.timestamp)
                - float(previous_frame.timestamp)
            )

            if delta_time <= 0.0:
                raise ValueError(
                    "SensorFrame timestamps must be strictly increasing"
                )

        # --------------------------------------------------------------------
        # Optional ego-motion provider
        # --------------------------------------------------------------------

        ego_motion: EgoMotion | None = None

        if (
            previous_frame is not None
            and self._ego_motion_provider is not None
        ):
            ego_motion = self._ego_motion_provider(
                frame,
                previous_frame,
            )

            if not isinstance(
                ego_motion,
                EgoMotion,
            ):
                raise TypeError(
                    "ego_motion_provider must return an EgoMotion"
                )

        # --------------------------------------------------------------------
        # Production or caller-supplied base signals
        # --------------------------------------------------------------------

        base_signals = self._build_base_signals(
            frame
        )

        # --------------------------------------------------------------------
        # A30 — semantic + dynamic integration
        # --------------------------------------------------------------------

        integrated_signals = build_integrated_signals(
            current_frame=frame,
            previous_frame=previous_frame,
            ego_motion=ego_motion,
            config=self._signal_config,
        )

        # --------------------------------------------------------------------
        # Combine base + A30 signals.
        #
        # A30 signal names intentionally remain lowercase here so the
        # public A31 result exposes the actual A30 API.
        # --------------------------------------------------------------------

        signals: dict[str, np.ndarray] = dict(
            base_signals
        )

        for name, values in integrated_signals.items():

            if name in signals:
                raise ValueError(
                    f"signal '{name}' is already supplied by "
                    "base_signal_provider"
                )

            signals[name] = np.asarray(
                values,
                dtype=np.float64,
            )

        # The native production builder always provides three signals.
        # A custom provider may intentionally return an empty dictionary;
        # A20 requires at least one importance signal, so reject it here.
        if not signals:
            raise ValueError(
                "base and integrated signals must contain at least "
                "one importance signal"
            )

        # --------------------------------------------------------------------
        # Validate every final signal before crossing into A20.
        # --------------------------------------------------------------------

        signals = _validate_base_signals(
            signals,
            frame.num_points,
        )

        # --------------------------------------------------------------------
        # A31 adapter boundary.
        #
        # Keep ``signals`` unchanged for the public result, but canonicalize
        # names for the frozen A20 -> A17 -> A18.3 chain.
        # --------------------------------------------------------------------

        a20_signals = _canonicalize_signals_for_a20(
            signals
        )

        # --------------------------------------------------------------------
        # A20 — complete FoveaMap pipeline
        # --------------------------------------------------------------------

        pipeline_result = run_foveamap_pipeline(
            frame.xyz,
            a20_signals,
        )

        if not isinstance(
            pipeline_result,
            FoveaMapPipelineResult,
        ):
            raise TypeError(
                "run_foveamap_pipeline() must return "
                "a FoveaMapPipelineResult"
            )

        # --------------------------------------------------------------------
        # Build immutable per-frame result.
        # --------------------------------------------------------------------

        frame_result = TemporalFoveaMapFrameResult(
            frame_id=frame.frame_id,
            timestamp=float(frame.timestamp),
            delta_time=delta_time,
            ego_motion=ego_motion,
            signals={
                name: np.asarray(
                    values,
                    dtype=np.float64,
                ).copy()
                for name, values in signals.items()
            },
            pipeline_result=pipeline_result,
        )

        self._processed_frames.append(
            frame_result
        )

        self._previous_frame = frame

        return frame_result

    # ------------------------------------------------------------------------
    # Sequence processing
    # ------------------------------------------------------------------------

    def process_sequence(
        self,
        frames: Iterable[SensorFrame],
    ) -> TemporalFoveaMapSequenceResult:
        """
        Process a sequence of SensorFrames in iteration order.

        The processor state is intentionally retained between calls.
        Therefore multiple sequence calls can form one continuous temporal
        stream.
        """

        results: list[
            TemporalFoveaMapFrameResult
        ] = []

        for frame in frames:
            results.append(
                self.process_frame(frame)
            )

        return _build_sequence_result(
            results
        )

    # ------------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------------

    def reset(self) -> None:
        """Clear temporal and accumulated processing state."""

        self._previous_frame = None
        self._processed_frames.clear()


# ============================================================================
# Sequence result construction
# ============================================================================


def _build_sequence_result(
    frames: list[TemporalFoveaMapFrameResult],
) -> TemporalFoveaMapSequenceResult:
    """Construct aggregate sequence statistics."""

    frame_tuple = tuple(frames)

    total_frames = len(
        frame_tuple
    )

    frame_ids = tuple(
        frame.frame_id
        for frame in frame_tuple
    )

    timestamps = tuple(
        frame.timestamp
        for frame in frame_tuple
    )

    delta_times = tuple(
        frame.delta_time
        for frame in frame_tuple
    )

    valid_delta_times = tuple(
        delta
        for delta in delta_times
        if delta is not None
    )

    if valid_delta_times:
        mean_delta_time = float(
            np.mean(
                np.asarray(
                    valid_delta_times,
                    dtype=np.float64,
                )
            )
        )
    else:
        mean_delta_time = None

    if total_frames >= 2:
        total_duration_seconds = (
            timestamps[-1]
            - timestamps[0]
        )
    else:
        total_duration_seconds = 0.0

    return TemporalFoveaMapSequenceResult(
        frames=frame_tuple,
        total_frames=total_frames,
        frame_ids=frame_ids,
        timestamps=timestamps,
        delta_times=delta_times,
        mean_delta_time=mean_delta_time,
        total_duration_seconds=float(
            total_duration_seconds
        ),
    )


# ============================================================================
# Convenience function
# ============================================================================


def process_temporal_foveamap_sequence(
    frames: Iterable[SensorFrame],
    base_signal_provider: BaseSignalProvider | None = None,
    *,
    ego_motion_provider: EgoMotionProvider | None = None,
    signal_config: SignalIntegrationConfig | None = None,
    production_foveation_config: (
        ProductionFoveationSignalConfig | None
    ) = None,
) -> TemporalFoveaMapSequenceResult:
    """
    Convenience wrapper for processing a SensorFrame sequence.

    ``base_signal_provider`` remains the second positional argument for
    backwards compatibility.

    When omitted, the native production foveation signal builder is used.
    """

    processor = TemporalFoveaMapProcessor(
        base_signal_provider,
        ego_motion_provider=ego_motion_provider,
        signal_config=signal_config,
        production_foveation_config=production_foveation_config,
    )

    return processor.process_sequence(
        frames
    )