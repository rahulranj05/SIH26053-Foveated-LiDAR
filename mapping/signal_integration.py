"""
A30 — Semantic + Dynamic Signal Integration

Integrates canonical SensorFrame annotations and temporal motion evidence
into the per-point importance-signal interface consumed by FoveaMap.

A30 is intentionally an integration layer.

It does NOT modify:
    - A15 semantic foveation
    - A17 foveation controller
    - A20 FoveaMap pipeline
    - A21 SensorFrame
    - A23 sequential processor
    - A25 temporal motion
    - A26 ego-motion compensation

Signal flow
-----------

Current SensorFrame
    |
    +-- semantic_labels
    |       |
    |       +--> A15 semantic_importance()
    |
    +-- dynamic_probability
    |       |
    |       +--> existing dynamic signal
    |
    +-- previous SensorFrame
            |
            +--> A26 ego-motion compensation (optional)
            |
            +--> A25 temporal motion
                    |
                    +--> dynamic signal

If no semantic or dynamic information is available, A30 does not fabricate
a signal. In that case the returned dictionary is empty and the caller may
combine the result with another subsystem such as DISTANCE or KINEMATIC.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mapping.ego_motion import EgoMotion, compensate_previous_frame
from mapping.semantic_foveation import (
    SemanticFoveationConfig,
    semantic_importance,
)
from mapping.sensor_frame import SensorFrame
from mapping.temporal_motion import (
    TemporalMotionConfig,
    temporal_dynamic_probability,
)


@dataclass(frozen=True)
class SignalIntegrationConfig:
    """
    Configuration for A30 semantic and dynamic signal integration.

    Parameters
    ----------
    semantic_config:
        Configuration forwarded to A15 semantic foveation.

    temporal_config:
        Configuration forwarded to A25 temporal motion estimation.

    use_temporal_motion:
        Whether A30 may derive dynamic probability from consecutive
        SensorFrames when the current frame does not already provide an
        explicit dynamic probability.

    prefer_supplied_dynamic_probability:
        If True, an explicit dynamic_probability stored in the current
        SensorFrame takes precedence over temporal estimation.

    compensate_ego_motion:
        If True, temporal comparison uses A26 compensation when an
        EgoMotion object is supplied.

        If True but no EgoMotion is supplied, the previous frame is used
        unchanged rather than fabricating a motion estimate.
    """

    semantic_config: SemanticFoveationConfig = SemanticFoveationConfig()
    temporal_config: TemporalMotionConfig = TemporalMotionConfig()
    use_temporal_motion: bool = True
    prefer_supplied_dynamic_probability: bool = True
    compensate_ego_motion: bool = True

    def __post_init__(self) -> None:
        if not isinstance(
            self.semantic_config,
            SemanticFoveationConfig,
        ):
            raise TypeError(
                "semantic_config must be a SemanticFoveationConfig"
            )

        if not isinstance(
            self.temporal_config,
            TemporalMotionConfig,
        ):
            raise TypeError(
                "temporal_config must be a TemporalMotionConfig"
            )

        if not isinstance(
            self.use_temporal_motion,
            bool,
        ):
            raise TypeError(
                "use_temporal_motion must be a bool"
            )

        if not isinstance(
            self.prefer_supplied_dynamic_probability,
            bool,
        ):
            raise TypeError(
                "prefer_supplied_dynamic_probability must be a bool"
            )

        if not isinstance(
            self.compensate_ego_motion,
            bool,
        ):
            raise TypeError(
                "compensate_ego_motion must be a bool"
            )


def _validate_frame(
    frame: SensorFrame,
    name: str,
) -> None:
    """Validate that an object is a canonical SensorFrame."""
    if not isinstance(frame, SensorFrame):
        raise TypeError(
            f"{name} must be a SensorFrame"
        )


def _validate_signal(
    signal: np.ndarray,
    expected_length: int,
    name: str,
) -> np.ndarray:
    """
    Validate and normalize one A20-compatible importance signal.
    """
    signal = np.asarray(
        signal,
        dtype=np.float64,
    )

    if signal.ndim != 1:
        raise ValueError(
            f"{name} must be a 1D array"
        )

    if signal.shape[0] != expected_length:
        raise ValueError(
            f"{name} length must match the current frame point count"
        )

    if not np.all(np.isfinite(signal)):
        raise ValueError(
            f"{name} must contain only finite values"
        )

    if np.any(signal < 0.0) or np.any(signal > 1.0):
        raise ValueError(
            f"{name} values must be in [0, 1]"
        )

    return signal


def semantic_signal(
    frame: SensorFrame,
    config: SemanticFoveationConfig | None = None,
) -> np.ndarray | None:
    """
    Convert current-frame semantic labels into semantic importance.

    Returns None when the frame has no semantic labels.

    No semantic information is fabricated.
    """
    _validate_frame(
        frame,
        "frame",
    )

    if frame.semantic_labels is None:
        return None

    if config is None:
        config = SemanticFoveationConfig()

    importance = semantic_importance(
        frame.semantic_labels,
        config=config,
    )

    return _validate_signal(
        importance,
        frame.num_points,
        "semantic importance",
    )


def _temporal_previous_points(
    previous_frame: SensorFrame,
    current_frame: SensorFrame,
    ego_motion: EgoMotion | None,
    config: SignalIntegrationConfig,
) -> np.ndarray:
    """
    Prepare previous-frame points for temporal comparison.

    When A26 compensation is enabled and an EgoMotion is supplied, the
    previous frame is transformed into the current-frame coordinate system.

    Otherwise the previous frame is returned unchanged.
    """
    if (
        config.compensate_ego_motion
        and ego_motion is not None
    ):
        return compensate_previous_frame(
            previous_frame.xyz,
            ego_motion,
        )

    return np.asarray(
        previous_frame.xyz,
        dtype=np.float64,
    )


def dynamic_signal(
    current_frame: SensorFrame,
    previous_frame: SensorFrame | None = None,
    ego_motion: EgoMotion | None = None,
    config: SignalIntegrationConfig | None = None,
) -> np.ndarray | None:
    """
    Produce the dynamic importance signal for the current frame.

    Priority
    --------
    1. Explicit ``current_frame.dynamic_probability`` when configured
       to take precedence.
    2. A25 temporal dynamic probability when a previous frame exists
       and temporal estimation is enabled.
    3. None when neither source is available.

    Ego-motion
    ----------
    If an EgoMotion is supplied and compensation is enabled, A26 transforms
    the previous point cloud into the current-frame coordinate system before
    A25 computes temporal motion evidence.

    The current frame itself is never transformed.

    No dynamic information is fabricated.
    """
    _validate_frame(
        current_frame,
        "current_frame",
    )

    if previous_frame is not None:
        _validate_frame(
            previous_frame,
            "previous_frame",
        )

    if ego_motion is not None and not isinstance(
        ego_motion,
        EgoMotion,
    ):
        raise TypeError(
            "ego_motion must be an EgoMotion or None"
        )

    if config is None:
        config = SignalIntegrationConfig()

    if (
        current_frame.dynamic_probability is not None
        and config.prefer_supplied_dynamic_probability
    ):
        return _validate_signal(
            current_frame.dynamic_probability,
            current_frame.num_points,
            "dynamic probability",
        )

    if (
        previous_frame is None
        or not config.use_temporal_motion
    ):
        if current_frame.dynamic_probability is not None:
            return _validate_signal(
                current_frame.dynamic_probability,
                current_frame.num_points,
                "dynamic probability",
            )

        return None

    previous_points = _temporal_previous_points(
        previous_frame,
        current_frame,
        ego_motion,
        config,
    )

    # A25 itself requires SensorFrame objects for its public frame-based
    # helper. For ego-compensated comparison, construct a temporary frame
    # carrying the compensated previous XYZ while preserving the canonical
    # previous-frame metadata.
    compensated_previous_frame = SensorFrame(
        xyz=previous_points,
        vehicle_state=previous_frame.vehicle_state,
        timestamp=previous_frame.timestamp,
        frame_id=previous_frame.frame_id,
        semantic_labels=None,
        dynamic_probability=None,
    )

    probability = temporal_dynamic_probability(
        compensated_previous_frame,
        current_frame,
        config=config.temporal_config,
    )

    return _validate_signal(
        probability,
        current_frame.num_points,
        "temporal dynamic probability",
    )


def build_integrated_signals(
    current_frame: SensorFrame,
    previous_frame: SensorFrame | None = None,
    ego_motion: EgoMotion | None = None,
    config: SignalIntegrationConfig | None = None,
) -> dict[str, np.ndarray]:
    """
    Build all available A30 semantic and dynamic importance signals.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing zero or more of:

            ``semantic``
                A15-derived semantic importance.

            ``dynamic``
                Explicit SensorFrame dynamic probability or A25-derived
                temporal dynamic probability.

    Missing information is omitted rather than fabricated.

    Notes
    -----
    This dictionary is directly compatible with the signal-provider contract
    used by A23 and the importance-signal input expected by A20/A17.

    A30 intentionally does not add distance, kinematic, or predicted-path
    signals. Those remain independent subsystems.
    """
    _validate_frame(
        current_frame,
        "current_frame",
    )

    if previous_frame is not None:
        _validate_frame(
            previous_frame,
            "previous_frame",
        )

    if config is None:
        config = SignalIntegrationConfig()

    signals: dict[str, np.ndarray] = {}

    semantic = semantic_signal(
        current_frame,
        config=config.semantic_config,
    )

    if semantic is not None:
        signals["semantic"] = semantic

    dynamic = dynamic_signal(
        current_frame,
        previous_frame=previous_frame,
        ego_motion=ego_motion,
        config=config,
    )

    if dynamic is not None:
        signals["dynamic"] = dynamic

    return signals


def integrate_signal_provider(
    config: SignalIntegrationConfig | None = None,
):
    """
    Create an A23-compatible signal provider.

    The returned callable accepts a current SensorFrame.

    Since A23's existing SignalProvider interface only passes the current
    frame, this provider uses signals already attached to that frame.
    Temporal A25 integration requires ``build_integrated_signals`` directly
    or a caller-managed closure that supplies the previous frame.

    This helper is therefore intended for non-temporal annotation signals.
    """
    if config is None:
        config = SignalIntegrationConfig()

    def provider(
        frame: SensorFrame,
    ) -> dict[str, np.ndarray]:
        return build_integrated_signals(
            current_frame=frame,
            previous_frame=None,
            ego_motion=None,
            config=config,
        )

    return provider