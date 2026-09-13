"""
A30 — Semantic + Dynamic Signal Integration tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from mapping.ego_motion import EgoMotion
from mapping.kinematic_foveation import VehicleState
from mapping.semantic_foveation import SemanticFoveationConfig
from mapping.sensor_frame import SensorFrame
from mapping.signal_integration import (
    SignalIntegrationConfig,
    build_integrated_signals,
    dynamic_signal,
    integrate_signal_provider,
    semantic_signal,
)
from mapping.temporal_motion import TemporalMotionConfig


def make_frame(
    xyz: np.ndarray,
    timestamp: float,
    frame_id: int,
    semantic_labels: np.ndarray | None = None,
    dynamic_probability: np.ndarray | None = None,
) -> SensorFrame:
    return SensorFrame(
        xyz=xyz,
        vehicle_state=VehicleState(
            speed=0.0,
            yaw_rate=0.0,
            heading=0.0,
        ),
        timestamp=timestamp,
        frame_id=frame_id,
        semantic_labels=semantic_labels,
        dynamic_probability=dynamic_probability,
    )


def test_default_config_is_valid() -> None:
    config = SignalIntegrationConfig()

    assert config.use_temporal_motion is True
    assert config.prefer_supplied_dynamic_probability is True
    assert config.compensate_ego_motion is True


def test_config_rejects_wrong_semantic_config() -> None:
    with pytest.raises(TypeError):
        SignalIntegrationConfig(
            semantic_config="invalid",  # type: ignore[arg-type]
        )


def test_config_rejects_wrong_temporal_config() -> None:
    with pytest.raises(TypeError):
        SignalIntegrationConfig(
            temporal_config="invalid",  # type: ignore[arg-type]
        )


def test_config_rejects_non_boolean_options() -> None:
    with pytest.raises(TypeError):
        SignalIntegrationConfig(
            use_temporal_motion=1,  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError):
        SignalIntegrationConfig(
            prefer_supplied_dynamic_probability=1,  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError):
        SignalIntegrationConfig(
            compensate_ego_motion=1,  # type: ignore[arg-type]
        )


def test_semantic_signal_returns_none_when_unavailable() -> None:
    frame = make_frame(
        xyz=np.zeros((3, 3)),
        timestamp=1.0,
        frame_id=1,
    )

    assert semantic_signal(frame) is None


def test_semantic_signal_converts_labels_to_importance() -> None:
    frame = make_frame(
        xyz=np.zeros((6, 3)),
        timestamp=1.0,
        frame_id=1,
        semantic_labels=np.array(
            [0, 1, 2, 3, 4, 5],
            dtype=np.int64,
        ),
    )

    result = semantic_signal(frame)

    assert result is not None

    np.testing.assert_allclose(
        result,
        np.array(
            [0.35, 0.00, 0.40, 0.75, 0.85, 1.00],
        ),
    )


def test_semantic_signal_preserves_point_count() -> None:
    labels = np.array(
        [1, 2, 3, 4, 5],
        dtype=np.int64,
    )

    frame = make_frame(
        xyz=np.zeros((5, 3)),
        timestamp=1.0,
        frame_id=1,
        semantic_labels=labels,
    )

    result = semantic_signal(frame)

    assert result is not None
    assert result.shape == (5,)


def test_semantic_signal_accepts_custom_config() -> None:
    frame = make_frame(
        xyz=np.zeros((2, 3)),
        timestamp=1.0,
        frame_id=1,
        semantic_labels=np.array(
            [4, 5],
            dtype=np.int64,
        ),
    )

    config = SemanticFoveationConfig(
        vehicle_importance=0.60,
        vulnerable_user_importance=0.90,
    )

    result = semantic_signal(
        frame,
        config=config,
    )

    assert result is not None

    np.testing.assert_allclose(
        result,
        np.array([0.60, 0.90]),
    )


def test_dynamic_signal_returns_none_without_dynamic_information() -> None:
    frame = make_frame(
        xyz=np.zeros((3, 3)),
        timestamp=1.0,
        frame_id=1,
    )

    result = dynamic_signal(frame)

    assert result is None


def test_dynamic_signal_uses_explicit_probability() -> None:
    probability = np.array(
        [0.0, 0.25, 0.80],
        dtype=np.float64,
    )

    frame = make_frame(
        xyz=np.zeros((3, 3)),
        timestamp=1.0,
        frame_id=1,
        dynamic_probability=probability,
    )

    result = dynamic_signal(frame)

    assert result is not None
    np.testing.assert_array_equal(
        result,
        probability,
    )


def test_explicit_dynamic_probability_is_preferred_by_default() -> None:
    previous = make_frame(
        xyz=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ),
        timestamp=1.0,
        frame_id=1,
    )

    current = make_frame(
        xyz=np.array(
            [
                [10.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],
            ]
        ),
        timestamp=2.0,
        frame_id=2,
        dynamic_probability=np.array(
            [0.20, 0.30],
        ),
    )

    result = dynamic_signal(
        current_frame=current,
        previous_frame=previous,
    )

    assert result is not None

    np.testing.assert_allclose(
        result,
        np.array([0.20, 0.30]),
    )


def test_temporal_dynamic_signal_is_used_when_no_explicit_probability() -> None:
    previous = make_frame(
        xyz=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ),
        timestamp=1.0,
        frame_id=1,
    )

    current = make_frame(
        xyz=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ),
        timestamp=2.0,
        frame_id=2,
    )

    result = dynamic_signal(
        current_frame=current,
        previous_frame=previous,
    )

    assert result is not None
    assert result.shape == (2,)
    np.testing.assert_allclose(
        result,
        np.zeros(2),
    )


def test_temporal_motion_detects_displacement() -> None:
    previous = make_frame(
        xyz=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ),
        timestamp=1.0,
        frame_id=1,
    )

    current = make_frame(
        xyz=np.array(
            [
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        ),
        timestamp=2.0,
        frame_id=2,
    )

    result = dynamic_signal(
        current_frame=current,
        previous_frame=previous,
    )

    assert result is not None
    assert result.shape == (2,)
    assert np.all(result > 0.0)


def test_temporal_motion_can_be_disabled() -> None:
    previous = make_frame(
        xyz=np.zeros((2, 3)),
        timestamp=1.0,
        frame_id=1,
    )

    current = make_frame(
        xyz=np.ones((2, 3)),
        timestamp=2.0,
        frame_id=2,
    )

    config = SignalIntegrationConfig(
        use_temporal_motion=False,
    )

    result = dynamic_signal(
        current_frame=current,
        previous_frame=previous,
        config=config,
    )

    assert result is None


def test_temporal_motion_rejects_nonchronological_frames() -> None:
    previous = make_frame(
        xyz=np.zeros((1, 3)),
        timestamp=2.0,
        frame_id=1,
    )

    current = make_frame(
        xyz=np.zeros((1, 3)),
        timestamp=1.0,
        frame_id=2,
    )

    with pytest.raises(ValueError):
        dynamic_signal(
            current_frame=current,
            previous_frame=previous,
        )


def test_ego_motion_compensation_removes_pure_ego_translation() -> None:
    previous = make_frame(
        xyz=np.array(
            [
                [10.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
            ]
        ),
        timestamp=1.0,
        frame_id=1,
    )

    current = make_frame(
        xyz=np.array(
            [
                [11.0, 0.0, 0.0],
                [21.0, 0.0, 0.0],
            ]
        ),
        timestamp=2.0,
        frame_id=2,
    )

    ego_motion = EgoMotion(
        translation_x=1.0,
        translation_y=0.0,
        translation_z=0.0,
        yaw=0.0,
    )

    result = dynamic_signal(
        current_frame=current,
        previous_frame=previous,
        ego_motion=ego_motion,
    )

    assert result is not None
    np.testing.assert_allclose(
        result,
        np.zeros(2),
    )


def test_ego_motion_can_be_disabled() -> None:
    previous = make_frame(
        xyz=np.array(
            [
                [10.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
            ]
        ),
        timestamp=1.0,
        frame_id=1,
    )

    current = make_frame(
        xyz=np.array(
            [
                [11.0, 0.0, 0.0],
                [21.0, 0.0, 0.0],
            ]
        ),
        timestamp=2.0,
        frame_id=2,
    )

    ego_motion = EgoMotion(
        translation_x=1.0,
    )

    config = SignalIntegrationConfig(
        compensate_ego_motion=False,
    )

    result = dynamic_signal(
        current_frame=current,
        previous_frame=previous,
        ego_motion=ego_motion,
        config=config,
    )

    assert result is not None
    assert np.all(result >= 0.0)
    assert np.any(result > 0.0)


def test_invalid_ego_motion_is_rejected() -> None:
    frame = make_frame(
        xyz=np.zeros((1, 3)),
        timestamp=1.0,
        frame_id=1,
    )

    with pytest.raises(TypeError):
        dynamic_signal(
            current_frame=frame,
            ego_motion="invalid",  # type: ignore[arg-type]
        )


def test_integrated_signals_contains_semantics() -> None:
    frame = make_frame(
        xyz=np.zeros((3, 3)),
        timestamp=1.0,
        frame_id=1,
        semantic_labels=np.array(
            [1, 4, 5],
            dtype=np.int64,
        ),
    )

    signals = build_integrated_signals(frame)

    assert set(signals) == {"semantic"}

    np.testing.assert_allclose(
        signals["semantic"],
        np.array([0.0, 0.85, 1.0]),
    )


def test_integrated_signals_contains_explicit_dynamic_probability() -> None:
    frame = make_frame(
        xyz=np.zeros((3, 3)),
        timestamp=1.0,
        frame_id=1,
        dynamic_probability=np.array(
            [0.10, 0.40, 0.90],
        ),
    )

    signals = build_integrated_signals(frame)

    assert set(signals) == {"dynamic"}

    np.testing.assert_allclose(
        signals["dynamic"],
        np.array([0.10, 0.40, 0.90]),
    )


def test_integrated_signals_contains_semantic_and_dynamic() -> None:
    frame = make_frame(
        xyz=np.zeros((3, 3)),
        timestamp=1.0,
        frame_id=1,
        semantic_labels=np.array(
            [1, 4, 5],
            dtype=np.int64,
        ),
        dynamic_probability=np.array(
            [0.1, 0.4, 0.9],
        ),
    )

    signals = build_integrated_signals(frame)

    assert set(signals) == {
        "semantic",
        "dynamic",
    }

    np.testing.assert_allclose(
        signals["semantic"],
        np.array([0.0, 0.85, 1.0]),
    )

    np.testing.assert_allclose(
        signals["dynamic"],
        np.array([0.1, 0.4, 0.9]),
    )


def test_integrated_signals_do_not_fabricate_missing_information() -> None:
    frame = make_frame(
        xyz=np.zeros((4, 3)),
        timestamp=1.0,
        frame_id=1,
    )

    signals = build_integrated_signals(frame)

    assert signals == {}


def test_integrated_signals_use_temporal_dynamic_signal() -> None:
    previous = make_frame(
        xyz=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ),
        timestamp=1.0,
        frame_id=1,
    )

    current = make_frame(
        xyz=np.array(
            [
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        ),
        timestamp=2.0,
        frame_id=2,
    )

    signals = build_integrated_signals(
        current_frame=current,
        previous_frame=previous,
    )

    assert set(signals) == {"dynamic"}
    assert np.all(signals["dynamic"] > 0.0)


def test_integrated_signals_combine_semantic_and_temporal_dynamic() -> None:
    previous = make_frame(
        xyz=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ),
        timestamp=1.0,
        frame_id=1,
    )

    current = make_frame(
        xyz=np.array(
            [
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        ),
        timestamp=2.0,
        frame_id=2,
        semantic_labels=np.array(
            [4, 5],
            dtype=np.int64,
        ),
    )

    signals = build_integrated_signals(
        current_frame=current,
        previous_frame=previous,
    )

    assert set(signals) == {
        "semantic",
        "dynamic",
    }

    assert np.all(signals["semantic"] >= 0.0)
    assert np.all(signals["dynamic"] >= 0.0)


def test_integrated_signals_preserve_current_point_count() -> None:
    rng = np.random.default_rng(30)

    previous = make_frame(
        xyz=rng.normal(size=(7, 3)),
        timestamp=1.0,
        frame_id=1,
    )

    current = make_frame(
        xyz=rng.normal(size=(11, 3)),
        timestamp=2.0,
        frame_id=2,
        semantic_labels=np.arange(11) % 6,
    )

    signals = build_integrated_signals(
        current_frame=current,
        previous_frame=previous,
    )

    assert signals["semantic"].shape == (11,)
    assert signals["dynamic"].shape == (11,)


def test_integrated_signals_accept_custom_temporal_config() -> None:
    previous = make_frame(
        xyz=np.array([[0.0, 0.0, 0.0]]),
        timestamp=1.0,
        frame_id=1,
    )

    current = make_frame(
        xyz=np.array([[0.6, 0.0, 0.0]]),
        timestamp=2.0,
        frame_id=2,
    )

    config = SignalIntegrationConfig(
        temporal_config=TemporalMotionConfig(
            motion_threshold=0.1,
            max_correspondence_distance=1.0,
        ),
    )

    signals = build_integrated_signals(
        current_frame=current,
        previous_frame=previous,
        config=config,
    )

    assert "dynamic" in signals
    assert signals["dynamic"][0] > 0.0


def test_integrate_signal_provider_uses_frame_annotations() -> None:
    provider = integrate_signal_provider()

    frame = make_frame(
        xyz=np.zeros((2, 3)),
        timestamp=1.0,
        frame_id=1,
        semantic_labels=np.array(
            [4, 5],
            dtype=np.int64,
        ),
    )

    signals = provider(frame)

    assert set(signals) == {"semantic"}

    np.testing.assert_allclose(
        signals["semantic"],
        np.array([0.85, 1.0]),
    )


def test_non_sensor_frame_is_rejected() -> None:
    with pytest.raises(TypeError):
        build_integrated_signals(
            current_frame="invalid",  # type: ignore[arg-type]
        )


def test_previous_non_sensor_frame_is_rejected() -> None:
    frame = make_frame(
        xyz=np.zeros((1, 3)),
        timestamp=1.0,
        frame_id=1,
    )

    with pytest.raises(TypeError):
        build_integrated_signals(
            current_frame=frame,
            previous_frame="invalid",  # type: ignore[arg-type]
        )


def test_temporal_config_is_used_for_probability_bounds() -> None:
    previous = make_frame(
        xyz=np.array([[0.0, 0.0, 0.0]]),
        timestamp=1.0,
        frame_id=1,
    )

    current = make_frame(
        xyz=np.array([[0.0, 0.0, 0.0]]),
        timestamp=2.0,
        frame_id=2,
    )

    config = SignalIntegrationConfig(
        temporal_config=TemporalMotionConfig(
            min_probability=0.4,
        ),
    )

    signals = build_integrated_signals(
        current_frame=current,
        previous_frame=previous,
        config=config,
    )

    assert "dynamic" in signals
    np.testing.assert_allclose(
        signals["dynamic"],
        np.array([0.4]),
    )