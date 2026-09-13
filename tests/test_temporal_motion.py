"""
A25 — Temporal Motion Estimation tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from mapping.kinematic_foveation import VehicleState
from mapping.sensor_frame import SensorFrame
from mapping.temporal_motion import (
    TemporalMotionConfig,
    temporal_dynamic_probability,
    temporal_motion_evidence,
    temporal_motion_signal,
)


def make_frame(
    xyz: np.ndarray,
    timestamp: float,
    frame_id: int,
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
    )


def test_config_defaults_are_valid() -> None:
    config = TemporalMotionConfig()

    assert config.voxel_size > 0.0
    assert config.motion_threshold >= 0.0
    assert config.max_correspondence_distance > 0.0
    assert 0.0 <= config.min_probability <= 1.0


def test_config_rejects_invalid_voxel_size() -> None:
    with pytest.raises(ValueError):
        TemporalMotionConfig(voxel_size=0.0)


def test_config_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError):
        TemporalMotionConfig(motion_threshold=-1.0)


def test_config_rejects_invalid_max_distance() -> None:
    with pytest.raises(ValueError):
        TemporalMotionConfig(max_correspondence_distance=0.0)


def test_config_rejects_invalid_probability() -> None:
    with pytest.raises(ValueError):
        TemporalMotionConfig(min_probability=1.5)


def test_identical_frames_have_zero_motion_evidence() -> None:
    xyz = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, -1.0, 0.2],
        ]
    )

    evidence = temporal_motion_evidence(
        xyz,
        xyz.copy(),
    )

    assert evidence.shape == (3,)
    assert np.allclose(evidence, 0.0)


def test_small_displacement_has_low_evidence() -> None:
    previous = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )

    current = previous + np.array([0.05, 0.0, 0.0])

    evidence = temporal_motion_evidence(
        previous,
        current,
    )

    assert np.all(evidence < 0.5)


def test_large_displacement_has_high_evidence() -> None:
    previous = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )

    current = previous + np.array([2.0, 0.0, 0.0])

    evidence = temporal_motion_evidence(
        previous,
        current,
    )

    assert np.all(evidence > 0.0)
    assert np.all(evidence <= 1.0)


def test_empty_current_frame_returns_empty() -> None:
    previous = np.array([[1.0, 0.0, 0.0]])
    current = np.empty((0, 3))

    evidence = temporal_motion_evidence(
        previous,
        current,
    )

    assert evidence.shape == (0,)


def test_empty_previous_frame_marks_current_points_dynamic() -> None:
    previous = np.empty((0, 3))
    current = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )

    evidence = temporal_motion_evidence(
        previous,
        current,
    )

    assert np.allclose(evidence, 1.0)


def test_evidence_is_bounded() -> None:
    rng = np.random.default_rng(25)

    previous = rng.normal(size=(50, 3))
    current = rng.normal(size=(75, 3))

    evidence = temporal_motion_evidence(
        previous,
        current,
    )

    assert evidence.shape == (75,)
    assert np.all(evidence >= 0.0)
    assert np.all(evidence <= 1.0)


def test_evidence_is_deterministic() -> None:
    rng = np.random.default_rng(123)

    previous = rng.normal(size=(40, 3))
    current = rng.normal(size=(60, 3))

    first = temporal_motion_evidence(
        previous,
        current,
    )

    second = temporal_motion_evidence(
        previous,
        current,
    )

    assert np.array_equal(first, second)


def test_nan_points_are_rejected() -> None:
    previous = np.array(
        [
            [1.0, 0.0, 0.0],
            [np.nan, 1.0, 0.0],
        ]
    )

    current = np.array([[1.0, 0.0, 0.0]])

    with pytest.raises(ValueError):
        temporal_motion_evidence(
            previous,
            current,
        )


def test_invalid_shape_is_rejected() -> None:
    with pytest.raises(ValueError):
        temporal_motion_evidence(
            np.zeros((3, 2)),
            np.zeros((3, 3)),
        )


def test_temporal_dynamic_probability_uses_frames() -> None:
    previous = make_frame(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        ),
        timestamp=1.0,
        frame_id=1,
    )

    current = make_frame(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        ),
        timestamp=1.1,
        frame_id=2,
    )

    probability = temporal_dynamic_probability(
        previous,
        current,
    )

    assert probability.shape == (2,)
    assert np.allclose(probability, 0.0)


def test_temporal_dynamic_probability_rejects_nonchronological_frames() -> None:
    previous = make_frame(
        np.zeros((1, 3)),
        timestamp=2.0,
        frame_id=1,
    )

    current = make_frame(
        np.zeros((1, 3)),
        timestamp=1.0,
        frame_id=2,
    )

    with pytest.raises(ValueError):
        temporal_dynamic_probability(
            previous,
            current,
        )


def test_temporal_dynamic_probability_rejects_invalid_frame_types() -> None:
    frame = make_frame(
        np.zeros((1, 3)),
        timestamp=1.0,
        frame_id=1,
    )

    with pytest.raises(TypeError):
        temporal_dynamic_probability(
            frame,
            object(),  # type: ignore[arg-type]
        )


def test_temporal_signal_uses_dynamic_namespace() -> None:
    previous = make_frame(
        np.array([[1.0, 0.0, 0.0]]),
        timestamp=1.0,
        frame_id=1,
    )

    current = make_frame(
        np.array([[1.0, 0.0, 0.0]]),
        timestamp=1.1,
        frame_id=2,
    )

    signals = temporal_motion_signal(
        previous,
        current,
    )

    assert set(signals) == {"dynamic"}
    assert signals["dynamic"].shape == (1,)
    assert np.allclose(signals["dynamic"], 0.0)


def test_custom_threshold_changes_evidence() -> None:
    previous = np.array([[0.0, 0.0, 0.0]])
    current = np.array([[0.75, 0.0, 0.0]])

    config = TemporalMotionConfig(
        voxel_size=0.10,
        motion_threshold=0.25,
        max_correspondence_distance=1.25,
    )

    evidence = temporal_motion_evidence(
        previous,
        current,
        config,
    )

    assert evidence[0] > 0.0


def test_min_probability_is_respected() -> None:
    previous = np.array([[0.0, 0.0, 0.0]])
    current = np.array([[0.0, 0.0, 0.0]])

    config = TemporalMotionConfig(
        min_probability=0.25,
    )

    evidence = temporal_motion_evidence(
        previous,
        current,
        config,
    )

    assert np.all(evidence >= 0.25)