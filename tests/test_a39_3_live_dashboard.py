"""
Tests for A39.3 live FoveaMap demonstration.
"""

from __future__ import annotations

import numpy as np
import pytest

from mapping.foveamap_dashboard import (
    DashboardConfig,
    DashboardFrame,
)
from mapping.safety_controller import (
    SafetyConfig,
)
from mapping.signal_integration import (
    SignalIntegrationConfig,
)
from mapping.temporal_foveamap_pipeline import (
    TemporalFoveaMapProcessor,
)

from scripts.run_foveamap_dashboard import (
    DemoConfig,
    _base_signal_provider,
    _build_safety_config,
    _make_sensor_frame,
    _phase_for_frame,
    process_demo_frame,
    run_demo,
)


def test_default_demo_config():
    config = DemoConfig()

    assert config.total_frames == 60
    assert config.frame_rate_hz == 10.0

    assert config.normal_points > 2_000

    assert (
        500
        < config.degraded_points
        < 2_000
    )

    assert config.safety_points <= 500


def test_phase_sequence():
    config = DemoConfig()

    assert (
        _phase_for_frame(
            0,
            config,
        )
        == "NORMAL"
    )

    assert (
        _phase_for_frame(
            19,
            config,
        )
        == "NORMAL"
    )

    assert (
        _phase_for_frame(
            20,
            config,
        )
        == "DEGRADED"
    )

    assert (
        _phase_for_frame(
            39,
            config,
        )
        == "DEGRADED"
    )

    assert (
        _phase_for_frame(
            40,
            config,
        )
        == "SAFETY"
    )

    assert (
        _phase_for_frame(
            49,
            config,
        )
        == "SAFETY"
    )

    assert (
        _phase_for_frame(
            50,
            config,
        )
        == "NORMAL"
    )


def test_demo_frame_has_canonical_sensor_data():
    rng = np.random.default_rng(
        39
    )

    config = DemoConfig()

    frame = _make_sensor_frame(
        frame_index=1,
        timestamp=0.1,
        rng=rng,
        config=config,
    )

    assert (
        frame.num_points
        == config.normal_points
    )

    assert (
        frame.semantic_labels
        is not None
    )

    assert (
        frame.dynamic_probability
        is not None
    )

    assert frame.semantic_labels.shape == (
        frame.num_points,
    )

    assert frame.dynamic_probability.shape == (
        frame.num_points,
    )

    assert np.all(
        np.isin(
            frame.semantic_labels,
            [
                1,
                2,
                3,
                4,
                5,
            ],
        )
    )

    assert np.all(
        frame.dynamic_probability
        >= 0.0
    )

    assert np.all(
        frame.dynamic_probability
        <= 1.0
    )


def test_first_demo_frame_has_no_explicit_dynamic_probability():
    rng = np.random.default_rng(
        39
    )

    config = DemoConfig()

    frame = _make_sensor_frame(
        frame_index=0,
        timestamp=0.0,
        rng=rng,
        config=config,
    )

    assert (
        frame.semantic_labels
        is not None
    )

    assert (
        frame.dynamic_probability
        is None
    )


def test_base_signal_provider_returns_valid_signals():
    rng = np.random.default_rng(
        39
    )

    config = DemoConfig()

    frame = _make_sensor_frame(
        frame_index=1,
        timestamp=0.1,
        rng=rng,
        config=config,
    )

    signals = _base_signal_provider(
        frame
    )

    assert set(signals) == {
        "DISTANCE",
        "KINEMATIC",
    }

    for values in signals.values():
        assert values.shape == (
            frame.num_points,
        )

        assert np.all(
            np.isfinite(values)
        )

        assert np.all(
            values >= 0.0
        )

        assert np.all(
            values <= 1.0
        )


def test_safety_config_preserves_demo_thresholds():
    config = _build_safety_config()

    assert isinstance(
        config,
        SafetyConfig,
    )

    assert (
        config.min_points_degraded
        == 2_000
    )

    assert (
        config.min_points_safety
        == 500
    )

    assert (
        config.max_latency_ms_degraded
        > 1_000.0
    )

    assert (
        config.max_latency_ms_safety
        > config.max_latency_ms_degraded
    )


def test_process_demo_frame_runs_real_temporal_pipeline():
    rng = np.random.default_rng(
        39
    )

    config = DemoConfig()

    processor = TemporalFoveaMapProcessor(
        _base_signal_provider,
        signal_config=SignalIntegrationConfig(
            prefer_supplied_dynamic_probability=True,
            compensate_ego_motion=True,
        ),
    )

    frame = _make_sensor_frame(
        frame_index=0,
        timestamp=0.0,
        rng=rng,
        config=config,
    )

    (
        dashboard_frame,
        result,
    ) = process_demo_frame(
        processor=processor,
        frame=frame,
        previous_timestamp=None,
        safety_config=_build_safety_config(),
        dashboard_config=DashboardConfig(),
    )

    assert isinstance(
        dashboard_frame,
        DashboardFrame,
    )

    assert result.frame_id == 0

    assert (
        "DISTANCE"
        in result.signals
    )

    assert (
        "KINEMATIC"
        in result.signals
    )

    assert (
        "semantic"
        in result.signals
    )

    assert (
        result.pipeline_result.foveation
    )

    assert (
        result.pipeline_result.leaf_map
        is not None
    )

    assert (
        dashboard_frame.point_count
        == frame.num_points
    )

    assert (
        dashboard_frame.semantic_available
        is True
    )

    assert (
        dashboard_frame.processing_latency_ms
        >= 0.0
    )


def test_second_frame_gets_dynamic_signal():
    rng = np.random.default_rng(
        39
    )

    config = DemoConfig()

    processor = TemporalFoveaMapProcessor(
        _base_signal_provider,
        signal_config=SignalIntegrationConfig(
            prefer_supplied_dynamic_probability=True,
            compensate_ego_motion=True,
        ),
    )

    first = _make_sensor_frame(
        frame_index=0,
        timestamp=0.0,
        rng=rng,
        config=config,
    )

    second = _make_sensor_frame(
        frame_index=1,
        timestamp=0.1,
        rng=rng,
        config=config,
    )

    processor.process_frame(
        first
    )

    result = processor.process_frame(
        second
    )

    assert (
        "dynamic"
        in result.signals
    )

    assert result.signals[
        "dynamic"
    ].shape == (
        second.num_points,
    )


def test_run_demo_headless_produces_dashboard_frames():
    config = DemoConfig(
        total_frames=3,
        frame_rate_hz=100.0,
    )

    frames = run_demo(
        config=config,
        show=False,
    )

    assert len(frames) == 3

    assert all(
        isinstance(
            frame,
            DashboardFrame,
        )
        for frame in frames
    )

    assert (
        frames[0].frame_id
        == 0
    )

    assert (
        frames[-1].frame_id
        == 2
    )


def test_run_demo_exercises_safety_modes():
    config = DemoConfig(
        total_frames=3,
        frame_rate_hz=100.0,
        safety_degraded_start=1,
        safety_critical_start=2,
        safety_recovery_start=3,
    )

    frames = run_demo(
        config=config,
        show=False,
    )

    assert (
        frames[0].safety_mode
        == "NORMAL"
    )

    assert (
        frames[1].safety_mode
        == "DEGRADED"
    )

    assert (
        frames[2].safety_mode
        == "SAFETY"
    )


@pytest.mark.parametrize(
    "frames",
    [
        0,
        -1,
    ],
)
def test_invalid_frame_count(
    frames: int,
):
    with pytest.raises(ValueError):
        DemoConfig(
            total_frames=frames
        )


def test_invalid_demo_rate():
    with pytest.raises(ValueError):
        DemoConfig(
            frame_rate_hz=0.0
        )


def test_invalid_point_count():
    with pytest.raises(ValueError):
        DemoConfig(
            normal_points=0
        )


def test_invalid_degraded_point_count():
    with pytest.raises(ValueError):
        DemoConfig(
            degraded_points=0
        )


def test_invalid_safety_point_count():
    with pytest.raises(ValueError):
        DemoConfig(
            safety_points=0
        )


def test_invalid_phase_order():
    with pytest.raises(ValueError):
        DemoConfig(
            safety_normal_start=20,
            safety_degraded_start=10,
        )


def test_deterministic_scene_generation():
    config = DemoConfig()

    rng_a = np.random.default_rng(
        config.random_seed
    )

    rng_b = np.random.default_rng(
        config.random_seed
    )

    frame_a = _make_sensor_frame(
        frame_index=1,
        timestamp=0.1,
        rng=rng_a,
        config=config,
    )

    frame_b = _make_sensor_frame(
        frame_index=1,
        timestamp=0.1,
        rng=rng_b,
        config=config,
    )

    assert np.array_equal(
        frame_a.xyz,
        frame_b.xyz,
    )

    assert np.array_equal(
        frame_a.semantic_labels,
        frame_b.semantic_labels,
    )

    assert np.array_equal(
        frame_a.dynamic_probability,
        frame_b.dynamic_probability,
    )


def test_recovery_returns_to_normal():
    config = DemoConfig(
        total_frames=4,
        frame_rate_hz=100.0,
        safety_degraded_start=1,
        safety_critical_start=2,
        safety_recovery_start=3,
    )

    frames = run_demo(
        config=config,
        show=False,
    )

    assert [
        frame.safety_mode
        for frame in frames
    ] == [
        "NORMAL",
        "DEGRADED",
        "SAFETY",
        "NORMAL",
    ]