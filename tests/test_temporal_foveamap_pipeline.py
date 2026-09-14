from __future__ import annotations

import numpy as np
import pytest

from mapping.ego_motion import EgoMotion
from mapping.sensor_frame import SensorFrame, VehicleState
from mapping.signal_integration import SignalIntegrationConfig
from mapping.temporal_foveamap_pipeline import (
    TemporalFoveaMapProcessor,
    process_temporal_foveamap_sequence,
)
from mapping.production_foveation_signals import (
    ProductionFoveationSignalConfig,
)


def make_frame(
    frame_id: str,
    timestamp: float,
    *,
    semantic_labels: np.ndarray | None = None,
    dynamic_probability: np.ndarray | None = None,
) -> SensorFrame:
    xyz = np.array(
        [
            [2.0, 0.0, 0.0],
            [5.0, 1.0, 0.2],
            [10.0, -1.0, 0.5],
            [20.0, 0.0, 1.0],
            [35.0, 2.0, 0.3],
            [50.0, -2.0, 0.8],
        ],
        dtype=np.float64,
    )

    return SensorFrame(
        xyz=xyz,
        vehicle_state=VehicleState(
            speed=8.0,
            yaw_rate=0.05,
            heading=0.0,
        ),
        timestamp=timestamp,
        frame_id=frame_id,
        semantic_labels=semantic_labels,
        dynamic_probability=dynamic_probability,
    )


def base_signal_provider(frame: SensorFrame) -> dict[str, np.ndarray]:
    distance = np.linalg.norm(
        frame.xyz[:, :2],
        axis=1,
    )

    return {
        "DISTANCE": 1.0 / (1.0 + distance),
    }


def test_processor_uses_native_production_provider_by_default():
    frame = make_frame("000000", 0.0)

    processor = TemporalFoveaMapProcessor()

    result = processor.process_frame(frame)

    assert set(result.signals) == {
        "DISTANCE",
        "KINEMATIC",
        "PREDICTED_PATH",
    }

    for signal in result.signals.values():
        assert signal.shape == (frame.num_points,)
        assert np.all(np.isfinite(signal))
        assert np.all(signal >= 0.0)
        assert np.all(signal <= 1.0)


def test_default_production_provider_reaches_a20():
    frame = make_frame("000000", 0.0)

    processor = TemporalFoveaMapProcessor()

    result = processor.process_frame(frame)

    assert result.pipeline_result.foveation
    assert result.pipeline_result.leaf_map is not None

    assert result.signals["DISTANCE"].shape == (
        frame.num_points,
    )
    assert result.signals["KINEMATIC"].shape == (
        frame.num_points,
    )
    assert result.signals["PREDICTED_PATH"].shape == (
        frame.num_points,
    )


def test_custom_base_signal_provider_overrides_production_provider():
    frame = make_frame("000000", 0.0)

    processor = TemporalFoveaMapProcessor(
        base_signal_provider,
    )

    result = processor.process_frame(frame)

    assert set(result.signals) == {"DISTANCE"}

    expected = base_signal_provider(frame)["DISTANCE"]

    np.testing.assert_allclose(
        result.signals["DISTANCE"],
        expected,
    )


def test_single_frame_runs_full_a20_pipeline():
    frame = make_frame("000000", 0.0)

    processor = TemporalFoveaMapProcessor(
        base_signal_provider,
    )

    result = processor.process_frame(frame)

    assert result.frame_id == "000000"
    assert result.timestamp == 0.0
    assert result.delta_time is None
    assert result.ego_motion is None

    assert "DISTANCE" in result.signals
    assert "semantic" not in result.signals
    assert "dynamic" not in result.signals

    assert result.pipeline_result.foveation
    assert result.pipeline_result.leaf_map is not None


def test_timestamps_must_be_strictly_increasing():
    processor = TemporalFoveaMapProcessor(
        base_signal_provider,
    )

    processor.process_frame(
        make_frame("000000", 1.0),
    )

    with pytest.raises(ValueError):
        processor.process_frame(
            make_frame("000001", 1.0),
        )


def test_semantic_signal_propagates():
    labels = np.array(
        [3, 3, 4, 4, 5, 5],
        dtype=np.int64,
    )

    frame = make_frame(
        "000000",
        0.0,
        semantic_labels=labels,
    )

    processor = TemporalFoveaMapProcessor(
        base_signal_provider,
    )

    result = processor.process_frame(frame)

    assert "semantic" in result.signals
    assert result.signals["semantic"].shape == (
        frame.num_points,
    )


def test_missing_semantic_is_not_fabricated():
    frame = make_frame("000000", 0.0)

    processor = TemporalFoveaMapProcessor(
        base_signal_provider,
    )

    result = processor.process_frame(frame)

    assert "semantic" not in result.signals


def test_explicit_dynamic_probability_is_used():
    dynamic = np.array(
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        dtype=np.float64,
    )

    frame = make_frame(
        "000000",
        0.0,
        dynamic_probability=dynamic,
    )

    processor = TemporalFoveaMapProcessor(
        base_signal_provider,
    )

    result = processor.process_frame(frame)

    assert "dynamic" in result.signals

    np.testing.assert_allclose(
        result.signals["dynamic"],
        dynamic,
    )


def test_temporal_dynamic_signal_is_created_on_second_frame():
    frame1 = make_frame("000000", 0.0)
    frame2 = make_frame("000001", 1.0)

    processor = TemporalFoveaMapProcessor(
        base_signal_provider,
    )

    first = processor.process_frame(frame1)
    second = processor.process_frame(frame2)

    assert "dynamic" not in first.signals
    assert "dynamic" in second.signals

    assert second.signals["dynamic"].shape == (
        frame2.num_points,
    )


def test_explicit_dynamic_overrides_temporal_dynamic():
    frame1 = make_frame("000000", 0.0)

    explicit_dynamic = np.array(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        dtype=np.float64,
    )

    frame2 = make_frame(
        "000001",
        1.0,
        dynamic_probability=explicit_dynamic,
    )

    processor = TemporalFoveaMapProcessor(
        base_signal_provider,
    )

    processor.process_frame(frame1)
    second = processor.process_frame(frame2)

    assert "dynamic" in second.signals

    np.testing.assert_allclose(
        second.signals["dynamic"],
        explicit_dynamic,
    )


def test_ego_motion_provider_is_not_called_for_first_frame():
    calls = []

    def ego_motion_provider(
        current: SensorFrame,
        previous: SensorFrame,
    ) -> EgoMotion:
        calls.append(
            (
                current.frame_id,
                previous.frame_id,
            )
        )
        return EgoMotion(
            translation_x=0.0,
            translation_y=0.0,
            translation_z=0.0,
            yaw=0.0,
        )

    processor = TemporalFoveaMapProcessor(
        base_signal_provider,
        ego_motion_provider=ego_motion_provider,
    )

    processor.process_frame(
        make_frame("000000", 0.0),
    )

    assert calls == []


def test_ego_motion_provider_is_called_after_first_frame():
    calls = []

    def ego_motion_provider(
        current: SensorFrame,
        previous: SensorFrame,
    ) -> EgoMotion:
        calls.append(
            (
                current.frame_id,
                previous.frame_id,
            )
        )

        return EgoMotion(
            translation_x=0.0,
            translation_y=0.0,
            translation_z=0.0,
            yaw=0.0,
        )

    processor = TemporalFoveaMapProcessor(
        base_signal_provider,
        ego_motion_provider=ego_motion_provider,
    )

    processor.process_frame(
        make_frame("000000", 0.0),
    )

    processor.process_frame(
        make_frame("000001", 1.0),
    )

    assert calls == [
        ("000001", "000000"),
    ]


def test_sequence_result_contains_statistics():
    frames = [
        make_frame("000000", 0.0),
        make_frame("000001", 1.0),
        make_frame("000002", 2.0),
    ]

    processor = TemporalFoveaMapProcessor(
        base_signal_provider,
    )

    result = processor.process_sequence(frames)

    assert result.total_frames == 3
    assert len(result.frames) == 3
    assert result.timestamps == (0.0, 1.0, 2.0)
    assert result.delta_times == (None, 1.0, 1.0)


def test_empty_sequence_returns_empty_result():
    processor = TemporalFoveaMapProcessor(
        base_signal_provider,
    )

    result = processor.process_sequence([])

    assert result.total_frames == 0
    assert result.frames == ()
    assert result.frame_ids == ()
    assert result.timestamps == ()
    assert result.delta_times == ()


def test_previous_frame_is_retained():
    processor = TemporalFoveaMapProcessor(
        base_signal_provider,
    )

    frame1 = make_frame("000000", 0.0)
    frame2 = make_frame("000001", 1.0)

    processor.process_frame(frame1)
    processor.process_frame(frame2)

    assert processor.previous_frame is frame2


def test_reset_clears_previous_frame():
    processor = TemporalFoveaMapProcessor(
        base_signal_provider,
    )

    frame = make_frame("000000", 0.0)

    processor.process_frame(frame)

    assert processor.previous_frame is frame

    processor.reset()

    assert processor.previous_frame is None


def test_base_signal_shape_is_validated():
    def invalid_provider(frame: SensorFrame):
        return {
            "DISTANCE": np.zeros(
                frame.num_points - 1,
                dtype=np.float64,
            )
        }

    processor = TemporalFoveaMapProcessor(
        invalid_provider,
    )

    with pytest.raises(ValueError):
        processor.process_frame(
            make_frame("000000", 0.0),
        )


def test_nonfinite_base_signal_is_rejected():
    def invalid_provider(frame: SensorFrame):
        values = np.ones(
            frame.num_points,
            dtype=np.float64,
        )
        values[0] = np.nan

        return {
            "DISTANCE": values,
        }

    processor = TemporalFoveaMapProcessor(
        invalid_provider,
    )

    with pytest.raises(ValueError):
        processor.process_frame(
            make_frame("000000", 0.0),
        )


def test_empty_custom_base_signals_are_rejected():
    def empty_provider(frame: SensorFrame):
        return {}

    processor = TemporalFoveaMapProcessor(
        empty_provider,
    )

    with pytest.raises(ValueError):
        processor.process_frame(
            make_frame("000000", 0.0),
        )


def test_point_count_is_preserved_through_pipeline():
    frame = make_frame("000000", 0.0)

    processor = TemporalFoveaMapProcessor(
        base_signal_provider,
    )

    result = processor.process_frame(frame)

    assert result.pipeline_result.leaf_map is not None
    assert result.pipeline_result.leaf_map.input_points == (
        frame.num_points
    )


def test_convenience_function_runs_sequence():
    frames = [
        make_frame("000000", 0.0),
        make_frame("000001", 1.0),
    ]

    result = process_temporal_foveamap_sequence(
        frames,
        base_signal_provider,
    )

    assert result.total_frames == 2
    assert len(result.frames) == 2


def test_convenience_function_uses_production_provider_by_default():
    frames = [
        make_frame("000000", 0.0),
        make_frame("000001", 1.0),
    ]

    result = process_temporal_foveamap_sequence(
        frames,
    )

    assert result.total_frames == 2

    for frame_result in result.frames:
        assert {
            "DISTANCE",
            "KINEMATIC",
            "PREDICTED_PATH",
        }.issubset(frame_result.signals)

        assert frame_result.pipeline_result.leaf_map is not None


def test_production_foveation_config_is_forwarded():
    config = ProductionFoveationSignalConfig(
        maximum_distance=50.0,
        path_horizon=4.0,
        path_step=0.5,
        path_corridor_width=2.0,
    )

    frame = make_frame("000000", 0.0)

    processor = TemporalFoveaMapProcessor(
        production_foveation_config=config,
    )

    result = processor.process_frame(frame)

    assert set(result.signals) == {
        "DISTANCE",
        "KINEMATIC",
        "PREDICTED_PATH",
    }

    expected_distance = np.clip(
        1.0
        - (
            np.linalg.norm(
                frame.xyz[:, :2],
                axis=1,
            )
            / 50.0
        ),
        0.0,
        1.0,
    )

    np.testing.assert_allclose(
        result.signals["DISTANCE"],
        expected_distance,
    )


def test_multiple_sequence_calls_retain_temporal_state():
    processor = TemporalFoveaMapProcessor(
        base_signal_provider,
    )

    first = processor.process_sequence(
        [
            make_frame("000000", 0.0),
            make_frame("000001", 1.0),
        ]
    )

    second = processor.process_sequence(
        [
            make_frame("000010", 10.0),
            make_frame("000011", 11.0),
        ]
    )

    assert first.total_frames == 2
    assert second.total_frames == 2

    assert second.frames[0].delta_time == 9.0
    assert second.frames[0].ego_motion is None
