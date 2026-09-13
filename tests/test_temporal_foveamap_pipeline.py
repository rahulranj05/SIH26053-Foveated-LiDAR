from __future__ import annotations

import numpy as np
import pytest

from mapping.ego_motion import EgoMotion
from mapping.kinematic_foveation import VehicleState
from mapping.sensor_frame import SensorFrame
from mapping.temporal_foveamap_pipeline import (
    TemporalFoveaMapProcessor,
    TemporalFoveaMapSequenceResult,
    process_temporal_foveamap_sequence,
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
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
            [4.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )

    return SensorFrame(
        xyz=xyz,
        vehicle_state=VehicleState(
            speed=5.0,
            yaw_rate=0.1,
            heading=0.0,
        ),
        timestamp=timestamp,
        frame_id=frame_id,
        semantic_labels=semantic_labels,
        dynamic_probability=dynamic_probability,
    )


def base_signal_provider(
    frame: SensorFrame,
) -> dict[str, np.ndarray]:
    """
    Minimal deterministic base signal used only for A31 tests.

    This represents a caller-supplied non-semantic/non-dynamic
    importance signal.
    """

    distance = np.linalg.norm(
        frame.xyz[:, :2],
        axis=1,
    )

    return {
        "DISTANCE": 1.0 / (1.0 + distance),
    }


def test_processor_requires_callable_base_signal_provider():
    with pytest.raises(TypeError):
        TemporalFoveaMapProcessor(None)  # type: ignore[arg-type]


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


def test_frame_id_and_timestamp_are_preserved():
    frames = [
        make_frame("000010", 10.0),
        make_frame("000011", 10.1),
    ]

    processor = TemporalFoveaMapProcessor(
        base_signal_provider,
    )

    first = processor.process_frame(frames[0])
    second = processor.process_frame(frames[1])

    assert first.frame_id == "000010"
    assert second.frame_id == "000011"

    assert first.timestamp == 10.0
    assert second.timestamp == 10.1

    assert second.delta_time == pytest.approx(0.1)


def test_timestamps_must_be_strictly_increasing():
    processor = TemporalFoveaMapProcessor(
        base_signal_provider,
    )

    processor.process_frame(
        make_frame("000000", 1.0)
    )

    with pytest.raises(ValueError):
        processor.process_frame(
            make_frame("000001", 1.0)
        )


def test_decreasing_timestamp_is_rejected():
    processor = TemporalFoveaMapProcessor(
        base_signal_provider,
    )

    processor.process_frame(
        make_frame("000000", 2.0)
    )

    with pytest.raises(ValueError):
        processor.process_frame(
            make_frame("000001", 1.0)
        )


def test_semantic_signal_is_propagated_when_present():
    labels = np.array(
        [1, 2, 3, 5],
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
    assert result.signals["semantic"].shape == (4,)


def test_missing_semantics_are_not_fabricated():
    frame = make_frame(
        "000000",
        0.0,
        semantic_labels=None,
    )

    processor = TemporalFoveaMapProcessor(
        base_signal_provider,
    )

    result = processor.process_frame(frame)

    assert "semantic" not in result.signals


def test_supplied_dynamic_probability_is_used():
    dynamic = np.array(
        [0.0, 0.25, 0.75, 1.0],
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


def test_temporal_dynamic_signal_appears_with_previous_frame():
    first = make_frame(
        "000000",
        0.0,
    )

    second = SensorFrame(
        xyz=np.array(
            [
                [1.8, 0.0, 0.0],
                [2.8, 0.0, 0.0],
                [3.8, 1.0, 0.0],
                [4.8, -1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        vehicle_state=VehicleState(
            speed=5.0,
            yaw_rate=0.1,
            heading=0.0,
        ),
        timestamp=1.0,
        frame_id="000001",
    )

    processor = TemporalFoveaMapProcessor(
        base_signal_provider,
    )

    first_result = processor.process_frame(first)
    second_result = processor.process_frame(second)

    assert "dynamic" not in first_result.signals
    assert "dynamic" in second_result.signals

    assert second_result.signals["dynamic"].shape == (
        second.num_points,
    )


def test_explicit_dynamic_probability_overrides_temporal_signal():
    first = make_frame(
        "000000",
        0.0,
    )

    explicit_dynamic = np.array(
        [0.1, 0.2, 0.3, 0.4],
        dtype=np.float64,
    )

    second = make_frame(
        "000001",
        1.0,
        dynamic_probability=explicit_dynamic,
    )

    processor = TemporalFoveaMapProcessor(
        base_signal_provider,
    )

    processor.process_frame(first)
    result = processor.process_frame(second)

    np.testing.assert_allclose(
        result.signals["dynamic"],
        explicit_dynamic,
    )


def test_ego_motion_provider_is_called_for_temporal_frame():
    first = make_frame(
        "000000",
        0.0,
    )

    second = make_frame(
        "000001",
        1.0,
    )

    calls: list[tuple[str, str]] = []

    def ego_provider(
        current: SensorFrame,
        previous: SensorFrame,
    ) -> EgoMotion:
        calls.append(
            (
                str(previous.frame_id),
                str(current.frame_id),
            )
        )

        return EgoMotion(
            translation_x=0.5,
            translation_y=0.0,
            translation_z=0.0,
            yaw=0.0,
        )

    processor = TemporalFoveaMapProcessor(
        base_signal_provider,
        ego_motion_provider=ego_provider,
    )

    processor.process_frame(first)
    result = processor.process_frame(second)

    assert calls == [
        ("000000", "000001"),
    ]

    assert result.ego_motion is not None
    assert result.ego_motion.translation_x == pytest.approx(0.5)


def test_no_ego_motion_is_requested_for_first_frame():
    calls = 0

    def ego_provider(
        current: SensorFrame,
        previous: SensorFrame,
    ) -> EgoMotion:
        nonlocal calls
        calls += 1
        return EgoMotion()

    processor = TemporalFoveaMapProcessor(
        base_signal_provider,
        ego_motion_provider=ego_provider,
    )

    processor.process_frame(
        make_frame("000000", 0.0)
    )

    assert calls == 0


def test_sequence_result_contains_all_processed_frames():
    frames = [
        make_frame("000000", 0.0),
        make_frame("000001", 1.0),
        make_frame("000002", 2.0),
    ]

    processor = TemporalFoveaMapProcessor(
        base_signal_provider,
    )

    result = processor.process_sequence(frames)

    assert isinstance(
        result,
        TemporalFoveaMapSequenceResult,
    )

    assert result.total_frames == 3
    assert result.frame_ids == (
        "000000",
        "000001",
        "000002",
    )

    assert result.timestamps == (
        0.0,
        1.0,
        2.0,
    )

    assert result.delta_times == (
        None,
        1.0,
        1.0,
    )

    assert result.mean_delta_time == pytest.approx(1.0)
    assert result.total_duration_seconds == pytest.approx(2.0)


def test_empty_sequence_is_supported():
    processor = TemporalFoveaMapProcessor(
        base_signal_provider,
    )

    result = processor.process_sequence([])

    assert result.total_frames == 0
    assert result.frames == ()
    assert result.total_duration_seconds == 0.0
    assert result.mean_delta_time is None


def test_processor_retains_previous_frame():
    first = make_frame(
        "000000",
        0.0,
    )

    processor = TemporalFoveaMapProcessor(
        base_signal_provider,
    )

    assert processor.previous_frame is None

    processor.process_frame(first)

    assert processor.previous_frame is first


def test_reset_clears_temporal_state():
    processor = TemporalFoveaMapProcessor(
        base_signal_provider,
    )

    processor.process_frame(
        make_frame("000000", 0.0)
    )

    processor.reset()

    assert processor.previous_frame is None
    assert processor.processed_frames == ()


def test_base_signal_shape_is_validated():
    frame = make_frame(
        "000000",
        0.0,
    )

    def bad_provider(
        frame: SensorFrame,
    ) -> dict[str, np.ndarray]:
        return {
            "DISTANCE": np.zeros(
                frame.num_points - 1,
                dtype=np.float64,
            )
        }

    processor = TemporalFoveaMapProcessor(
        bad_provider,
    )

    with pytest.raises(ValueError):
        processor.process_frame(frame)


def test_nonfinite_base_signal_is_rejected():
    frame = make_frame(
        "000000",
        0.0,
    )

    def bad_provider(
        frame: SensorFrame,
    ) -> dict[str, np.ndarray]:
        values = np.ones(
            frame.num_points,
            dtype=np.float64,
        )
        values[0] = np.nan

        return {
            "DISTANCE": values,
        }

    processor = TemporalFoveaMapProcessor(
        bad_provider,
    )

    with pytest.raises(ValueError):
        processor.process_frame(frame)


def test_empty_base_and_integrated_signals_are_rejected():
    frame = make_frame(
        "000000",
        0.0,
    )

    def empty_provider(
        frame: SensorFrame,
    ) -> dict[str, np.ndarray]:
        return {}

    processor = TemporalFoveaMapProcessor(
        empty_provider,
    )

    with pytest.raises(ValueError):
        processor.process_frame(frame)


def test_point_count_is_preserved_through_a31():
    frames = [
        make_frame("000000", 0.0),
        make_frame("000001", 1.0),
    ]

    processor = TemporalFoveaMapProcessor(
        base_signal_provider,
    )

    result = processor.process_sequence(frames)

    for frame_result in result.frames:
        assert (
            frame_result.pipeline_result.foveation[
                "importance"
            ].shape
            == (4,)
        )

        assert (
            frame_result.pipeline_result.foveation[
                "resolution"
            ].shape
            == (4,)
        )

        assert (
            frame_result.pipeline_result.foveation[
                "dominant_reason"
            ].shape
            == (4,)
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
    assert result.frame_ids == (
        "000000",
        "000001",
    )


def test_multiple_sequence_calls_are_supported():
    processor = TemporalFoveaMapProcessor(
        base_signal_provider,
    )

    first_sequence = [
        make_frame("000000", 0.0),
        make_frame("000001", 1.0),
    ]

    second_sequence = [
        make_frame("000002", 2.0),
    ]

    first_result = processor.process_sequence(
        first_sequence
    )

    second_result = processor.process_sequence(
        second_sequence
    )

    assert first_result.total_frames == 2
    assert second_result.total_frames == 1

    assert processor.processed_frames[0].frame_id == "000000"
    assert processor.processed_frames[-1].frame_id == "000002"