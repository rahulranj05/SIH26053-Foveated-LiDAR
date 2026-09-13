from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

from mapping.kinematic_foveation import VehicleState
from mapping.sensor_frame import SensorFrame
from mapping.sequential_processor import (
    SequentialFoveaMapProcessor,
    iter_sensor_frames,
    process_sensor_sequence,
)


def make_frame(
    frame_id: int,
    timestamp: float,
    num_points: int = 32,
) -> SensorFrame:
    xyz = np.zeros(
        (num_points, 3),
        dtype=np.float64,
    )

    xyz[:, 0] = np.linspace(
        1.0,
        20.0,
        num_points,
    )

    xyz[:, 1] = np.linspace(
        -5.0,
        5.0,
        num_points,
    )

    xyz[:, 2] = 0.0

    return SensorFrame(
        xyz=xyz,
        vehicle_state=VehicleState(
            speed=5.0,
            yaw_rate=0.0,
            heading=0.0,
        ),
        timestamp=timestamp,
        frame_id=frame_id,
    )


def basic_signal_provider(
    frame: SensorFrame,
) -> dict[str, np.ndarray]:
    """
    Minimal deterministic DISTANCE signal provider for A23 tests.

    Uses the existing FoveaMap signal namespace with synthetic values.
    It is not a replacement for the real distance-foveation computation.
    """
    return {
        "DISTANCE": np.full(
            frame.num_points,
            0.5,
            dtype=np.float64,
        )
    }


def test_processor_starts_with_no_previous_frame():
    processor = SequentialFoveaMapProcessor(
        basic_signal_provider,
    )

    assert processor.previous_frame is None
    assert processor.processed_frames == 0


def test_first_frame_has_no_delta_time():
    processor = SequentialFoveaMapProcessor(
        basic_signal_provider,
    )

    frame = make_frame(
        frame_id=0,
        timestamp=1.0,
    )

    result = processor.process_frame(frame)

    assert result.frame_id == 0
    assert result.timestamp == 1.0
    assert result.delta_time is None
    assert processor.processed_frames == 1
    assert processor.previous_frame is frame


def test_second_frame_computes_delta_time():
    processor = SequentialFoveaMapProcessor(
        basic_signal_provider,
    )

    first = make_frame(
        frame_id=0,
        timestamp=1.0,
    )

    second = make_frame(
        frame_id=1,
        timestamp=1.1,
    )

    processor.process_frame(first)
    result = processor.process_frame(second)

    assert result.delta_time == pytest.approx(0.1)
    assert result.frame_id == 1
    assert processor.processed_frames == 2
    assert processor.previous_frame is second


def test_timestamps_must_increase():
    processor = SequentialFoveaMapProcessor(
        basic_signal_provider,
    )

    processor.process_frame(
        make_frame(
            frame_id=0,
            timestamp=2.0,
        )
    )

    with pytest.raises(
        ValueError,
        match="strictly increasing",
    ):
        processor.process_frame(
            make_frame(
                frame_id=1,
                timestamp=2.0,
            )
        )


def test_decreasing_timestamp_is_rejected():
    processor = SequentialFoveaMapProcessor(
        basic_signal_provider,
    )

    processor.process_frame(
        make_frame(
            frame_id=0,
            timestamp=5.0,
        )
    )

    with pytest.raises(
        ValueError,
        match="strictly increasing",
    ):
        processor.process_frame(
            make_frame(
                frame_id=1,
                timestamp=4.0,
            )
        )


def test_non_sensor_frame_is_rejected():
    processor = SequentialFoveaMapProcessor(
        basic_signal_provider,
    )

    with pytest.raises(
        TypeError,
        match="SensorFrame",
    ):
        processor.process_frame("not a frame")


def test_signal_provider_must_be_callable():
    with pytest.raises(
        TypeError,
        match="signal_provider must be callable",
    ):
        SequentialFoveaMapProcessor(None)


def test_signal_provider_must_return_dict():
    def invalid_provider(frame):
        return np.zeros(
            frame.num_points,
            dtype=np.float64,
        )

    processor = SequentialFoveaMapProcessor(
        invalid_provider,
    )

    with pytest.raises(
        TypeError,
        match="dictionary of signals",
    ):
        processor.process_frame(
            make_frame(
                frame_id=0,
                timestamp=1.0,
            )
        )


def test_sequence_processing_preserves_order():
    frames = [
        make_frame(0, 1.0),
        make_frame(1, 1.1),
        make_frame(2, 1.2),
    ]

    result = process_sensor_sequence(
        frames,
        basic_signal_provider,
    )

    assert result.total_frames == 3
    assert result.frame_ids == (0, 1, 2)

    np.testing.assert_allclose(
        result.timestamps,
        (1.0, 1.1, 1.2),
    )


def test_sequence_duration():
    frames = [
        make_frame(0, 10.0),
        make_frame(1, 10.1),
        make_frame(2, 10.25),
    ]

    result = process_sensor_sequence(
        frames,
        basic_signal_provider,
    )

    assert result.total_duration_seconds == pytest.approx(
        0.25
    )


def test_mean_delta_time():
    frames = [
        make_frame(0, 0.0),
        make_frame(1, 0.1),
        make_frame(2, 0.3),
    ]

    result = process_sensor_sequence(
        frames,
        basic_signal_provider,
    )

    assert result.mean_delta_time == pytest.approx(
        0.15
    )


def test_empty_sequence():
    result = process_sensor_sequence(
        [],
        basic_signal_provider,
    )

    assert result.total_frames == 0
    assert result.total_duration_seconds == 0.0
    assert result.frame_ids == ()
    assert result.timestamps == ()
    assert result.mean_delta_time == 0.0


def test_reset_clears_temporal_state():
    processor = SequentialFoveaMapProcessor(
        basic_signal_provider,
    )

    frame = make_frame(
        frame_id=0,
        timestamp=1.0,
    )

    processor.process_frame(frame)

    assert processor.previous_frame is frame
    assert processor.processed_frames == 1

    processor.reset()

    assert processor.previous_frame is None
    assert processor.processed_frames == 0


def test_processing_after_reset_starts_new_sequence():
    processor = SequentialFoveaMapProcessor(
        basic_signal_provider,
    )

    processor.process_frame(
        make_frame(
            frame_id=0,
            timestamp=1.0,
        )
    )

    processor.reset()

    result = processor.process_frame(
        make_frame(
            frame_id=100,
            timestamp=50.0,
        )
    )

    assert result.frame_id == 100
    assert result.delta_time is None
    assert processor.processed_frames == 1


def test_iter_sensor_frames_is_lazy_and_validates():
    frames = [
        make_frame(0, 1.0),
        make_frame(1, 1.1),
    ]

    iterator = iter_sensor_frames(frames)

    assert next(iterator) is frames[0]
    assert next(iterator) is frames[1]

    with pytest.raises(StopIteration):
        next(iterator)


def test_iter_sensor_frames_rejects_invalid_item():
    with pytest.raises(
        TypeError,
        match="SensorFrame",
    ):
        list(
            iter_sensor_frames(
                [
                    make_frame(0, 1.0),
                    "invalid",
                ]
            )
        )


def test_a20_result_is_attached_to_each_frame():
    frames = [
        make_frame(0, 1.0),
        make_frame(1, 1.1),
    ]

    result = process_sensor_sequence(
        frames,
        basic_signal_provider,
    )

    assert len(result.frames) == 2

    for frame_result in result.frames:
        assert frame_result.result is not None
        assert hasattr(
            frame_result.result,
            "leaf_map",
        )
        assert hasattr(
            frame_result.result,
            "foveation",
        )


def test_signal_provider_receives_actual_sensor_frame():
    received = []

    def provider(frame):
        received.append(frame)

        return {
            "DISTANCE": np.full(
                frame.num_points,
                0.5,
                dtype=np.float64,
            )
        }

    processor = SequentialFoveaMapProcessor(
        provider,
    )

    frame = make_frame(
        frame_id=7,
        timestamp=3.0,
    )

    processor.process_frame(frame)

    assert received == [frame]


def test_each_frame_can_have_different_signals():
    def provider(frame):
        value = (
            0.2
            if frame.frame_id == 0
            else 0.9
        )

        return {
            "DISTANCE": np.full(
                frame.num_points,
                value,
                dtype=np.float64,
            )
        }

    result = process_sensor_sequence(
        [
            make_frame(0, 1.0),
            make_frame(1, 1.1),
        ],
        provider,
    )

    first = result.frames[0].result
    second = result.frames[1].result

    assert not np.array_equal(
        first.foveation["importance"],
        second.foveation["importance"],
    )
