from pathlib import Path

import numpy as np
import pytest

from mapping.ego_motion import EgoMotion
from mapping.kinematic_foveation import VehicleState
from mapping.sequential_dataset import SequentialLidarDataset
from mapping.sensor_frame import SensorFrame
from mapping.dataset_foveamap_runner import (
    SequentialDatasetFoveaMapRunner,
    run_sequential_dataset_foveamap,
    run_sequential_dataset_path,
)


def write_scan(
    path: Path,
    points: np.ndarray,
) -> None:

    points = np.asarray(
        points,
        dtype=np.float32,
    )

    assert points.ndim == 2
    assert points.shape[1] == 4

    points.tofile(path)


def make_points(
    offset: float = 0.0,
) -> np.ndarray:

    return np.array(
        [
            [1.0 + offset, 0.0, 0.0, 0.1],
            [2.0 + offset, 0.0, 0.0, 0.2],
            [3.0 + offset, 1.0, 0.0, 0.3],
            [4.0 + offset, -1.0, 0.0, 0.4],
        ],
        dtype=np.float32,
    )


def make_dataset(
    tmp_path: Path,
) -> SequentialLidarDataset:

    root = tmp_path / "dataset"
    root.mkdir()

    write_scan(
        root / "000002.bin",
        make_points(2.0),
    )

    write_scan(
        root / "000000.bin",
        make_points(0.0),
    )

    write_scan(
        root / "000001.bin",
        make_points(1.0),
    )

    return SequentialLidarDataset(
        root
    )


def base_signal_provider(
    frame: SensorFrame,
) -> dict[str, np.ndarray]:

    distance = np.linalg.norm(
        frame.xyz[:, :2],
        axis=1,
    )

    return {
        "DISTANCE": (
            1.0 / (1.0 + distance)
        )
    }


def test_runner_requires_sequential_lidar_dataset(
    tmp_path: Path,
) -> None:

    with pytest.raises(TypeError):
        SequentialDatasetFoveaMapRunner(
            tmp_path,
            base_signal_provider,
        )


def test_runner_requires_callable_base_provider(
    tmp_path: Path,
) -> None:

    dataset = make_dataset(
        tmp_path
    )

    with pytest.raises(TypeError):
        SequentialDatasetFoveaMapRunner(
            dataset,
            None,
        )


def test_runner_rejects_noncallable_ego_provider(
    tmp_path: Path,
) -> None:

    dataset = make_dataset(
        tmp_path
    )

    with pytest.raises(TypeError):
        SequentialDatasetFoveaMapRunner(
            dataset,
            base_signal_provider,
            ego_motion_provider=123,
        )


def test_runner_properties(
    tmp_path: Path,
) -> None:

    dataset = make_dataset(
        tmp_path
    )

    runner = SequentialDatasetFoveaMapRunner(
        dataset,
        base_signal_provider,
    )

    assert runner.dataset is dataset
    assert runner.processor is not None
    assert runner.processed_frames == ()


def test_runner_processes_entire_dataset(
    tmp_path: Path,
) -> None:

    dataset = make_dataset(
        tmp_path
    )

    runner = SequentialDatasetFoveaMapRunner(
        dataset,
        base_signal_provider,
    )

    result = runner.run()

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

    assert result.mean_delta_time == pytest.approx(
        1.0
    )

    assert result.total_duration_seconds == pytest.approx(
        2.0
    )


def test_runner_preserves_a31_results(
    tmp_path: Path,
) -> None:

    dataset = make_dataset(
        tmp_path
    )

    runner = SequentialDatasetFoveaMapRunner(
        dataset,
        base_signal_provider,
    )

    result = runner.run()

    assert len(
        result.frames
    ) == dataset.frame_count

    for frame_result in result.frames:

        assert frame_result.pipeline_result is not None

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


def test_runner_preserves_temporal_state(
    tmp_path: Path,
) -> None:

    dataset = make_dataset(
        tmp_path
    )

    runner = SequentialDatasetFoveaMapRunner(
        dataset,
        base_signal_provider,
    )

    result = runner.run()

    assert runner.processor.previous_frame is not None

    assert (
        runner.processor.previous_frame.frame_id
        == "000002"
    )

    assert len(
        runner.processed_frames
    ) == result.total_frames


def test_runner_reset_clears_a31_state(
    tmp_path: Path,
) -> None:

    dataset = make_dataset(
        tmp_path
    )

    runner = SequentialDatasetFoveaMapRunner(
        dataset,
        base_signal_provider,
    )

    runner.run()

    assert runner.processed_frames

    runner.reset()

    assert runner.processor.previous_frame is None
    assert runner.processed_frames == ()


def test_runner_can_be_reused_after_reset(
    tmp_path: Path,
) -> None:

    dataset = make_dataset(
        tmp_path
    )

    runner = SequentialDatasetFoveaMapRunner(
        dataset,
        base_signal_provider,
    )

    first = runner.run()

    runner.reset()

    second = runner.run()

    assert first.total_frames == 3
    assert second.total_frames == 3

    assert second.frame_ids == (
        "000000",
        "000001",
        "000002",
    )


def test_convenience_function_runs_dataset(
    tmp_path: Path,
) -> None:

    dataset = make_dataset(
        tmp_path
    )

    result = run_sequential_dataset_foveamap(
        dataset,
        base_signal_provider,
    )

    assert result.total_frames == 3

    assert result.frame_ids == (
        "000000",
        "000001",
        "000002",
    )


def test_path_convenience_function_runs_dataset(
    tmp_path: Path,
) -> None:

    root = tmp_path / "dataset"
    root.mkdir()

    write_scan(
        root / "000001.bin",
        make_points(1.0),
    )

    write_scan(
        root / "000000.bin",
        make_points(0.0),
    )

    result = run_sequential_dataset_path(
        root,
        base_signal_provider,
    )

    assert result.total_frames == 2

    assert result.frame_ids == (
        "000000",
        "000001",
    )


def test_empty_dataset_is_supported(
    tmp_path: Path,
) -> None:

    root = tmp_path / "empty_dataset"
    root.mkdir()

    dataset = SequentialLidarDataset(
        root
    )

    runner = SequentialDatasetFoveaMapRunner(
        dataset,
        base_signal_provider,
    )

    result = runner.run()

    assert result.total_frames == 0
    assert result.frames == ()
    assert result.frame_ids == ()
    assert result.timestamps == ()
    assert result.delta_times == ()
    assert result.mean_delta_time is None
    assert result.total_duration_seconds == 0.0


def test_semantic_information_reaches_a31(
    tmp_path: Path,
) -> None:

    root = tmp_path / "semantic_dataset"
    root.mkdir()

    write_scan(
        root / "000000.bin",
        make_points(),
    )

    def semantic_provider(
        index: int,
        path: Path,
    ) -> np.ndarray:

        return np.array(
            [1, 2, 3, 5],
            dtype=np.int64,
        )

    dataset = SequentialLidarDataset(
        root,
        semantic_label_provider=semantic_provider,
    )

    runner = SequentialDatasetFoveaMapRunner(
        dataset,
        base_signal_provider,
    )

    result = runner.run()

    assert result.total_frames == 1

    frame_result = result.frames[0]

    assert "semantic" in frame_result.signals

    assert frame_result.signals[
        "semantic"
    ].shape == (4,)


def test_vehicle_state_is_preserved_from_dataset(
    tmp_path: Path,
) -> None:

    dataset = make_dataset(
        tmp_path
    )

    def vehicle_state_provider(
        index: int,
        path: Path,
    ) -> VehicleState:

        return VehicleState(
            speed=float(index + 1),
            yaw_rate=0.1,
            heading=0.2,
        )

    dataset = SequentialLidarDataset(
        dataset.root,
        vehicle_state_provider=vehicle_state_provider,
    )

    runner = SequentialDatasetFoveaMapRunner(
        dataset,
        base_signal_provider,
    )

    result = runner.run()

    assert result.total_frames == 3

    assert (
        runner.processor.previous_frame.vehicle_state
        == VehicleState(
            speed=3.0,
            yaw_rate=0.1,
            heading=0.2,
        )
    )


def test_custom_timestamps_flow_into_a31(
    tmp_path: Path,
) -> None:

    root = tmp_path / "timestamp_dataset"
    root.mkdir()

    write_scan(
        root / "000000.bin",
        make_points(),
    )

    write_scan(
        root / "000001.bin",
        make_points(1.0),
    )

    def timestamp_provider(
        index: int,
        path: Path,
    ) -> float:

        return 100.0 + (
            index * 0.1
        )

    dataset = SequentialLidarDataset(
        root,
        timestamp_provider=timestamp_provider,
    )

    runner = SequentialDatasetFoveaMapRunner(
        dataset,
        base_signal_provider,
    )

    result = runner.run()

    assert result.timestamps == (
        100.0,
        100.1,
    )

    assert result.delta_times[1] == pytest.approx(
        0.1
    )


def test_ego_motion_provider_is_forwarded(
    tmp_path: Path,
) -> None:

    dataset = make_dataset(
        tmp_path
    )

    calls: list[tuple[str, str]] = []

    def ego_motion_provider(
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

    runner = SequentialDatasetFoveaMapRunner(
        dataset,
        base_signal_provider,
        ego_motion_provider=ego_motion_provider,
    )

    result = runner.run()

    assert result.total_frames == 3

    assert calls == [
        ("000000", "000001"),
        ("000001", "000002"),
    ]

    assert result.frames[0].ego_motion is None

    assert result.frames[1].ego_motion is not None
    assert result.frames[2].ego_motion is not None


def test_convenience_function_forwards_ego_motion(
    tmp_path: Path,
) -> None:

    dataset = make_dataset(
        tmp_path
    )

    calls = 0

    def ego_motion_provider(
        current: SensorFrame,
        previous: SensorFrame,
    ) -> EgoMotion:

        nonlocal calls

        calls += 1

        return EgoMotion()

    result = run_sequential_dataset_foveamap(
        dataset,
        base_signal_provider,
        ego_motion_provider=ego_motion_provider,
    )

    assert result.total_frames == 3
    assert calls == 2


def test_dataset_order_is_used_even_when_files_are_unsorted(
    tmp_path: Path,
) -> None:

    root = tmp_path / "ordered_dataset"
    root.mkdir()

    write_scan(
        root / "000010.bin",
        make_points(10.0),
    )

    write_scan(
        root / "000002.bin",
        make_points(2.0),
    )

    write_scan(
        root / "000001.bin",
        make_points(1.0),
    )

    dataset = SequentialLidarDataset(
        root
    )

    runner = SequentialDatasetFoveaMapRunner(
        dataset,
        base_signal_provider,
    )

    result = runner.run()

    assert result.frame_ids == (
        "000001",
        "000002",
        "000010",
    )


def test_runner_does_not_fabricate_dynamic_information(
    tmp_path: Path,
) -> None:

    dataset = make_dataset(
        tmp_path
    )

    frames = tuple(
        dataset.iter_frames()
    )

    assert frames

    for frame in frames:
        assert frame.dynamic_probability is None

    runner = SequentialDatasetFoveaMapRunner(
        dataset,
        base_signal_provider,
    )

    result = runner.run()

    assert result.total_frames == dataset.frame_count

    assert "dynamic" not in result.frames[0].signals

    assert "dynamic" in result.frames[1].signals
    assert "dynamic" in result.frames[2].signals