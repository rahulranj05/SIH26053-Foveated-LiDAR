from pathlib import Path

import numpy as np
import pytest

from mapping.kinematic_foveation import VehicleState
from mapping.sensor_frame import SensorFrame
from mapping.sequential_dataset import (
    DatasetFrameMetadata,
    SequentialDatasetConfig,
    SequentialLidarDataset,
    discover_sequential_scans,
    load_sequential_frames,
    validate_semantic_labels,
    validate_sequential_timestamps,
)


def write_scan(path: Path, points: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float32)

    assert points.ndim == 2
    assert points.shape[1] == 4

    points.tofile(path)


def make_points(offset: float = 0.0) -> np.ndarray:
    return np.array(
        [
            [1.0 + offset, 2.0, 0.5, 0.1],
            [3.0 + offset, 4.0, 0.7, 0.2],
            [5.0 + offset, 6.0, 0.9, 0.3],
        ],
        dtype=np.float32,
    )


def make_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    root.mkdir()

    write_scan(root / "000002.bin", make_points(2.0))
    write_scan(root / "000000.bin", make_points(0.0))
    write_scan(root / "000001.bin", make_points(1.0))

    return root


def test_config_defaults():
    config = SequentialDatasetConfig()

    assert config.scan_suffix == ".bin"
    assert config.recursive is False


def test_config_requires_dot_suffix():
    with pytest.raises(ValueError):
        SequentialDatasetConfig(scan_suffix="bin")


def test_config_requires_nonempty_suffix():
    with pytest.raises(ValueError):
        SequentialDatasetConfig(scan_suffix="")


def test_config_requires_boolean_recursive():
    with pytest.raises(TypeError):
        SequentialDatasetConfig(recursive=1)


def test_missing_root_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        SequentialLidarDataset(tmp_path / "missing")


def test_file_root_rejected(tmp_path):
    file_path = tmp_path / "not_a_directory.bin"
    file_path.write_bytes(b"")

    with pytest.raises(NotADirectoryError):
        SequentialLidarDataset(file_path)


def test_scan_discovery_is_numeric(tmp_path):
    root = make_dataset(tmp_path)

    scans = discover_sequential_scans(root)

    assert [path.name for path in scans] == [
        "000000.bin",
        "000001.bin",
        "000002.bin",
    ]


def test_frame_ids_are_ordered(tmp_path):
    root = make_dataset(tmp_path)

    dataset = SequentialLidarDataset(root)

    assert dataset.frame_ids == (
        "000000",
        "000001",
        "000002",
    )


def test_frame_count(tmp_path):
    root = make_dataset(tmp_path)

    dataset = SequentialLidarDataset(root)

    assert dataset.frame_count == 3


def test_scan_paths_are_exposed(tmp_path):
    root = make_dataset(tmp_path)

    dataset = SequentialLidarDataset(root)

    assert all(path.suffix == ".bin" for path in dataset.scans)


def test_metadata(tmp_path):
    root = make_dataset(tmp_path)

    dataset = SequentialLidarDataset(root)

    metadata = dataset.metadata(1)

    assert isinstance(metadata, DatasetFrameMetadata)
    assert metadata.frame_id == "000001"
    assert metadata.index == 1
    assert metadata.path.name == "000001.bin"
    assert metadata.timestamp == 1.0


def test_metadata_iteration(tmp_path):
    root = make_dataset(tmp_path)

    dataset = SequentialLidarDataset(root)

    metadata = tuple(dataset.iter_metadata())

    assert len(metadata) == 3
    assert [item.frame_id for item in metadata] == [
        "000000",
        "000001",
        "000002",
    ]
    assert [item.timestamp for item in metadata] == [
        0.0,
        1.0,
        2.0,
    ]


def test_load_frame_returns_sensor_frame(tmp_path):
    root = make_dataset(tmp_path)

    dataset = SequentialLidarDataset(root)

    frame = dataset.load_frame(0)

    assert isinstance(frame, SensorFrame)
    assert frame.frame_id == "000000"
    assert frame.timestamp == 0.0
    assert frame.xyz.shape == (3, 3)
    assert frame.xyz.dtype == np.float64


def test_load_frame_defaults_to_stationary_vehicle(tmp_path):
    root = make_dataset(tmp_path)

    dataset = SequentialLidarDataset(root)

    frame = dataset.load_frame(0)

    assert frame.vehicle_state == VehicleState(
        speed=0.0,
        yaw_rate=0.0,
        heading=0.0,
    )


def test_load_frame_does_not_fabricate_semantics(tmp_path):
    root = make_dataset(tmp_path)

    dataset = SequentialLidarDataset(root)

    frame = dataset.load_frame(0)

    assert frame.semantic_labels is None
    assert frame.dynamic_probability is None
    assert frame.has_semantics is False
    assert frame.has_dynamic_probability is False


def test_iter_frames_returns_all_frames(tmp_path):
    root = make_dataset(tmp_path)

    dataset = SequentialLidarDataset(root)

    frames = tuple(dataset.iter_frames())

    assert len(frames) == 3
    assert [frame.frame_id for frame in frames] == [
        "000000",
        "000001",
        "000002",
    ]


def test_dataset_is_iterable(tmp_path):
    root = make_dataset(tmp_path)

    dataset = SequentialLidarDataset(root)

    frames = tuple(dataset)

    assert len(frames) == 3


def test_negative_index_rejected(tmp_path):
    root = make_dataset(tmp_path)

    dataset = SequentialLidarDataset(root)

    with pytest.raises(IndexError):
        dataset.load_frame(-1)


def test_out_of_range_index_rejected(tmp_path):
    root = make_dataset(tmp_path)

    dataset = SequentialLidarDataset(root)

    with pytest.raises(IndexError):
        dataset.load_frame(3)


def test_noninteger_index_rejected(tmp_path):
    root = make_dataset(tmp_path)

    dataset = SequentialLidarDataset(root)

    with pytest.raises(TypeError):
        dataset.load_frame("0")


def test_custom_timestamps(tmp_path):
    root = make_dataset(tmp_path)

    def timestamp_provider(index, path):
        return 100.0 + index * 0.1

    dataset = SequentialLidarDataset(
        root,
        timestamp_provider=timestamp_provider,
    )

    frames = tuple(dataset)

    assert [frame.timestamp for frame in frames] == [
        100.0,
        100.1,
        100.2,
    ]


def test_nonfinite_timestamp_rejected(tmp_path):
    root = make_dataset(tmp_path)

    def timestamp_provider(index, path):
        return np.nan

    dataset = SequentialLidarDataset(
        root,
        timestamp_provider=timestamp_provider,
    )

    with pytest.raises(ValueError):
        dataset.load_frame(0)


def test_vehicle_state_provider(tmp_path):
    root = make_dataset(tmp_path)

    def vehicle_state_provider(index, path):
        return VehicleState(
            speed=float(index + 1),
            yaw_rate=0.1,
            heading=0.2,
        )

    dataset = SequentialLidarDataset(
        root,
        vehicle_state_provider=vehicle_state_provider,
    )

    frame = dataset.load_frame(2)

    assert frame.vehicle_state == VehicleState(
        speed=3.0,
        yaw_rate=0.1,
        heading=0.2,
    )


def test_vehicle_state_provider_must_return_vehicle_state(tmp_path):
    root = make_dataset(tmp_path)

    def vehicle_state_provider(index, path):
        return "invalid"

    dataset = SequentialLidarDataset(
        root,
        vehicle_state_provider=vehicle_state_provider,
    )

    with pytest.raises(TypeError):
        dataset.load_frame(0)


def test_semantic_label_provider(tmp_path):
    root = make_dataset(tmp_path)

    def semantic_provider(index, path):
        return np.array([1, 2, 3], dtype=np.int32)

    dataset = SequentialLidarDataset(
        root,
        semantic_label_provider=semantic_provider,
    )

    frame = dataset.load_frame(0)

    assert frame.semantic_labels is not None
    assert np.array_equal(
        frame.semantic_labels,
        np.array([1, 2, 3], dtype=np.int32),
    )


def test_semantic_label_provider_can_return_none(tmp_path):
    root = make_dataset(tmp_path)

    def semantic_provider(index, path):
        return None

    dataset = SequentialLidarDataset(
        root,
        semantic_label_provider=semantic_provider,
    )

    frame = dataset.load_frame(0)

    assert frame.semantic_labels is None


def test_wrong_semantic_label_count_rejected(tmp_path):
    root = make_dataset(tmp_path)

    def semantic_provider(index, path):
        return np.array([1, 2], dtype=np.int32)

    dataset = SequentialLidarDataset(
        root,
        semantic_label_provider=semantic_provider,
    )

    with pytest.raises(ValueError):
        dataset.load_frame(0)


def test_validate_semantic_labels():
    labels = np.array([1, 2, 3], dtype=np.int32)

    result = validate_semantic_labels(labels, 3)

    assert np.array_equal(result, labels)
    assert result is not labels


def test_validate_semantic_labels_requires_one_dimension():
    labels = np.array([[1, 2, 3]], dtype=np.int32)

    with pytest.raises(ValueError):
        validate_semantic_labels(labels, 3)


def test_validate_semantic_labels_requires_matching_count():
    labels = np.array([1, 2], dtype=np.int32)

    with pytest.raises(ValueError):
        validate_semantic_labels(labels, 3)


def test_validate_semantic_labels_requires_integer_dtype():
    labels = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    with pytest.raises(ValueError):
        validate_semantic_labels(labels, 3)


def test_validate_sequential_timestamps():
    frames = (
        SensorFrame(
            xyz=np.zeros((1, 3)),
            vehicle_state=VehicleState(
                speed=0.0,
                yaw_rate=0.0,
                heading=0.0,
            ),
            timestamp=0.0,
            frame_id=0,
        ),
        SensorFrame(
            xyz=np.zeros((1, 3)),
            vehicle_state=VehicleState(
                speed=0.0,
                yaw_rate=0.0,
                heading=0.0,
            ),
            timestamp=1.0,
            frame_id=1,
        ),
    )

    validate_sequential_timestamps(frames)


def test_validate_sequential_timestamps_rejects_equal():
    frames = (
        SensorFrame(
            xyz=np.zeros((1, 3)),
            vehicle_state=VehicleState(
                speed=0.0,
                yaw_rate=0.0,
                heading=0.0,
            ),
            timestamp=1.0,
            frame_id=0,
        ),
        SensorFrame(
            xyz=np.zeros((1, 3)),
            vehicle_state=VehicleState(
                speed=0.0,
                yaw_rate=0.0,
                heading=0.0,
            ),
            timestamp=1.0,
            frame_id=1,
        ),
    )

    with pytest.raises(ValueError):
        validate_sequential_timestamps(frames)


def test_validate_sequential_timestamps_rejects_reverse_order():
    frames = (
        SensorFrame(
            xyz=np.zeros((1, 3)),
            vehicle_state=VehicleState(
                speed=0.0,
                yaw_rate=0.0,
                heading=0.0,
            ),
            timestamp=2.0,
            frame_id=0,
        ),
        SensorFrame(
            xyz=np.zeros((1, 3)),
            vehicle_state=VehicleState(
                speed=0.0,
                yaw_rate=0.0,
                heading=0.0,
            ),
            timestamp=1.0,
            frame_id=1,
        ),
    )

    with pytest.raises(ValueError):
        validate_sequential_timestamps(frames)


def test_recursive_discovery(tmp_path):
    root = tmp_path / "dataset"
    nested = root / "sequence_00"
    nested.mkdir(parents=True)

    write_scan(nested / "000000.bin", make_points())

    dataset = SequentialLidarDataset(
        root,
        config=SequentialDatasetConfig(recursive=True),
    )

    assert dataset.frame_count == 1
    assert dataset.scans[0].name == "000000.bin"


def test_nonrecursive_discovery_ignores_nested_scans(tmp_path):
    root = tmp_path / "dataset"
    nested = root / "sequence_00"
    nested.mkdir(parents=True)

    write_scan(root / "000000.bin", make_points())
    write_scan(nested / "000001.bin", make_points())

    dataset = SequentialLidarDataset(root)

    assert dataset.frame_count == 1
    assert dataset.frame_ids == ("000000",)


def test_custom_suffix(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()

    write_scan(root / "000000.dat", make_points())
    write_scan(root / "000001.bin", make_points())

    dataset = SequentialLidarDataset(
        root,
        config=SequentialDatasetConfig(scan_suffix=".dat"),
    )

    assert dataset.frame_count == 1
    assert dataset.frame_ids == ("000000",)


def test_load_sequential_frames_helper(tmp_path):
    root = make_dataset(tmp_path)

    frames = load_sequential_frames(root)

    assert len(frames) == 3
    assert all(isinstance(frame, SensorFrame) for frame in frames)


def test_empty_dataset_is_valid(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()

    dataset = SequentialLidarDataset(root)

    assert dataset.frame_count == 0
    assert dataset.frame_ids == ()
    assert tuple(dataset.iter_frames()) == ()


def test_numeric_sorting_handles_unpadded_names(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()

    write_scan(root / "10.bin", make_points())
    write_scan(root / "2.bin", make_points())
    write_scan(root / "1.bin", make_points())

    dataset = SequentialLidarDataset(root)

    assert dataset.frame_ids == ("1", "2", "10")


def test_non_numeric_names_sort_after_numeric_names(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()

    write_scan(root / "frame_b.bin", make_points())
    write_scan(root / "2.bin", make_points())
    write_scan(root / "frame_a.bin", make_points())
    write_scan(root / "1.bin", make_points())

    dataset = SequentialLidarDataset(root)

    assert dataset.frame_ids == (
        "1",
        "2",
        "frame_a",
        "frame_b",
    )