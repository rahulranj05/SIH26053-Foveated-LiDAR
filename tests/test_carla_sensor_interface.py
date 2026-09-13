"""
Tests for A27 — CARLA sensor/data interface.
"""

from __future__ import annotations

import numpy as np
import pytest

from mapping.carla_sensor_interface import (
    CarlaSensorInterfaceConfig,
    carla_lidar_to_sensor_frame,
    carla_measurement_to_sensor_frame,
    carla_vehicle_state,
    parse_carla_lidar_raw_data,
)


class FakeVector:
    def __init__(
        self,
        x: float,
        y: float,
        z: float,
    ) -> None:
        self.x = x
        self.y = y
        self.z = z


class FakeRotation:
    def __init__(
        self,
        yaw: float,
    ) -> None:
        self.yaw = yaw


class FakeTransform:
    def __init__(
        self,
        yaw: float,
    ) -> None:
        self.rotation = FakeRotation(
            yaw
        )


class FakeVehicle:
    def __init__(
        self,
        velocity: FakeVector,
        angular_velocity: FakeVector,
        yaw: float,
    ) -> None:
        self._velocity = velocity
        self._angular_velocity = angular_velocity
        self._transform = FakeTransform(
            yaw
        )

    def get_velocity(self) -> FakeVector:
        return self._velocity

    def get_angular_velocity(self) -> FakeVector:
        return self._angular_velocity

    def get_transform(self) -> FakeTransform:
        return self._transform


class FakeLidarMeasurement:
    def __init__(
        self,
        points: np.ndarray,
        frame: int = 0,
        timestamp: float = 0.0,
    ) -> None:
        points = np.asarray(
            points,
            dtype=np.float32,
        )

        intensity = np.zeros(
            (points.shape[0], 1),
            dtype=np.float32,
        )

        raw_points = np.concatenate(
            [
                points,
                intensity,
            ],
            axis=1,
        )

        self.raw_data = raw_points.tobytes()
        self.frame = frame
        self.timestamp = timestamp


def make_vehicle() -> FakeVehicle:
    return FakeVehicle(
        velocity=FakeVector(
            3.0,
            4.0,
            0.0,
        ),
        angular_velocity=FakeVector(
            0.0,
            0.0,
            10.0,
        ),
        yaw=90.0,
    )


def make_measurement() -> FakeLidarMeasurement:
    return FakeLidarMeasurement(
        points=np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [-1.0, 0.5, 2.0],
            ],
            dtype=np.float32,
        ),
        frame=42,
        timestamp=12.5,
    )


def test_default_config_uses_four_values_per_lidar_point() -> None:
    config = CarlaSensorInterfaceConfig()

    assert config.lidar_point_stride == 4


def test_invalid_stride_is_rejected() -> None:
    with pytest.raises(ValueError):
        CarlaSensorInterfaceConfig(
            lidar_point_stride=0
        )


def test_parse_carla_lidar_raw_data_extracts_xyz() -> None:
    measurement = make_measurement()

    xyz = parse_carla_lidar_raw_data(
        measurement.raw_data
    )

    expected = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [-1.0, 0.5, 2.0],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        xyz,
        expected,
    )

    assert xyz.dtype == np.float64
    assert xyz.shape == (3, 3)


def test_empty_lidar_data_returns_empty_xyz() -> None:
    xyz = parse_carla_lidar_raw_data(
        b""
    )

    assert xyz.shape == (0, 3)
    assert xyz.dtype == np.float64


def test_invalid_raw_data_type_is_rejected() -> None:
    with pytest.raises(TypeError):
        parse_carla_lidar_raw_data(
            "not bytes"  # type: ignore[arg-type]
        )


def test_invalid_raw_data_size_is_rejected() -> None:
    raw = np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float32,
    ).tobytes()

    with pytest.raises(ValueError):
        parse_carla_lidar_raw_data(
            raw
        )


def test_nonfinite_lidar_xyz_is_rejected() -> None:
    points = np.array(
        [
            [1.0, np.nan, 3.0, 0.0],
        ],
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        parse_carla_lidar_raw_data(
            points.tobytes()
        )


def test_vehicle_speed_is_computed_from_velocity_vector() -> None:
    vehicle = make_vehicle()

    state = carla_vehicle_state(
        vehicle
    )

    assert state.speed == pytest.approx(
        5.0
    )


def test_vehicle_yaw_rate_is_converted_to_radians() -> None:
    vehicle = make_vehicle()

    state = carla_vehicle_state(
        vehicle
    )

    assert state.yaw_rate == pytest.approx(
        np.deg2rad(10.0)
    )


def test_vehicle_heading_is_converted_to_radians() -> None:
    vehicle = make_vehicle()

    state = carla_vehicle_state(
        vehicle
    )

    assert state.heading == pytest.approx(
        np.deg2rad(90.0)
    )


def test_missing_vehicle_method_is_rejected() -> None:
    class IncompleteVehicle:
        pass

    with pytest.raises(AttributeError):
        carla_vehicle_state(
            IncompleteVehicle()
        )


def test_nonfinite_vehicle_velocity_is_rejected() -> None:
    vehicle = FakeVehicle(
        velocity=FakeVector(
            np.nan,
            0.0,
            0.0,
        ),
        angular_velocity=FakeVector(
            0.0,
            0.0,
            0.0,
        ),
        yaw=0.0,
    )

    with pytest.raises(ValueError):
        carla_vehicle_state(
            vehicle
        )


def test_lidar_measurement_becomes_sensor_frame() -> None:
    measurement = make_measurement()
    vehicle = make_vehicle()

    frame = carla_lidar_to_sensor_frame(
        measurement,
        vehicle,
    )

    assert frame.frame_id == 42
    assert frame.timestamp == pytest.approx(
        12.5
    )
    assert frame.num_points == 3

    np.testing.assert_allclose(
        frame.xyz,
        np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [-1.0, 0.5, 2.0],
            ]
        ),
    )

    assert frame.vehicle_state.speed == pytest.approx(
        5.0
    )


def test_semantic_labels_are_preserved() -> None:
    measurement = make_measurement()
    vehicle = make_vehicle()

    labels = np.array(
        [1, 3, 4],
        dtype=np.int32,
    )

    frame = carla_lidar_to_sensor_frame(
        measurement,
        vehicle,
        semantic_labels=labels,
    )

    assert frame.has_semantics

    np.testing.assert_array_equal(
        frame.semantic_labels,
        labels,
    )


def test_dynamic_probability_is_preserved() -> None:
    measurement = make_measurement()
    vehicle = make_vehicle()

    dynamic = np.array(
        [0.0, 0.5, 1.0],
        dtype=np.float64,
    )

    frame = carla_lidar_to_sensor_frame(
        measurement,
        vehicle,
        dynamic_probability=dynamic,
    )

    assert frame.has_dynamic_probability

    np.testing.assert_allclose(
        frame.dynamic_probability,
        dynamic,
    )


def test_wrong_semantic_length_is_rejected() -> None:
    measurement = make_measurement()
    vehicle = make_vehicle()

    labels = np.array(
        [1, 2],
        dtype=np.int32,
    )

    with pytest.raises(ValueError):
        carla_lidar_to_sensor_frame(
            measurement,
            vehicle,
            semantic_labels=labels,
        )


def test_wrong_dynamic_length_is_rejected() -> None:
    measurement = make_measurement()
    vehicle = make_vehicle()

    dynamic = np.array(
        [0.5, 0.5],
        dtype=np.float64,
    )

    with pytest.raises(ValueError):
        carla_lidar_to_sensor_frame(
            measurement,
            vehicle,
            dynamic_probability=dynamic,
        )


def test_measurement_frame_id_is_required() -> None:
    measurement = make_measurement()
    del measurement.frame

    vehicle = make_vehicle()

    with pytest.raises(AttributeError):
        carla_lidar_to_sensor_frame(
            measurement,
            vehicle,
        )


def test_measurement_timestamp_must_be_finite() -> None:
    measurement = make_measurement()
    measurement.timestamp = np.nan

    vehicle = make_vehicle()

    with pytest.raises(ValueError):
        carla_lidar_to_sensor_frame(
            measurement,
            vehicle,
        )


def test_measurement_to_sensor_frame_alias_matches_primary_adapter() -> None:
    measurement = make_measurement()
    vehicle = make_vehicle()

    frame_a = carla_lidar_to_sensor_frame(
        measurement,
        vehicle,
    )

    frame_b = carla_measurement_to_sensor_frame(
        measurement,
        vehicle,
    )

    assert frame_a.frame_id == frame_b.frame_id
    assert frame_a.timestamp == frame_b.timestamp

    np.testing.assert_allclose(
        frame_a.xyz,
        frame_b.xyz,
    )


def test_sensor_frame_arrays_are_read_only() -> None:
    measurement = make_measurement()
    vehicle = make_vehicle()

    frame = carla_lidar_to_sensor_frame(
        measurement,
        vehicle,
    )

    assert not frame.xyz.flags.writeable


def test_carla_interface_preserves_missing_semantics() -> None:
    measurement = make_measurement()
    vehicle = make_vehicle()

    frame = carla_lidar_to_sensor_frame(
        measurement,
        vehicle,
    )

    assert frame.semantic_labels is None
    assert frame.dynamic_probability is None
    assert not frame.has_semantics
    assert not frame.has_dynamic_probability