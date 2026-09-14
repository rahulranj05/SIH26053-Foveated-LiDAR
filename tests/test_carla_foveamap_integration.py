"""
P7 — CARLA to FoveaMap production integration test.

This test does not require the CARLA Python package or a running simulator.
It uses CARLA-compatible fake sensor/vehicle objects and exercises the real
CARLA adapter plus the production temporal FoveaMap pipeline.
"""

from __future__ import annotations

import numpy as np

from mapping.carla_sensor_interface import carla_lidar_to_sensor_frame
from mapping.ego_motion import EgoMotion, transform_points
from mapping.temporal_foveamap_pipeline import TemporalFoveaMapProcessor


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
        self.rotation = FakeRotation(yaw)


class FakeVehicle:
    def __init__(
        self,
        speed: float,
        yaw_rate_degrees: float,
        yaw_degrees: float,
    ) -> None:
        self._velocity = FakeVector(
            speed,
            0.0,
            0.0,
        )
        self._angular_velocity = FakeVector(
            0.0,
            0.0,
            yaw_rate_degrees,
        )
        self._transform = FakeTransform(
            yaw_degrees,
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
        frame: int,
        timestamp: float,
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


def test_carla_frames_flow_through_temporal_foveamap_pipeline() -> None:
    points = np.fromfile(
        "datasets/TEST/000000.bin",
        dtype=np.float32,
    ).reshape(-1, 4)[:, :3].astype(np.float64)

    motion = EgoMotion(
        translation_x=1.5,
        translation_y=-0.5,
        translation_z=0.0,
        yaw=0.08,
    )

    current_points = transform_points(
        points,
        motion,
    )

    moving_mask = (
        (points[:, 0] > 5.0)
        & (points[:, 0] < 10.0)
        & (points[:, 1] > -5.0)
        & (points[:, 1] < 5.0)
    )

    current_points[moving_mask, 0] += 2.0

    previous_measurement = FakeLidarMeasurement(
        points=points,
        frame=0,
        timestamp=0.0,
    )

    current_measurement = FakeLidarMeasurement(
        points=current_points,
        frame=1,
        timestamp=0.1,
    )

    previous_vehicle = FakeVehicle(
        speed=8.0,
        yaw_rate_degrees=np.rad2deg(0.08),
        yaw_degrees=0.0,
    )

    current_vehicle = FakeVehicle(
        speed=8.0,
        yaw_rate_degrees=np.rad2deg(0.08),
        yaw_degrees=np.rad2deg(0.08),
    )

    previous_frame = carla_lidar_to_sensor_frame(
        previous_measurement,
        previous_vehicle,
    )

    current_frame = carla_lidar_to_sensor_frame(
        current_measurement,
        current_vehicle,
    )

    assert previous_frame.frame_id == 0
    assert current_frame.frame_id == 1
    assert previous_frame.timestamp == 0.0
    assert current_frame.timestamp == 0.1
    assert previous_frame.num_points == points.shape[0]
    assert current_frame.num_points == points.shape[0]

    def ego_motion_provider(current, previous):
        assert current.frame_id == 1
        assert previous.frame_id == 0
        return motion

    processor = TemporalFoveaMapProcessor(
        ego_motion_provider=ego_motion_provider,
    )

    processor.process_frame(previous_frame)
    result = processor.process_frame(current_frame)

    dynamic = result.signals["dynamic"]

    stationary = dynamic[~moving_mask]
    moving = dynamic[moving_mask]

    assert result.ego_motion is not None
    assert np.all(np.isfinite(dynamic))
    assert stationary.mean() == 0.0
    assert moving.mean() > stationary.mean()
    assert moving.max() > stationary.max()

    leaf_map = result.pipeline_result.leaf_map

    assert leaf_map.active_cells > 0
    assert np.all(np.isfinite(leaf_map.z_min))
    assert np.all(np.isfinite(leaf_map.z_max))
    assert np.all(np.isfinite(leaf_map.z_mean))
    assert np.all(np.isfinite(leaf_map.z_variance))
