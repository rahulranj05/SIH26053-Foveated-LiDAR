"""
A27 — CARLA sensor/data interface for FoveaMap.

This module converts CARLA-style LiDAR measurements and vehicle-state
information into the canonical FoveaMap SensorFrame interface.

The implementation intentionally does not import the CARLA Python package.
Instead, it uses the attributes exposed by CARLA sensor/vehicle objects.
This keeps the adapter unit-testable without requiring a running CARLA
installation.

Expected CARLA-style LiDAR measurement attributes:

    raw_data
    frame
    timestamp

Expected CARLA-style vehicle attributes:

    get_velocity()
    get_angular_velocity()
    get_transform()

Expected vector attributes:

    x
    y
    z

Expected transform structure:

    transform.rotation.yaw

CARLA yaw is expressed in degrees and is converted to radians.
CARLA angular velocity is expressed in degrees/second and is converted
to radians/second.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mapping.kinematic_foveation import VehicleState
from mapping.sensor_frame import SensorFrame


@dataclass(frozen=True)
class CarlaSensorInterfaceConfig:
    """
    Configuration for the CARLA sensor adapter.

    Parameters
    ----------
    lidar_point_stride:
        Number of float32 values per LiDAR point.

        CARLA's standard LiDAR raw representation is:

            x, y, z, intensity

        therefore the default is 4.
    """

    lidar_point_stride: int = 4

    def __post_init__(self) -> None:
        if self.lidar_point_stride <= 0:
            raise ValueError(
                "lidar_point_stride must be positive"
            )


def parse_carla_lidar_raw_data(
    raw_data: bytes | bytearray | memoryview,
    config: CarlaSensorInterfaceConfig | None = None,
) -> np.ndarray:
    """
    Parse CARLA LiDAR raw_data into an XYZ point cloud.

    CARLA LiDAR data is represented as packed float32 values:

        x, y, z, intensity,
        x, y, z, intensity,
        ...

    Only XYZ is retained because SensorFrame stores XYZ separately from
    intensity.

    Parameters
    ----------
    raw_data:
        CARLA LiDAR ``raw_data`` byte buffer.

    config:
        Optional adapter configuration.

    Returns
    -------
    numpy.ndarray
        Float64 array with shape (N, 3).
    """
    if config is None:
        config = CarlaSensorInterfaceConfig()

    if not isinstance(
        raw_data,
        (bytes, bytearray, memoryview),
    ):
        raise TypeError(
            "raw_data must be bytes-like"
        )

    raw = np.frombuffer(
        raw_data,
        dtype=np.float32,
    )

    if raw.size == 0:
        return np.empty(
            (0, 3),
            dtype=np.float64,
        )

    if raw.size % config.lidar_point_stride != 0:
        raise ValueError(
            "CARLA LiDAR raw_data size is not divisible "
            "by lidar_point_stride"
        )

    points = raw.reshape(
        -1,
        config.lidar_point_stride,
    )

    xyz = np.asarray(
        points[:, :3],
        dtype=np.float64,
    )

    if not np.all(np.isfinite(xyz)):
        raise ValueError(
            "CARLA LiDAR XYZ values must be finite"
        )

    return xyz


def _read_vector_component(
    vector: Any,
    component: str,
) -> float:
    """
    Read one numeric component from a CARLA-style vector.
    """
    if not hasattr(vector, component):
        raise AttributeError(
            f"CARLA vector is missing '{component}'"
        )

    value = float(
        getattr(vector, component)
    )

    if not np.isfinite(value):
        raise ValueError(
            f"CARLA vector component '{component}' must be finite"
        )

    return value


def _velocity_magnitude(
    velocity: Any,
) -> float:
    """
    Compute vehicle speed from a CARLA-style velocity vector.
    """
    vx = _read_vector_component(
        velocity,
        "x",
    )
    vy = _read_vector_component(
        velocity,
        "y",
    )
    vz = _read_vector_component(
        velocity,
        "z",
    )

    return float(
        np.sqrt(
            vx * vx
            + vy * vy
            + vz * vz
        )
    )


def _yaw_rate_radians_per_second(
    angular_velocity: Any,
) -> float:
    """
    Convert CARLA angular velocity around Z from degrees/s to rad/s.
    """
    yaw_rate_degrees = _read_vector_component(
        angular_velocity,
        "z",
    )

    return float(
        np.deg2rad(yaw_rate_degrees)
    )


def _heading_radians(
    transform: Any,
) -> float:
    """
    Convert CARLA transform yaw from degrees to radians.
    """
    if not hasattr(
        transform,
        "rotation",
    ):
        raise AttributeError(
            "CARLA transform is missing 'rotation'"
        )

    rotation = transform.rotation

    if not hasattr(
        rotation,
        "yaw",
    ):
        raise AttributeError(
            "CARLA rotation is missing 'yaw'"
        )

    yaw_degrees = float(
        rotation.yaw
    )

    if not np.isfinite(yaw_degrees):
        raise ValueError(
            "CARLA yaw must be finite"
        )

    return float(
        np.deg2rad(yaw_degrees)
    )


def carla_vehicle_state(
    vehicle: Any,
) -> VehicleState:
    """
    Convert a CARLA vehicle actor into a FoveaMap VehicleState.

    The CARLA vehicle must provide:

        get_velocity()
        get_angular_velocity()
        get_transform()

    Returns
    -------
    VehicleState
        FoveaMap vehicle state containing speed, yaw rate and heading.
    """
    if vehicle is None:
        raise TypeError(
            "vehicle must not be None"
        )

    required_methods = (
        "get_velocity",
        "get_angular_velocity",
        "get_transform",
    )

    for method_name in required_methods:
        if not callable(
            getattr(
                vehicle,
                method_name,
                None,
            )
        ):
            raise AttributeError(
                f"vehicle must provide {method_name}()"
            )

    velocity = vehicle.get_velocity()
    angular_velocity = vehicle.get_angular_velocity()
    transform = vehicle.get_transform()

    return VehicleState(
        speed=_velocity_magnitude(
            velocity
        ),
        yaw_rate=_yaw_rate_radians_per_second(
            angular_velocity
        ),
        heading=_heading_radians(
            transform
        ),
    )


def _measurement_frame_id(
    measurement: Any,
) -> Any:
    """
    Extract the CARLA LiDAR frame identifier.
    """
    if not hasattr(
        measurement,
        "frame",
    ):
        raise AttributeError(
            "CARLA measurement is missing 'frame'"
        )

    frame_id = measurement.frame

    if frame_id is None:
        raise ValueError(
            "CARLA measurement frame must not be None"
        )

    return frame_id


def _measurement_timestamp(
    measurement: Any,
) -> float:
    """
    Extract and validate the CARLA measurement timestamp.
    """
    if not hasattr(
        measurement,
        "timestamp",
    ):
        raise AttributeError(
            "CARLA measurement is missing 'timestamp'"
        )

    timestamp = float(
        measurement.timestamp
    )

    if not np.isfinite(timestamp):
        raise ValueError(
            "CARLA measurement timestamp must be finite"
        )

    return timestamp


def carla_lidar_to_sensor_frame(
    measurement: Any,
    vehicle: Any,
    semantic_labels: np.ndarray | None = None,
    dynamic_probability: np.ndarray | None = None,
    config: CarlaSensorInterfaceConfig | None = None,
) -> SensorFrame:
    """
    Convert one CARLA LiDAR measurement into a SensorFrame.

    Parameters
    ----------
    measurement:
        CARLA-style LiDAR measurement exposing raw_data, frame and timestamp.

    vehicle:
        CARLA-style vehicle actor exposing velocity, angular velocity and
        transform.

    semantic_labels:
        Optional per-point semantic labels.

    dynamic_probability:
        Optional per-point dynamic probabilities.

    config:
        Optional CARLA adapter configuration.

    Returns
    -------
    SensorFrame
        Canonical FoveaMap sensor frame.
    """
    if measurement is None:
        raise TypeError(
            "measurement must not be None"
        )

    if not hasattr(
        measurement,
        "raw_data",
    ):
        raise AttributeError(
            "CARLA measurement is missing 'raw_data'"
        )

    xyz = parse_carla_lidar_raw_data(
        measurement.raw_data,
        config=config,
    )

    frame_id = _measurement_frame_id(
        measurement
    )

    timestamp = _measurement_timestamp(
        measurement
    )

    vehicle_state = carla_vehicle_state(
        vehicle
    )

    if semantic_labels is not None:
        semantic_labels = np.asarray(
            semantic_labels
        )

        if semantic_labels.shape != (
            xyz.shape[0],
        ):
            raise ValueError(
                "semantic_labels must contain exactly "
                "one value per LiDAR point"
            )

    if dynamic_probability is not None:
        dynamic_probability = np.asarray(
            dynamic_probability,
            dtype=np.float64,
        )

        if dynamic_probability.shape != (
            xyz.shape[0],
        ):
            raise ValueError(
                "dynamic_probability must contain exactly "
                "one value per LiDAR point"
            )

    return SensorFrame(
        xyz=xyz,
        vehicle_state=vehicle_state,
        timestamp=timestamp,
        frame_id=frame_id,
        semantic_labels=semantic_labels,
        dynamic_probability=dynamic_probability,
    )


def carla_measurement_to_sensor_frame(
    measurement: Any,
    vehicle: Any,
    semantic_labels: np.ndarray | None = None,
    dynamic_probability: np.ndarray | None = None,
    config: CarlaSensorInterfaceConfig | None = None,
) -> SensorFrame:
    """
    Compatibility alias for ``carla_lidar_to_sensor_frame``.
    """
    return carla_lidar_to_sensor_frame(
        measurement=measurement,
        vehicle=vehicle,
        semantic_labels=semantic_labels,
        dynamic_probability=dynamic_probability,
        config=config,
    )