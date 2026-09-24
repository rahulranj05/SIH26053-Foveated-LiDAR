from __future__ import annotations

"""
CARLA -> F1 perception -> FoveaMap integration boundary.

This module intentionally keeps:
    - CARLA parsing
    - neural perception
    - semantic ontology adaptation
    - mapping/backend processing

as separate concerns.

F1 produces unified semantic IDs 1..24.
FoveaMap consumes coarse semantic IDs 0..5.

Until the authoritative historical 24-class taxonomy is recovered,
unmapped unified IDs safely become UNKNOWN (0). A verified mapping can
be supplied explicitly without modifying F1 or MappingBackend.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np

from backend.pipeline import MappingBackend
from mapping.carla_sensor_interface import (
    CarlaSensorInterfaceConfig,
    carla_lidar_to_sensor_frame,
)
from mapping.sensor_frame import SensorFrame
from perception.inference import PerceptionResult


UNKNOWN = 0
DRIVABLE = 1
NON_DRIVABLE = 2
STATIC_OBSTACLE = 3
VEHICLE = 4
VULNERABLE_USER = 5

VALID_BACKEND_CLASSES = frozenset(range(6))
VALID_F1_CLASSES = frozenset(range(1, 25))


class PerceptionEngine(Protocol):
    def predict(
        self,
        xyz: np.ndarray,
        intensity: np.ndarray,
        dataset_id: str,
        source_point_id: np.ndarray | None = None,
    ) -> PerceptionResult:
        ...


@dataclass(frozen=True)
class CarlaPerceptionInput:
    xyz: np.ndarray
    intensity: np.ndarray

    def __post_init__(self) -> None:
        xyz = np.asarray(self.xyz, dtype=np.float32)
        intensity = np.asarray(self.intensity, dtype=np.float32)

        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz must have shape (N, 3)")

        if intensity.ndim != 1:
            raise ValueError("intensity must have shape (N,)")

        if len(xyz) != len(intensity):
            raise ValueError(
                "xyz and intensity must contain the same number of points"
            )

        if not np.all(np.isfinite(xyz)):
            raise ValueError("xyz must contain only finite values")

        if not np.all(np.isfinite(intensity)):
            raise ValueError("intensity must contain only finite values")

        object.__setattr__(
            self,
            "xyz",
            np.array(xyz, dtype=np.float32, copy=True),
        )
        object.__setattr__(
            self,
            "intensity",
            np.array(intensity, dtype=np.float32, copy=True),
        )


def parse_carla_lidar_for_perception(
    raw_data: bytes | bytearray | memoryview,
    config: CarlaSensorInterfaceConfig | None = None,
) -> CarlaPerceptionInput:
    """
    Preserve CARLA XYZ + intensity for F1.

    The existing mapping CARLA parser remains unchanged because its public
    contract is XYZ-only.
    """

    if config is None:
        config = CarlaSensorInterfaceConfig()

    if not isinstance(raw_data, (bytes, bytearray, memoryview)):
        raise TypeError("raw_data must be bytes-like")

    raw = np.frombuffer(raw_data, dtype=np.float32)

    if raw.size == 0:
        return CarlaPerceptionInput(
            xyz=np.empty((0, 3), dtype=np.float32),
            intensity=np.empty((0,), dtype=np.float32),
        )

    if raw.size % config.lidar_point_stride != 0:
        raise ValueError(
            "CARLA LiDAR raw_data size is not divisible by lidar_point_stride"
        )

    if config.lidar_point_stride < 4:
        raise ValueError(
            "F1 perception requires CARLA LiDAR stride >= 4 "
            "for x, y, z, intensity"
        )

    points = raw.reshape(-1, config.lidar_point_stride)

    return CarlaPerceptionInput(
        xyz=points[:, :3],
        intensity=points[:, 3],
    )


def map_f1_to_backend_semantics(
    unified_labels: np.ndarray,
    class_mapping: Mapping[int, int] | None = None,
) -> np.ndarray:
    """
    Convert F1 unified IDs 1..24 to FoveaMap IDs 0..5.

    Unmapped classes intentionally become UNKNOWN=0.

    No guessed historical taxonomy is embedded here.
    """

    labels = np.asarray(unified_labels)

    if labels.ndim != 1:
        raise ValueError("unified_labels must be a 1D array")

    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("unified_labels must contain integers")

    if labels.size:
        if np.any(labels < 1) or np.any(labels > 24):
            raise ValueError(
                "F1 unified semantic labels must be in the range 1..24"
            )

    mapping = dict(class_mapping or {})

    for source_id, target_id in mapping.items():
        if source_id not in VALID_F1_CLASSES:
            raise ValueError(
                f"invalid F1 unified class ID: {source_id}"
            )

        if target_id not in VALID_BACKEND_CLASSES:
            raise ValueError(
                f"invalid FoveaMap semantic class ID: {target_id}"
            )

    output = np.full(
        labels.shape,
        UNKNOWN,
        dtype=np.int64,
    )

    for source_id, target_id in mapping.items():
        output[labels == source_id] = target_id

    return output


def perception_result_to_sensor_frame(
    base_frame: SensorFrame,
    perception: PerceptionResult,
    class_mapping: Mapping[int, int] | None = None,
) -> SensorFrame:
    """
    Attach F1 predictions to an existing canonical CARLA SensorFrame.
    """

    perception.validate()

    if perception.xyz.shape[0] != base_frame.num_points:
        raise ValueError(
            "perception point count must match CARLA SensorFrame point count"
        )

    if not np.allclose(
        np.asarray(perception.xyz, dtype=np.float64),
        base_frame.xyz,
        rtol=0.0,
        atol=1e-5,
    ):
        raise ValueError(
            "perception XYZ does not match the CARLA SensorFrame XYZ"
        )

    backend_labels = map_f1_to_backend_semantics(
        perception.unified_labels,
        class_mapping=class_mapping,
    )

    return SensorFrame(
        xyz=base_frame.xyz,
        vehicle_state=base_frame.vehicle_state,
        timestamp=base_frame.timestamp,
        frame_id=base_frame.frame_id,
        semantic_labels=backend_labels,
        semantic_confidence=perception.confidence,
        dynamic_probability=base_frame.dynamic_probability,
    )


class CarlaF1MappingPipeline:
    """
    End-to-end integration boundary:

        CARLA
          -> F1
          -> semantic adapter
          -> SensorFrame
          -> MappingBackend
    """

    def __init__(
        self,
        perception_engine: PerceptionEngine,
        *,
        class_mapping: Mapping[int, int] | None = None,
        backend: MappingBackend | None = None,
        carla_config: CarlaSensorInterfaceConfig | None = None,
        dataset_id: str = "SemanticKITTI",
    ) -> None:
        self.perception_engine = perception_engine
        self.class_mapping = dict(class_mapping or {})
        self.backend = backend or MappingBackend()
        self.carla_config = carla_config or CarlaSensorInterfaceConfig()

        # F1 preprocessing currently requires one of its known projection
        # geometries. This is configurable rather than hidden.
        self.dataset_id = dataset_id

    def build_sensor_frame(
        self,
        measurement: Any,
        vehicle: Any,
    ) -> SensorFrame:

        perception_input = parse_carla_lidar_for_perception(
            measurement.raw_data,
            config=self.carla_config,
        )

        raw_base_frame = carla_lidar_to_sensor_frame(
            measurement=measurement,
            vehicle=vehicle,
            config=self.carla_config,
        )

        # BackendOutput schema v1.0 requires a non-empty string frame_id.
        # CARLA normally exposes its frame counter as an integer, so normalize
        # it at this integration boundary without changing the canonical
        # CARLA adapter or backend contract.
        base_frame = SensorFrame(
            xyz=raw_base_frame.xyz,
            vehicle_state=raw_base_frame.vehicle_state,
            timestamp=raw_base_frame.timestamp,
            frame_id=str(raw_base_frame.frame_id),
            semantic_labels=raw_base_frame.semantic_labels,
            dynamic_probability=raw_base_frame.dynamic_probability,
            semantic_confidence=raw_base_frame.semantic_confidence,
        )

        perception = self.perception_engine.predict(
            xyz=perception_input.xyz,
            intensity=perception_input.intensity,
            dataset_id=self.dataset_id,
        )

        return perception_result_to_sensor_frame(
            base_frame=base_frame,
            perception=perception,
            class_mapping=self.class_mapping,
        )

    def process(
        self,
        measurement: Any,
        vehicle: Any,
    ):
        frame = self.build_sensor_frame(
            measurement=measurement,
            vehicle=vehicle,
        )

        return self.backend.process(frame)

    def reset(self) -> None:
        self.backend.reset()

