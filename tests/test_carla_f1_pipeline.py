from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from perception.carla_pipeline import (
    CarlaF1MappingPipeline,
    map_f1_to_backend_semantics,
    parse_carla_lidar_for_perception,
)
from perception.inference import PerceptionResult


def make_result(
    xyz: np.ndarray,
    labels: np.ndarray,
) -> PerceptionResult:
    labels = np.asarray(labels, dtype=np.int64)

    model_predictions = labels - 1

    logits = np.zeros(
        (len(labels), 24),
        dtype=np.float32,
    )

    logits[
        np.arange(len(labels)),
        model_predictions,
    ] = 10.0

    return PerceptionResult(
        xyz=np.asarray(xyz, dtype=np.float32),
        unified_labels=labels,
        confidence=np.full(
            len(labels),
            0.95,
            dtype=np.float32,
        ),
        model_predictions=model_predictions,
        point_logits=logits,
    )


def test_carla_parser_preserves_intensity():
    points = np.array(
        [
            [1.0, 2.0, 3.0, 0.25],
            [4.0, 5.0, 6.0, 0.75],
        ],
        dtype=np.float32,
    )

    parsed = parse_carla_lidar_for_perception(
        points.tobytes()
    )

    np.testing.assert_allclose(
        parsed.xyz,
        points[:, :3],
    )

    np.testing.assert_allclose(
        parsed.intensity,
        points[:, 3],
    )


def test_unmapped_f1_classes_become_unknown():
    labels = np.arange(
        1,
        25,
        dtype=np.int64,
    )

    mapped = map_f1_to_backend_semantics(labels)

    np.testing.assert_array_equal(
        mapped,
        np.zeros(24, dtype=np.int64),
    )


def test_explicit_semantic_mapping():
    labels = np.array(
        [1, 2, 3, 4, 5, 6],
        dtype=np.int64,
    )

    mapped = map_f1_to_backend_semantics(
        labels,
        {
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 5,
        },
    )

    np.testing.assert_array_equal(
        mapped,
        [1, 2, 3, 4, 5, 0],
    )


def test_invalid_backend_mapping_rejected():
    with pytest.raises(ValueError):
        map_f1_to_backend_semantics(
            np.array([1], dtype=np.int64),
            {1: 6},
        )


@dataclass
class FakeVector:
    x: float
    y: float
    z: float


@dataclass
class FakeRotation:
    yaw: float


@dataclass
class FakeTransform:
    rotation: FakeRotation


class FakeVehicle:
    def get_velocity(self):
        return FakeVector(
            x=5.0,
            y=0.0,
            z=0.0,
        )

    def get_angular_velocity(self):
        return FakeVector(
            x=0.0,
            y=0.0,
            z=1.0,
        )

    def get_transform(self):
        return FakeTransform(
            rotation=FakeRotation(
                yaw=0.0
            )
        )


class FakeMeasurement:
    def __init__(self, points):
        self.raw_data = np.asarray(
            points,
            dtype=np.float32,
        ).tobytes()

        self.frame = 123
        self.timestamp = 10.0


class FakePerception:
    def __init__(self):
        self.last_intensity = None
        self.last_dataset_id = None

    def predict(
        self,
        xyz,
        intensity,
        dataset_id,
        source_point_id=None,
    ):
        self.last_intensity = np.asarray(
            intensity
        ).copy()

        self.last_dataset_id = dataset_id

        return make_result(
            xyz,
            np.array(
                [1, 2, 3],
                dtype=np.int64,
            ),
        )


def test_full_carla_f1_backend_mock_path():
    points = np.array(
        [
            [1.0, 0.0, 0.5, 0.10],
            [2.0, 0.0, 0.5, 0.20],
            [3.0, 0.0, 0.5, 0.30],
        ],
        dtype=np.float32,
    )

    measurement = FakeMeasurement(points)
    vehicle = FakeVehicle()
    perception = FakePerception()

    pipeline = CarlaF1MappingPipeline(
        perception_engine=perception,
        class_mapping={
            1: 1,
            2: 4,
            3: 5,
        },
    )

    frame = pipeline.build_sensor_frame(
        measurement,
        vehicle,
    )

    np.testing.assert_array_equal(
        frame.semantic_labels,
        [1, 4, 5],
    )

    np.testing.assert_allclose(
        frame.semantic_confidence,
        [0.95, 0.95, 0.95],
    )

    np.testing.assert_allclose(
        perception.last_intensity,
        [0.10, 0.20, 0.30],
    )

    assert frame.frame_id == "123"
    assert frame.timestamp == 10.0

    output = pipeline.backend.process(frame)

    assert output.schema_version == "1.0"
    assert output.frame_id == "123"
    assert output.map["input_points"] == 3

