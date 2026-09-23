
import json

import numpy as np

from backend.pipeline import MappingBackend
from backend.adapters.frontend_output import FrontendOutputAdapter
from mapping.kinematic_foveation import VehicleState
from mapping.sensor_frame import SensorFrame


def make_frame(
    frame_id: str = "backend-test-001",
    timestamp: float = 1.0,
) -> SensorFrame:

    xyz = np.array(
        [
            [1.0, 0.0, 0.5],
            [1.2, 0.1, 0.6],
            [2.0, 0.2, 0.7],
            [5.0, 1.0, 1.0],
            [10.0, 2.0, 1.2],
            [20.0, 3.0, 1.5],
        ],
        dtype=np.float64,
    )

    vehicle = VehicleState(
        speed=5.0,
        yaw_rate=0.1,
        heading=0.0,
    )

    return SensorFrame(
        xyz=xyz,
        vehicle_state=vehicle,
        timestamp=timestamp,
        frame_id=frame_id,
    )


def test_mapping_backend_processes_sensor_frame():

    backend = MappingBackend()

    output = backend.process(
        make_frame()
    )

    assert output.schema_version == "1.0"
    assert output.frame_id == "backend-test-001"
    assert output.timestamp == 1.0

    assert output.vehicle == {
        "speed": 5.0,
        "yaw_rate": 0.1,
        "heading": 0.0,
    }

    assert output.map["input_points"] == 6
    assert output.map["active_cells"] == 6

    assert len(output.map["levels"]) == 6
    assert len(output.map["resolutions"]) == 6
    assert len(output.map["point_count"]) == 6

    assert sum(output.map["point_count"]) == 6

    assert output.objects == []

    assert output.metrics["total_ms"] >= 0.0
    assert output.metrics["controller_ms"] >= 0.0
    assert output.metrics["mapper_ms"] >= 0.0

    assert output.diagnostics["input_points"] == 6
    assert output.diagnostics["active_cells"] == 6


def test_mapping_backend_serializes_for_frontend():

    backend = MappingBackend()

    output = backend.process(
        make_frame()
    )

    serializer = FrontendOutputAdapter()

    payload = serializer.serialize(
        output
    )

    decoded = json.loads(payload)

    assert decoded["schema_version"] == "1.0"
    assert decoded["frame_id"] == "backend-test-001"

    assert decoded["map"]["input_points"] == 6
    assert decoded["map"]["active_cells"] == 6

    assert len(decoded["map"]["levels"]) == 6


def test_mapping_backend_reset():

    backend = MappingBackend()

    backend.process(
        make_frame(
            frame_id="frame-001",
            timestamp=1.0,
        )
    )

    assert backend.previous_frame is not None

    backend.reset()

    assert backend.previous_frame is None


def test_mapping_backend_temporal_processing():

    backend = MappingBackend()

    first = backend.process(
        make_frame(
            frame_id="frame-001",
            timestamp=1.0,
        )
    )

    second = backend.process(
        make_frame(
            frame_id="frame-002",
            timestamp=2.0,
        )
    )

    assert first.metrics["delta_time"] is None
    assert second.metrics["delta_time"] == 1.0

    assert second.diagnostics["has_previous_frame"] is True
