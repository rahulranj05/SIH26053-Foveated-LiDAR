import json

import numpy as np
import pytest

from backend.adapters.frontend_output import FrontendOutputAdapter
from backend.contracts.backend_output import BackendOutput


def make_valid_output():
    return BackendOutput(
        schema_version="1.0",
        frame_id="frame_000001",
        timestamp=1.5,
        vehicle={
            "speed": np.float32(5.0),
            "heading": np.float64(0.25),
        },
        map={
            "values": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        },
        objects=[],
        metrics={
            "latency_ms": np.float64(12.5),
        },
        diagnostics={
            "status": "ok",
        },
    )


def test_backend_output_serializes():
    adapter = FrontendOutputAdapter()

    output = make_valid_output()
    payload = adapter.serialize(output)

    data = json.loads(payload)

    assert data["schema_version"] == "1.0"
    assert data["frame_id"] == "frame_000001"
    assert data["timestamp"] == 1.5
    assert data["vehicle"]["speed"] == 5.0
    assert data["map"]["values"] == [1.0, 2.0, 3.0]


def test_numpy_values_are_converted():
    adapter = FrontendOutputAdapter()

    output = make_valid_output()
    payload = adapter.serialize(output)

    # If this succeeds, the final payload is valid JSON.
    data = json.loads(payload)

    assert isinstance(data["vehicle"]["speed"], float)
    assert isinstance(data["map"]["values"], list)


def test_nan_is_rejected():
    adapter = FrontendOutputAdapter()

    output = make_valid_output()
    output.metrics["bad_value"] = np.nan

    with pytest.raises(ValueError):
        adapter.serialize(output)


def test_infinity_is_rejected():
    adapter = FrontendOutputAdapter()

    output = make_valid_output()
    output.metrics["bad_value"] = np.inf

    with pytest.raises(ValueError):
        adapter.serialize(output)


def test_wrong_schema_version_is_rejected():
    with pytest.raises(ValueError):
        BackendOutput(
            schema_version="2.0",
            frame_id="frame_000001",
            timestamp=1.5,
            vehicle={},
            map={},
            objects=[],
            metrics={},
            diagnostics={},
        )