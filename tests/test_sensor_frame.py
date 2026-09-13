"""
A21 — Canonical Sensor Frame tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mapping.kinematic_foveation import VehicleState
from mapping.sensor_frame import SensorFrame


def make_xyz(n: int = 8) -> np.ndarray:
    return np.column_stack(
        [
            np.linspace(1.0, 20.0, n),
            np.linspace(-2.0, 2.0, n),
            np.linspace(0.0, 1.0, n),
        ]
    )


def make_vehicle_state() -> VehicleState:
    return VehicleState(
        speed=5.0,
        yaw_rate=0.1,
        heading=0.25,
    )


def test_sensor_frame_constructs() -> None:
    frame = SensorFrame(
        xyz=make_xyz(),
        vehicle_state=make_vehicle_state(),
        timestamp=12.5,
        frame_id=42,
    )

    assert frame.num_points == 8
    assert frame.timestamp == 12.5
    assert frame.frame_id == 42
    assert frame.vehicle_state.speed == 5.0


def test_xyz_is_normalized_to_float64() -> None:
    xyz = make_xyz().astype(np.float32)

    frame = SensorFrame(
        xyz=xyz,
        vehicle_state=make_vehicle_state(),
        timestamp=1.0,
        frame_id=1,
    )

    assert frame.xyz.dtype == np.float64


def test_xyz_is_copied() -> None:
    xyz = make_xyz()

    frame = SensorFrame(
        xyz=xyz,
        vehicle_state=make_vehicle_state(),
        timestamp=1.0,
        frame_id=1,
    )

    xyz[0, 0] = 999.0

    assert frame.xyz[0, 0] != 999.0


def test_xyz_is_read_only() -> None:
    frame = SensorFrame(
        xyz=make_xyz(),
        vehicle_state=make_vehicle_state(),
        timestamp=1.0,
        frame_id=1,
    )

    assert frame.xyz.flags.writeable is False

    with pytest.raises(ValueError):
        frame.xyz[0, 0] = 999.0


def test_invalid_xyz_shape_raises() -> None:
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        SensorFrame(
            xyz=np.zeros((8, 2)),
            vehicle_state=make_vehicle_state(),
            timestamp=1.0,
            frame_id=1,
        )


def test_one_dimensional_xyz_raises() -> None:
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        SensorFrame(
            xyz=np.zeros(24),
            vehicle_state=make_vehicle_state(),
            timestamp=1.0,
            frame_id=1,
        )


def test_nonfinite_xyz_raises() -> None:
    xyz = make_xyz()
    xyz[2, 1] = np.nan

    with pytest.raises(ValueError, match="finite"):
        SensorFrame(
            xyz=xyz,
            vehicle_state=make_vehicle_state(),
            timestamp=1.0,
            frame_id=1,
        )


def test_invalid_timestamp_raises() -> None:
    with pytest.raises(ValueError, match="timestamp"):
        SensorFrame(
            xyz=make_xyz(),
            vehicle_state=make_vehicle_state(),
            timestamp=np.nan,
            frame_id=1,
        )


def test_invalid_vehicle_state_type_raises() -> None:
    with pytest.raises(TypeError, match="VehicleState"):
        SensorFrame(
            xyz=make_xyz(),
            vehicle_state=None,  # type: ignore[arg-type]
            timestamp=1.0,
            frame_id=1,
        )


def test_semantic_labels_are_supported() -> None:
    labels = np.arange(8)

    frame = SensorFrame(
        xyz=make_xyz(),
        semantic_labels=labels,
        vehicle_state=make_vehicle_state(),
        timestamp=1.0,
        frame_id=1,
    )

    assert frame.has_semantics
    assert frame.semantic_labels is not None
    assert frame.semantic_labels.dtype == np.int64
    assert np.array_equal(frame.semantic_labels, labels)


def test_semantic_labels_are_copied() -> None:
    labels = np.arange(8)

    frame = SensorFrame(
        xyz=make_xyz(),
        semantic_labels=labels,
        vehicle_state=make_vehicle_state(),
        timestamp=1.0,
        frame_id=1,
    )

    labels[0] = 99

    assert frame.semantic_labels is not None
    assert frame.semantic_labels[0] != 99


def test_semantic_labels_are_read_only() -> None:
    frame = SensorFrame(
        xyz=make_xyz(),
        semantic_labels=np.arange(8),
        vehicle_state=make_vehicle_state(),
        timestamp=1.0,
        frame_id=1,
    )

    assert frame.semantic_labels is not None
    assert frame.semantic_labels.flags.writeable is False

    with pytest.raises(ValueError):
        frame.semantic_labels[0] = 99


def test_semantic_label_length_mismatch_raises() -> None:
    with pytest.raises(
        ValueError,
        match="semantic_labels length",
    ):
        SensorFrame(
            xyz=make_xyz(8),
            semantic_labels=np.arange(7),
            vehicle_state=make_vehicle_state(),
            timestamp=1.0,
            frame_id=1,
        )


def test_semantic_labels_must_be_one_dimensional() -> None:
    with pytest.raises(
        ValueError,
        match="semantic_labels must be a 1D",
    ):
        SensorFrame(
            xyz=make_xyz(8),
            semantic_labels=np.zeros((8, 1)),
            vehicle_state=make_vehicle_state(),
            timestamp=1.0,
            frame_id=1,
        )


def test_semantic_labels_must_be_integer_valued() -> None:
    labels = np.arange(8, dtype=np.float64)
    labels[2] = 1.5

    with pytest.raises(
        ValueError,
        match="integer values",
    ):
        SensorFrame(
            xyz=make_xyz(),
            semantic_labels=labels,
            vehicle_state=make_vehicle_state(),
            timestamp=1.0,
            frame_id=1,
        )


def test_dynamic_probability_is_supported() -> None:
    probability = np.linspace(0.0, 1.0, 8)

    frame = SensorFrame(
        xyz=make_xyz(),
        dynamic_probability=probability,
        vehicle_state=make_vehicle_state(),
        timestamp=1.0,
        frame_id=1,
    )

    assert frame.has_dynamic_probability
    assert frame.dynamic_probability is not None
    assert frame.dynamic_probability.dtype == np.float64
    assert np.allclose(frame.dynamic_probability, probability)


def test_dynamic_probability_length_mismatch_raises() -> None:
    with pytest.raises(
        ValueError,
        match="dynamic_probability length",
    ):
        SensorFrame(
            xyz=make_xyz(8),
            dynamic_probability=np.zeros(7),
            vehicle_state=make_vehicle_state(),
            timestamp=1.0,
            frame_id=1,
        )


def test_dynamic_probability_must_be_in_zero_one() -> None:
    probability = np.zeros(8)
    probability[3] = 1.1

    with pytest.raises(
        ValueError,
        match=r"\[0, 1\]",
    ):
        SensorFrame(
            xyz=make_xyz(),
            dynamic_probability=probability,
            vehicle_state=make_vehicle_state(),
            timestamp=1.0,
            frame_id=1,
        )


def test_dynamic_probability_must_be_finite() -> None:
    probability = np.zeros(8)
    probability[3] = np.inf

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        SensorFrame(
            xyz=make_xyz(),
            dynamic_probability=probability,
            vehicle_state=make_vehicle_state(),
            timestamp=1.0,
            frame_id=1,
        )


def test_missing_optional_signals_are_supported() -> None:
    frame = SensorFrame(
        xyz=make_xyz(),
        vehicle_state=make_vehicle_state(),
        timestamp=1.0,
        frame_id=1,
    )

    assert frame.has_semantics is False
    assert frame.has_dynamic_probability is False
    assert frame.to_foveamap_signals() == {}


def test_to_foveamap_signals_contains_semantics() -> None:
    labels = np.array([0, 1, 2, 3, 4, 5, 1, 0])

    frame = SensorFrame(
        xyz=make_xyz(),
        semantic_labels=labels,
        vehicle_state=make_vehicle_state(),
        timestamp=1.0,
        frame_id=1,
    )

    signals = frame.to_foveamap_signals()

    assert set(signals) == {"semantic"}
    assert np.array_equal(
        signals["semantic"],
        labels.astype(np.float64),
    )


def test_to_foveamap_signals_contains_dynamic_probability() -> None:
    probability = np.linspace(0.0, 1.0, 8)

    frame = SensorFrame(
        xyz=make_xyz(),
        dynamic_probability=probability,
        vehicle_state=make_vehicle_state(),
        timestamp=1.0,
        frame_id=1,
    )

    signals = frame.to_foveamap_signals()

    assert set(signals) == {"dynamic"}
    assert np.allclose(
        signals["dynamic"],
        probability,
    )


def test_to_foveamap_signals_contains_all_available_signals() -> None:
    labels = np.array([0, 1, 2, 3, 4, 5, 1, 0])
    probability = np.linspace(0.0, 1.0, 8)

    frame = SensorFrame(
        xyz=make_xyz(),
        semantic_labels=labels,
        dynamic_probability=probability,
        vehicle_state=make_vehicle_state(),
        timestamp=1.0,
        frame_id="frame-001",
    )

    signals = frame.to_foveamap_signals()

    assert set(signals) == {"semantic", "dynamic"}
    assert np.array_equal(
        signals["semantic"],
        labels.astype(np.float64),
    )
    assert np.allclose(
        signals["dynamic"],
        probability,
    )


def test_copy_is_independent() -> None:
    frame = SensorFrame(
        xyz=make_xyz(),
        semantic_labels=np.arange(8),
        dynamic_probability=np.linspace(0.0, 1.0, 8),
        vehicle_state=make_vehicle_state(),
        timestamp=5.0,
        frame_id="original",
    )

    copied = frame.copy()

    assert copied is not frame
    assert copied.frame_id == frame.frame_id
    assert copied.timestamp == frame.timestamp
    assert copied.vehicle_state == frame.vehicle_state
    assert np.array_equal(copied.xyz, frame.xyz)
    assert np.array_equal(
        copied.semantic_labels,
        frame.semantic_labels,
    )
    assert np.array_equal(
        copied.dynamic_probability,
        frame.dynamic_probability,
    )

    assert copied.xyz is not frame.xyz
    assert copied.semantic_labels is not frame.semantic_labels
    assert copied.dynamic_probability is not frame.dynamic_probability


def test_frame_id_can_be_string() -> None:
    frame = SensorFrame(
        xyz=make_xyz(),
        vehicle_state=make_vehicle_state(),
        timestamp=3.0,
        frame_id="000123",
    )

    assert frame.frame_id == "000123"


def test_empty_frame_is_supported() -> None:
    frame = SensorFrame(
        xyz=np.empty((0, 3)),
        vehicle_state=make_vehicle_state(),
        timestamp=0.0,
        frame_id=0,
    )

    assert frame.num_points == 0
    assert frame.xyz.shape == (0, 3)
    assert frame.to_foveamap_signals() == {}