from pathlib import Path
import sys

import numpy as np
import pytest

# Add project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mapping.kinematic_foveation import VehicleState
from mapping.semantic_kitti_adapter import (
    load_semantickitti_frame,
    load_semantickitti_scan,
)


TEST_SCAN = PROJECT_ROOT / "datasets" / "TEST" / "000000.bin"


def test_load_real_semantickitti_scan():
    xyz = load_semantickitti_scan(TEST_SCAN)

    assert xyz.shape == (124668, 3)
    assert xyz.dtype == np.float64
    assert np.all(np.isfinite(xyz))


def test_real_scan_preserves_expected_bounds():
    xyz = load_semantickitti_scan(TEST_SCAN)

    np.testing.assert_allclose(
        xyz.min(axis=0),
        [-78.087395, -55.72341, -11.556541],
        rtol=0.0,
        atol=1e-5,
    )

    np.testing.assert_allclose(
        xyz.max(axis=0),
        [77.96733, 44.878613, 2.8253412],
        rtol=0.0,
        atol=1e-5,
    )


def test_missing_scan_raises():
    with pytest.raises(FileNotFoundError):
        load_semantickitti_scan(
            PROJECT_ROOT / "datasets" / "TEST" / "does_not_exist.bin"
        )


def test_directory_path_raises(tmp_path):
    with pytest.raises(ValueError):
        load_semantickitti_scan(tmp_path)


def test_invalid_scan_size_raises(tmp_path):
    path = tmp_path / "invalid.bin"

    np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float32,
    ).tofile(path)

    with pytest.raises(ValueError, match="not divisible by 4"):
        load_semantickitti_scan(path)


def test_nonfinite_xyz_raises(tmp_path):
    path = tmp_path / "nonfinite.bin"

    raw = np.array(
        [
            1.0,
            2.0,
            3.0,
            0.5,
            np.nan,
            2.0,
            3.0,
            0.5,
        ],
        dtype=np.float32,
    )

    raw.tofile(path)

    with pytest.raises(ValueError, match="non-finite"):
        load_semantickitti_scan(path)


def test_frame_creation_from_real_scan():
    frame = load_semantickitti_frame(
        TEST_SCAN,
        timestamp=12.5,
        frame_id="000000",
    )

    assert frame.num_points == 124668
    assert frame.xyz.shape == (124668, 3)
    assert frame.xyz.dtype == np.float64

    assert frame.timestamp == 12.5
    assert frame.frame_id == "000000"

    assert frame.has_semantics is False
    assert frame.has_dynamic_probability is False


def test_frame_uses_default_stationary_vehicle_state():
    frame = load_semantickitti_frame(TEST_SCAN)

    assert isinstance(frame.vehicle_state, VehicleState)
    assert frame.vehicle_state.speed == 0.0
    assert frame.vehicle_state.yaw_rate == 0.0
    assert frame.vehicle_state.heading == 0.0


def test_frame_accepts_vehicle_state():
    state = VehicleState(
        speed=8.0,
        yaw_rate=0.2,
        heading=1.0,
    )

    frame = load_semantickitti_frame(
        TEST_SCAN,
        vehicle_state=state,
    )

    assert frame.vehicle_state is state


def test_frame_preserves_xyz_values():
    xyz = load_semantickitti_scan(TEST_SCAN)

    frame = load_semantickitti_frame(TEST_SCAN)

    np.testing.assert_array_equal(frame.xyz, xyz)


def test_frame_xyz_is_read_only():
    frame = load_semantickitti_frame(TEST_SCAN)

    assert frame.xyz.flags.writeable is False

    with pytest.raises(ValueError):
        frame.xyz[0, 0] = 999.0


def test_frame_signal_conversion_is_empty():
    frame = load_semantickitti_frame(TEST_SCAN)

    signals = frame.to_foveamap_signals()

    assert signals == {}


def test_timestamp_is_preserved():
    frame = load_semantickitti_frame(
        TEST_SCAN,
        timestamp=123.456,
    )

    assert frame.timestamp == 123.456


def test_frame_id_is_preserved():
    frame = load_semantickitti_frame(
        TEST_SCAN,
        frame_id=("sequence", 0),
    )

    assert frame.frame_id == ("sequence", 0)