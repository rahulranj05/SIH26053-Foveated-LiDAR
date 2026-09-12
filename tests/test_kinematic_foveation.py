import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mapping.kinematic_foveation import (
    KinematicFoveationConfig,
    VehicleState,
    forward_range_for_speed,
    kinematic_importance,
    kinematic_resolution,
    normalize_speed,
    resolution_from_kinematic_importance,
    transform_to_vehicle_frame,
    yaw_shift_for_distance,
)


def test_vehicle_state_accepts_valid_values():
    state = VehicleState(
        speed=5.0,
        yaw_rate=0.2,
        heading=1.0,
    )

    assert state.speed == 5.0
    assert state.yaw_rate == 0.2
    assert state.heading == 1.0


def test_vehicle_state_rejects_negative_speed():
    with pytest.raises(ValueError):
        VehicleState(
            speed=-1.0,
            yaw_rate=0.0,
            heading=0.0,
        )


def test_vehicle_state_rejects_non_finite_values():
    with pytest.raises(ValueError):
        VehicleState(
            speed=np.nan,
            yaw_rate=0.0,
            heading=0.0,
        )

    with pytest.raises(ValueError):
        VehicleState(
            speed=0.0,
            yaw_rate=np.inf,
            heading=0.0,
        )

    with pytest.raises(ValueError):
        VehicleState(
            speed=0.0,
            yaw_rate=0.0,
            heading=np.nan,
        )


def test_normalize_speed():
    assert normalize_speed(0.0, 10.0) == 0.0
    assert normalize_speed(5.0, 10.0) == 0.5
    assert normalize_speed(10.0, 10.0) == 1.0
    assert normalize_speed(20.0, 10.0) == 1.0


def test_normalize_speed_rejects_invalid_values():
    with pytest.raises(ValueError):
        normalize_speed(-1.0, 10.0)

    with pytest.raises(ValueError):
        normalize_speed(1.0, 0.0)

    with pytest.raises(ValueError):
        normalize_speed(np.nan, 10.0)


def test_forward_range_increases_with_speed():
    config = KinematicFoveationConfig()

    low_speed_range = forward_range_for_speed(
        0.0,
        config,
    )

    high_speed_range = forward_range_for_speed(
        10.0,
        config,
    )

    assert low_speed_range == config.minimum_forward_range
    assert high_speed_range == config.maximum_forward_range
    assert high_speed_range > low_speed_range


def test_transform_to_vehicle_frame_zero_heading():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])

    forward, lateral = transform_to_vehicle_frame(
        x,
        y,
        0.0,
    )

    np.testing.assert_allclose(
        forward,
        x,
    )

    np.testing.assert_allclose(
        lateral,
        y,
    )


def test_transform_to_vehicle_frame_ninety_degrees():
    x = np.array([1.0])
    y = np.array([0.0])

    forward, lateral = transform_to_vehicle_frame(
        x,
        y,
        np.pi / 2.0,
    )

    np.testing.assert_allclose(
        forward,
        0.0,
        atol=1e-10,
    )

    np.testing.assert_allclose(
        lateral,
        -1.0,
        atol=1e-10,
    )


def test_yaw_shift():
    config = KinematicFoveationConfig(
        turn_gain=2.0,
    )

    forward_distance = np.array(
        [
            0.0,
            5.0,
            10.0,
        ]
    )

    shift = yaw_shift_for_distance(
        forward_distance,
        yaw_rate=0.5,
        config=config,
    )

    expected = np.array(
        [
            0.0,
            5.0,
            10.0,
        ]
    )

    np.testing.assert_allclose(
        shift,
        expected,
    )


def test_kinematic_importance_is_bounded():
    xyz = np.array(
        [
            [2.0, 0.0, 0.0],
            [5.0, 1.0, 0.0],
            [10.0, -2.0, 0.0],
            [20.0, 3.0, 0.0],
        ]
    )

    state = VehicleState(
        speed=5.0,
        yaw_rate=0.1,
        heading=0.0,
    )

    importance = kinematic_importance(
        xyz,
        state,
    )

    assert np.all(
        importance >= 0.0
    )

    assert np.all(
        importance <= 1.0
    )


def test_points_behind_vehicle_have_zero_importance():
    state = VehicleState(
        speed=5.0,
        yaw_rate=0.0,
        heading=0.0,
    )

    xyz = np.array(
        [
            [-1.0, 0.0, 0.0],
            [-10.0, 0.0, 0.0],
        ]
    )

    importance = kinematic_importance(
        xyz,
        state,
    )

    np.testing.assert_allclose(
        importance,
        0.0,
    )


def test_points_beyond_forward_range_have_zero_importance():
    state = VehicleState(
        speed=0.0,
        yaw_rate=0.0,
        heading=0.0,
    )

    xyz = np.array(
        [
            [10.0, 0.0, 0.0],
            [10.01, 0.0, 0.0],
        ]
    )

    importance = kinematic_importance(
        xyz,
        state,
    )

    assert importance[0] > 0.0
    assert importance[1] == 0.0


def test_higher_speed_extends_fovea():
    config = KinematicFoveationConfig()

    low_speed_state = VehicleState(
        speed=0.0,
        yaw_rate=0.0,
        heading=0.0,
    )

    high_speed_state = VehicleState(
        speed=10.0,
        yaw_rate=0.0,
        heading=0.0,
    )

    xyz = np.array(
        [
            [15.0, 0.0, 0.0],
        ]
    )

    low_speed_importance = kinematic_importance(
        xyz,
        low_speed_state,
        config,
    )

    high_speed_importance = kinematic_importance(
        xyz,
        high_speed_state,
        config,
    )

    assert low_speed_importance[0] == 0.0
    assert high_speed_importance[0] > 0.0


def test_turning_changes_foveation_corridor():
    straight_state = VehicleState(
        speed=5.0,
        yaw_rate=0.0,
        heading=0.0,
    )

    turning_state = VehicleState(
        speed=5.0,
        yaw_rate=0.2,
        heading=0.0,
    )

    xyz = np.array(
        [
            [5.0, 0.0, 0.0],
            [5.0, 5.0, 0.0],
        ]
    )

    straight_importance = kinematic_importance(
        xyz,
        straight_state,
    )

    turning_importance = kinematic_importance(
        xyz,
        turning_state,
    )

    assert not np.allclose(
        straight_importance,
        turning_importance,
    )


def test_resolution_thresholds():
    importance = np.array(
        [
            0.0,
            0.24,
            0.25,
            0.49,
            0.50,
            0.74,
            0.75,
            1.0,
        ]
    )

    resolution = resolution_from_kinematic_importance(
        importance
    )

    expected = np.array(
        [
            0.40,
            0.40,
            0.20,
            0.20,
            0.10,
            0.10,
            0.05,
            0.05,
        ]
    )

    np.testing.assert_allclose(
        resolution,
        expected,
    )


def test_kinematic_resolution_output_shape():
    xyz = np.array(
        [
            [1.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
        ]
    )

    state = VehicleState(
        speed=5.0,
        yaw_rate=0.0,
        heading=0.0,
    )

    resolution = kinematic_resolution(
        xyz,
        state,
    )

    assert resolution.shape == (4,)

    assert np.all(
        np.isin(
            resolution,
            [
                0.05,
                0.10,
                0.20,
                0.40,
            ],
        )
    )


def test_empty_point_cloud():
    xyz = np.empty(
        (0, 3),
        dtype=np.float64,
    )

    state = VehicleState(
        speed=5.0,
        yaw_rate=0.0,
        heading=0.0,
    )

    importance = kinematic_importance(
        xyz,
        state,
    )

    assert importance.shape == (0,)

    resolution = kinematic_resolution(
        xyz,
        state,
    )

    assert resolution.shape == (0,)


def test_invalid_xyz_shape():
    state = VehicleState(
        speed=5.0,
        yaw_rate=0.0,
        heading=0.0,
    )

    with pytest.raises(ValueError):
        kinematic_importance(
            np.array([1.0, 2.0, 3.0]),
            state,
        )


def test_invalid_xyz_values():
    state = VehicleState(
        speed=5.0,
        yaw_rate=0.0,
        heading=0.0,
    )

    xyz = np.array(
        [
            [1.0, 0.0, 0.0],
            [np.nan, 0.0, 0.0],
        ]
    )

    with pytest.raises(ValueError):
        kinematic_importance(
            xyz,
            state,
        )