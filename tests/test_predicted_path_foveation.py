import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mapping.predicted_path_foveation import (
    VehicleState,
    distance_to_path,
    path_importance,
    predict_path,
    resolution_from_path_importance,
)


def test_predict_path_straight():
    state = VehicleState(
        speed=10.0,
        yaw_rate=0.0,
    )

    path = predict_path(
        state,
        horizon=5.0,
        step=1.0,
    )

    assert path.shape == (6, 3)

    np.testing.assert_allclose(
        path[:, 0],
        np.array([0, 10, 20, 30, 40, 50]),
    )

    np.testing.assert_allclose(
        path[:, 1],
        0.0,
    )


def test_predict_path_turns():
    state = VehicleState(
        speed=10.0,
        yaw_rate=0.2,
    )

    path = predict_path(
        state,
        horizon=5.0,
        step=1.0,
    )

    assert path.shape == (6, 3)
    assert path[-1, 0] > 0.0
    assert path[-1, 1] > 0.0


def test_predict_path_negative_turn():
    state = VehicleState(
        speed=10.0,
        yaw_rate=-0.2,
    )

    path = predict_path(
        state,
        horizon=5.0,
        step=1.0,
    )

    assert path[-1, 0] > 0.0
    assert path[-1, 1] < 0.0


def test_turning_paths_are_symmetric():
    positive = predict_path(
        VehicleState(
            speed=10.0,
            yaw_rate=0.2,
        ),
        horizon=5.0,
        step=0.5,
    )

    negative = predict_path(
        VehicleState(
            speed=10.0,
            yaw_rate=-0.2,
        ),
        horizon=5.0,
        step=0.5,
    )

    np.testing.assert_allclose(
        positive[:, 0],
        negative[:, 0],
        atol=1e-10,
    )

    np.testing.assert_allclose(
        positive[:, 1],
        -negative[:, 1],
        atol=1e-10,
    )


def test_heading_rotates_path():
    state = VehicleState(
        speed=10.0,
        yaw_rate=0.0,
        heading=np.pi / 2,
    )

    path = predict_path(
        state,
        horizon=2.0,
        step=1.0,
    )

    np.testing.assert_allclose(
        path[:, 0],
        0.0,
        atol=1e-10,
    )

    np.testing.assert_allclose(
        path[:, 1],
        np.array([0, 10, 20]),
        atol=1e-10,
    )


def test_distance_to_straight_path():
    path = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
        ]
    )

    points = np.array(
        [
            [5.0, 0.0, 1.0],
            [5.0, 2.0, 0.0],
            [15.0, -3.0, 0.0],
        ]
    )

    distances = distance_to_path(
        points,
        path,
    )

    np.testing.assert_allclose(
        distances,
        np.array([0.0, 2.0, 3.0]),
    )


def test_points_on_path_have_maximum_importance():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ]
    )

    state = VehicleState(
        speed=5.0,
        yaw_rate=0.0,
    )

    importance = path_importance(
        points,
        state,
        horizon=3.0,
        step=0.25,
        corridor_width=3.0,
    )

    assert np.allclose(
        importance,
        1.0,
    )


def test_points_outside_corridor_have_zero_importance():
    points = np.array(
        [
            [5.0, 4.0, 0.0],
            [5.0, -4.0, 0.0],
        ]
    )

    state = VehicleState(
        speed=5.0,
        yaw_rate=0.0,
    )

    importance = path_importance(
        points,
        state,
        horizon=3.0,
        step=0.25,
        corridor_width=3.0,
    )

    assert np.allclose(
        importance,
        0.0,
    )


def test_importance_decreases_with_lateral_distance():
    points = np.array(
        [
            [5.0, 0.0, 0.0],
            [5.0, 1.0, 0.0],
            [5.0, 2.0, 0.0],
        ]
    )

    state = VehicleState(
        speed=5.0,
        yaw_rate=0.0,
    )

    importance = path_importance(
        points,
        state,
        horizon=3.0,
        step=0.25,
        corridor_width=3.0,
    )

    assert importance[0] > importance[1]
    assert importance[1] > importance[2]


def test_points_behind_vehicle_are_not_foveated():
    points = np.array(
        [
            [-5.0, 0.0, 0.0],
        ]
    )

    state = VehicleState(
        speed=5.0,
        yaw_rate=0.0,
    )

    importance = path_importance(
        points,
        state,
        horizon=3.0,
        step=0.25,
        corridor_width=3.0,
    )

    assert importance[0] == 0.0


def test_horizon_limits_path():
    state = VehicleState(
        speed=10.0,
        yaw_rate=0.0,
    )

    path = predict_path(
        state,
        horizon=3.0,
        step=0.5,
    )

    assert np.max(path[:, 0]) == pytest.approx(30.0)


def test_zero_speed_produces_stationary_path():
    state = VehicleState(
        speed=0.0,
        yaw_rate=0.5,
    )

    path = predict_path(
        state,
        horizon=5.0,
        step=0.5,
    )

    np.testing.assert_allclose(
        path,
        0.0,
        atol=1e-10,
    )


def test_resolution_mapping():
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

    resolutions = resolution_from_path_importance(
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
        resolutions,
        expected,
    )


def test_empty_cloud():
    xyz = np.empty(
        (0, 3),
        dtype=np.float64,
    )

    state = VehicleState(
        speed=5.0,
        yaw_rate=0.1,
    )

    importance = path_importance(
        xyz,
        state,
    )

    assert importance.shape == (0,)


def test_invalid_xyz_shape():
    xyz = np.zeros((10, 2))

    state = VehicleState(
        speed=5.0,
        yaw_rate=0.0,
    )

    with pytest.raises(ValueError):
        path_importance(
            xyz,
            state,
        )


def test_invalid_horizon():
    state = VehicleState(
        speed=5.0,
        yaw_rate=0.0,
    )

    with pytest.raises(ValueError):
        predict_path(
            state,
            horizon=0.0,
        )


def test_invalid_step():
    state = VehicleState(
        speed=5.0,
        yaw_rate=0.0,
    )

    with pytest.raises(ValueError):
        predict_path(
            state,
            step=0.0,
        )


def test_invalid_corridor_width():
    xyz = np.array(
        [[5.0, 0.0, 0.0]]
    )

    state = VehicleState(
        speed=5.0,
        yaw_rate=0.0,
    )

    with pytest.raises(ValueError):
        path_importance(
            xyz,
            state,
            corridor_width=0.0,
        )


def test_nonfinite_xyz_rejected():
    xyz = np.array(
        [[np.nan, 0.0, 0.0]]
    )

    state = VehicleState(
        speed=5.0,
        yaw_rate=0.0,
    )

    with pytest.raises(ValueError):
        path_importance(
            xyz,
            state,
        )


def test_nonfinite_importance_rejected():
    with pytest.raises(ValueError):
        resolution_from_path_importance(
            np.array([0.5, np.nan])
        )