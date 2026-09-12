import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mapping.vectorized_adaptive import (
    VectorizedAdaptiveMap,
    build_vectorized_adaptive_map,
    resolution_for_distance,
)


def test_resolution_for_distance():
    distance = np.array(
        [
            0.0,
            5.0,
            9.99,
            10.0,
            15.0,
            24.99,
            25.0,
            40.0,
            49.99,
            50.0,
            75.0,
            100.0,
            100.01,
        ]
    )

    result = resolution_for_distance(distance)

    expected = np.array(
        [
            0.05,
            0.05,
            0.05,
            0.10,
            0.10,
            0.10,
            0.20,
            0.20,
            0.20,
            0.50,
            0.50,
            0.50,
            np.nan,
        ]
    )

    np.testing.assert_allclose(
        result[:-1],
        expected[:-1],
    )

    assert np.isnan(result[-1])


def test_empty_input():
    xyz = np.empty((0, 3), dtype=np.float64)

    result = build_vectorized_adaptive_map(xyz)

    assert isinstance(result, VectorizedAdaptiveMap)
    assert result.cell_count == 0


def test_invalid_shape():
    xyz = np.zeros((10, 2), dtype=np.float64)

    with pytest.raises(ValueError):
        build_vectorized_adaptive_map(xyz)


def test_non_finite_input():
    xyz = np.zeros((10, 3), dtype=np.float64)
    xyz[0, 0] = np.nan

    with pytest.raises(ValueError):
        build_vectorized_adaptive_map(xyz)


def test_points_outside_range_are_ignored():
    xyz = np.array(
        [
            [150.0, 0.0, 1.0],
            [-150.0, 0.0, 2.0],
            [0.0, 150.0, 3.0],
        ]
    )

    result = build_vectorized_adaptive_map(xyz)

    assert result.cell_count == 0


def test_single_point():
    xyz = np.array(
        [
            [1.0, 2.0, 3.0],
        ]
    )

    result = build_vectorized_adaptive_map(xyz)

    assert result.cell_count == 1

    cell = result.cells[0]

    assert cell.resolution == 0.05
    assert cell.point_count == 1
    assert cell.z_min == 3.0
    assert cell.z_max == 3.0
    assert cell.z_mean == 3.0
    assert cell.z_var == 0.0


def test_points_in_same_cell_are_aggregated():
    xyz = np.array(
        [
            [1.00, 1.00, 1.0],
            [1.01, 1.01, 2.0],
            [1.02, 1.02, 3.0],
        ]
    )

    result = build_vectorized_adaptive_map(xyz)

    assert result.cell_count == 1

    cell = result.cells[0]

    assert cell.point_count == 3
    assert cell.z_min == 1.0
    assert cell.z_max == 3.0
    assert cell.z_mean == 2.0
    assert np.isclose(cell.z_var, 2.0 / 3.0)


def test_negative_coordinates():
    xyz = np.array(
        [
            [-1.0, -1.0, 1.0],
            [-1.01, -1.01, 2.0],
        ]
    )

    result = build_vectorized_adaptive_map(xyz)

    assert result.cell_count == 2

    assert result.cells[0].x_index < 0
    assert result.cells[0].y_index < 0


def test_resolution_bands():
    xyz = np.array(
        [
            [5.0, 0.0, 1.0],
            [15.0, 0.0, 2.0],
            [30.0, 0.0, 3.0],
            [60.0, 0.0, 4.0],
        ]
    )

    result = build_vectorized_adaptive_map(xyz)

    resolutions = sorted(
        cell.resolution
        for cell in result.cells
    )

    assert resolutions == [
        0.05,
        0.10,
        0.20,
        0.50,
    ]


def test_deterministic_ordering():
    xyz = np.array(
        [
            [15.0, 0.0, 2.0],
            [5.0, 0.0, 1.0],
            [30.0, 0.0, 3.0],
            [60.0, 0.0, 4.0],
        ]
    )

    result = build_vectorized_adaptive_map(xyz)

    keys = [
        (
            cell.resolution,
            cell.x_index,
            cell.y_index,
        )
        for cell in result.cells
    ]

    assert keys == sorted(keys)


def test_cell_geometry():
    xyz = np.array(
        [
            [1.0, 2.0, 3.0],
        ]
    )

    result = build_vectorized_adaptive_map(xyz)

    cell = result.cells[0]

    assert cell.x_min == cell.x_index * 0.05
    assert cell.y_min == cell.y_index * 0.05
    assert cell.x_max == (
        cell.x_index + 1
    ) * 0.05
    assert cell.y_max == (
        cell.y_index + 1
    ) * 0.05


def test_resolution_array():
    xyz = np.array(
        [
            [1.0, 0.0, 1.0],
            [15.0, 0.0, 2.0],
            [30.0, 0.0, 3.0],
            [60.0, 0.0, 4.0],
        ]
    )

    result = build_vectorized_adaptive_map(xyz)

    resolutions = result.resolutions()

    assert isinstance(
        resolutions,
        np.ndarray,
    )

    assert len(resolutions) == 4