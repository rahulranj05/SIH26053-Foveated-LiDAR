import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mapping.distance_adaptive import build_distance_adaptive_map
from mapping.vectorized_adaptive import build_vectorized_adaptive_map


def cells_to_array(cells):
    return np.array(
        [
            [
                cell.resolution,
                cell.x_index,
                cell.y_index,
                cell.point_count,
                cell.z_min,
                cell.z_max,
                cell.z_mean,
                cell.z_var,
            ]
            for cell in cells
        ],
        dtype=np.float64,
    )


def test_a8_and_a11_produce_same_results():
    rng = np.random.default_rng(42)

    xyz = rng.uniform(
        low=[-100.0, -100.0, -5.0],
        high=[100.0, 100.0, 5.0],
        size=(10000, 3),
    )

    a8 = build_distance_adaptive_map(xyz)
    a11 = build_vectorized_adaptive_map(xyz)

    assert a8.cell_count == a11.cell_count

    a8_array = cells_to_array(a8.cells)
    a11_array = cells_to_array(a11.cells)

    np.testing.assert_allclose(
        a8_array,
        a11_array,
        rtol=1e-10,
        atol=1e-10,
    )


def test_a8_and_a11_match_boundary_points():
    xyz = np.array(
        [
            [0.0, 0.0, 1.0],
            [9.999, 0.0, 2.0],
            [10.0, 0.0, 3.0],
            [24.999, 0.0, 4.0],
            [25.0, 0.0, 5.0],
            [49.999, 0.0, 6.0],
            [50.0, 0.0, 7.0],
            [100.0, 0.0, 8.0],
            [100.001, 0.0, 9.0],
            [-10.0, 0.0, 10.0],
            [-25.0, 0.0, 11.0],
            [-50.0, 0.0, 12.0],
            [-100.0, 0.0, 13.0],
        ],
        dtype=np.float64,
    )

    a8 = build_distance_adaptive_map(xyz)
    a11 = build_vectorized_adaptive_map(xyz)

    assert a8.cell_count == a11.cell_count

    a8_array = cells_to_array(a8.cells)
    a11_array = cells_to_array(a11.cells)

    np.testing.assert_allclose(
        a8_array,
        a11_array,
        rtol=1e-10,
        atol=1e-10,
    )


def test_a8_and_a11_match_negative_coordinates():
    rng = np.random.default_rng(123)

    xyz = rng.uniform(
        low=[-80.0, -80.0, -10.0],
        high=[80.0, 80.0, 10.0],
        size=(5000, 3),
    )

    a8 = build_distance_adaptive_map(xyz)
    a11 = build_vectorized_adaptive_map(xyz)

    assert a8.cell_count == a11.cell_count

    a8_array = cells_to_array(a8.cells)
    a11_array = cells_to_array(a11.cells)

    np.testing.assert_allclose(
        a8_array,
        a11_array,
        rtol=1e-10,
        atol=1e-10,
    )