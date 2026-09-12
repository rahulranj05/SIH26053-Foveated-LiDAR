import numpy as np

from mapping.distance_adaptive import (
    resolution_for_distance,
    build_distance_adaptive_map,
)


def test_resolution_near():
    distances = np.array([0.0, 1.0, 5.0, 9.99])
    resolutions = resolution_for_distance(distances)

    assert np.all(resolutions == 0.05)


def test_resolution_medium_near():
    distances = np.array([10.0, 12.0, 20.0, 24.99])
    resolutions = resolution_for_distance(distances)

    assert np.all(resolutions == 0.10)


def test_resolution_medium_far():
    distances = np.array([25.0, 30.0, 40.0, 49.99])
    resolutions = resolution_for_distance(distances)

    assert np.all(resolutions == 0.20)


def test_resolution_far():
    distances = np.array([50.0, 60.0, 80.0, 100.0])
    resolutions = resolution_for_distance(distances)

    assert np.all(resolutions == 0.50)


def test_distance_beyond_range_is_nan():
    distances = np.array([100.01, 120.0, 200.0])
    resolutions = resolution_for_distance(distances)

    assert np.all(np.isnan(resolutions))


def test_distance_band_boundaries():
    distances = np.array([9.999, 10.0, 24.999, 25.0, 49.999, 50.0, 100.0])

    resolutions = resolution_for_distance(distances)

    expected = np.array([
        0.05,
        0.10,
        0.10,
        0.20,
        0.20,
        0.50,
        0.50,
    ])

    assert np.allclose(resolutions, expected)


def test_empty_point_cloud():
    xyz = np.empty((0, 3), dtype=np.float64)

    adaptive_map = build_distance_adaptive_map(xyz)

    assert adaptive_map.cell_count == 0


def test_points_beyond_100m_are_ignored():
    xyz = np.array([
        [5.0, 0.0, 1.0],
        [20.0, 0.0, 2.0],
        [30.0, 0.0, 3.0],
        [60.0, 0.0, 4.0],
        [150.0, 0.0, 5.0],
    ])

    adaptive_map = build_distance_adaptive_map(xyz)

    assert adaptive_map.cell_count == 4

    distances = np.array([
        np.hypot(cell.x_center, cell.y_center)
        for cell in adaptive_map.cells
    ])

    assert np.all(distances <= 100.0)


def test_all_expected_resolutions_are_present():
    xyz = np.array([
        [5.0, 0.0, 1.0],
        [20.0, 0.0, 2.0],
        [30.0, 0.0, 3.0],
        [60.0, 0.0, 4.0],
    ])

    adaptive_map = build_distance_adaptive_map(xyz)

    resolutions = set(adaptive_map.resolutions())

    assert resolutions == {0.05, 0.10, 0.20, 0.50}


def test_cell_statistics():
    xyz = np.array([
        [1.01, 1.01, 1.0],
        [1.02, 1.02, 2.0],
        [1.03, 1.03, 3.0],
    ])

    adaptive_map = build_distance_adaptive_map(xyz)

    assert adaptive_map.cell_count == 1

    cell = adaptive_map.cells[0]

    assert cell.point_count == 3
    assert cell.z_min == 1.0
    assert cell.z_max == 3.0
    assert np.isclose(cell.z_mean, 2.0)
    assert np.isclose(cell.z_var, 2.0 / 3.0)


def test_negative_coordinates():
    xyz = np.array([
        [-5.0, -2.0, 1.0],
        [-5.01, -2.01, 2.0],
    ])

    adaptive_map = build_distance_adaptive_map(xyz)

    assert adaptive_map.cell_count >= 1

    for cell in adaptive_map.cells:
        assert cell.resolution == 0.05