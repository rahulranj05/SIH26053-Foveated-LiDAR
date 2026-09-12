import numpy as np

from mapping.basic_grid import build_uniform_grid


def test_basic_grid_statistics():
    """Check basic cell assignment and elevation statistics."""

    xyz = np.array(
        [
            [0.05, 0.05, 1.0],
            [0.06, 0.05, 1.2],
            [0.15, 0.05, 2.0],
        ]
    )

    grid = build_uniform_grid(
        xyz,
        resolution=0.10,
        x_range=(0.0, 0.30),
        y_range=(0.0, 0.20),
    )

    assert grid.count[0, 0] == 2
    assert grid.count[0, 1] == 1

    assert np.isclose(grid.z_min[0, 0], 1.0)
    assert np.isclose(grid.z_max[0, 0], 1.2)
    assert np.isclose(grid.z_mean[0, 0], 1.1)
    assert np.isclose(grid.z_var[0, 0], 0.01)

    assert np.isclose(grid.z_min[0, 1], 2.0)
    assert np.isclose(grid.z_max[0, 1], 2.0)
    assert np.isclose(grid.z_mean[0, 1], 2.0)
    assert np.isclose(grid.z_var[0, 1], 0.0)


def test_negative_coordinates():
    """Check that negative world coordinates map correctly."""

    xyz = np.array(
        [
            [-0.05, -0.05, 3.0],
        ]
    )

    grid = build_uniform_grid(
        xyz,
        resolution=0.10,
        x_range=(-0.20, 0.20),
        y_range=(-0.20, 0.20),
    )

    assert grid.count[1, 1] == 1
    assert np.isclose(grid.z_mean[1, 1], 3.0)


def test_out_of_range_points_are_rejected():
    """Points outside the requested map should not enter the grid."""

    xyz = np.array(
        [
            [0.05, 0.05, 1.0],
            [1.00, 0.05, 2.0],
            [0.05, 1.00, 3.0],
            [-1.00, 0.05, 4.0],
        ]
    )

    grid = build_uniform_grid(
        xyz,
        resolution=0.10,
        x_range=(0.0, 0.20),
        y_range=(0.0, 0.20),
    )

    assert grid.count.sum() == 1
    assert grid.count[0, 0] == 1
    assert np.isclose(grid.z_mean[0, 0], 1.0)


def test_empty_input():
    """An empty point cloud should produce an empty grid."""

    xyz = np.empty((0, 3))

    grid = build_uniform_grid(
        xyz,
        resolution=0.10,
        x_range=(0.0, 0.20),
        y_range=(0.0, 0.20),
    )

    assert grid.count.sum() == 0
    assert np.all(np.isnan(grid.z_min))
    assert np.all(np.isnan(grid.z_max))
    assert np.all(np.isnan(grid.z_mean))
    assert np.all(np.isnan(grid.z_var))


def test_empty_cells_are_nan():
    """Empty cells must have explicit NaN elevation statistics."""

    xyz = np.array(
        [
            [0.05, 0.05, 1.0],
        ]
    )

    grid = build_uniform_grid(
        xyz,
        resolution=0.10,
        x_range=(0.0, 0.30),
        y_range=(0.0, 0.30),
    )

    assert grid.count[0, 0] == 1
    assert grid.count[0, 1] == 0
    assert np.isnan(grid.z_mean[0, 1])
    assert np.isnan(grid.z_min[0, 1])
    assert np.isnan(grid.z_max[0, 1])
    assert np.isnan(grid.z_var[0, 1])


def test_flat_terrain():
    """A flat surface should have zero elevation variance."""

    xyz = np.array(
        [
            [0.05, 0.05, 2.0],
            [0.06, 0.05, 2.0],
            [0.05, 0.06, 2.0],
            [0.06, 0.06, 2.0],
        ]
    )

    grid = build_uniform_grid(
        xyz,
        resolution=0.10,
        x_range=(0.0, 0.20),
        y_range=(0.0, 0.20),
    )

    assert grid.count[0, 0] == 4
    assert np.isclose(grid.z_min[0, 0], 2.0)
    assert np.isclose(grid.z_max[0, 0], 2.0)
    assert np.isclose(grid.z_mean[0, 0], 2.0)
    assert np.isclose(grid.z_var[0, 0], 0.0)


def test_known_variance():
    """Check variance against a manually known result."""

    xyz = np.array(
        [
            [0.05, 0.05, 1.0],
            [0.05, 0.05, 2.0],
            [0.05, 0.05, 3.0],
            [0.05, 0.05, 4.0],
        ]
    )

    grid = build_uniform_grid(
        xyz,
        resolution=0.10,
        x_range=(0.0, 0.20),
        y_range=(0.0, 0.20),
    )

    # Population variance of [1, 2, 3, 4] = 1.25
    assert np.isclose(grid.z_mean[0, 0], 2.5)
    assert np.isclose(grid.z_var[0, 0], 1.25)


def test_boundary_coordinates():
    """Check points lying exactly on cell boundaries."""

    xyz = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.10, 0.10, 2.0],
        ]
    )

    grid = build_uniform_grid(
        xyz,
        resolution=0.10,
        x_range=(0.0, 0.20),
        y_range=(0.0, 0.20),
    )

    assert grid.count[0, 0] == 1
    assert grid.count[1, 1] == 1

    assert np.isclose(grid.z_mean[0, 0], 1.0)
    assert np.isclose(grid.z_mean[1, 1], 2.0)


def test_cell_centers():
    """Check that cell centers are recovered correctly."""

    xyz = np.array([[0.05, 0.05, 1.0]])

    grid = build_uniform_grid(
        xyz,
        resolution=0.10,
        x_range=(0.0, 0.20),
        y_range=(0.0, 0.20),
    )

    x_centers, y_centers = grid.cell_centers()

    assert np.allclose(x_centers, [0.05, 0.15])
    assert np.allclose(y_centers, [0.05, 0.15])


def test_deterministic_results():
    """The same input should always produce the same grid."""

    xyz = np.array(
        [
            [-0.15, 0.05, 1.0],
            [0.05, 0.05, 2.0],
            [0.15, -0.05, 3.0],
            [0.25, 0.15, 4.0],
        ]
    )

    grid_a = build_uniform_grid(
        xyz,
        resolution=0.10,
        x_range=(-0.20, 0.30),
        y_range=(-0.20, 0.30),
    )

    grid_b = build_uniform_grid(
        xyz,
        resolution=0.10,
        x_range=(-0.20, 0.30),
        y_range=(-0.20, 0.30),
    )

    assert np.array_equal(grid_a.count, grid_b.count)
    assert np.allclose(grid_a.z_min, grid_b.z_min, equal_nan=True)
    assert np.allclose(grid_a.z_max, grid_b.z_max, equal_nan=True)
    assert np.allclose(grid_a.z_mean, grid_b.z_mean, equal_nan=True)
    assert np.allclose(grid_a.z_var, grid_b.z_var, equal_nan=True)