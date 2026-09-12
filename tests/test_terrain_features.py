import numpy as np

from mapping.basic_grid import build_uniform_grid
from mapping.terrain_features import (
    compute_height_span,
    compute_height_discontinuity,
    compute_roughness,
    compute_slope,
)


def test_height_span():
    xyz = np.array([
        [0.05, 0.05, 1.0],
        [0.06, 0.06, 2.5],
        [0.07, 0.07, 4.0],
    ])

    grid = build_uniform_grid(
        xyz,
        resolution=0.10,
        x_range=(0.0, 1.0),
        y_range=(0.0, 1.0),
    )

    height_span = compute_height_span(grid)

    assert np.isclose(
        height_span[0, 0],
        3.0,
    )


def test_flat_cell_has_zero_height_span():
    xyz = np.array([
        [0.05, 0.05, 2.0],
        [0.06, 0.06, 2.0],
        [0.07, 0.07, 2.0],
    ])

    grid = build_uniform_grid(
        xyz,
        resolution=0.10,
        x_range=(0.0, 1.0),
        y_range=(0.0, 1.0),
    )

    height_span = compute_height_span(grid)

    assert np.isclose(
        height_span[0, 0],
        0.0,
    )


def test_empty_cells_are_nan():
    xyz = np.array([
        [0.05, 0.05, 1.0],
    ])

    grid = build_uniform_grid(
        xyz,
        resolution=0.10,
        x_range=(0.0, 1.0),
        y_range=(0.0, 1.0),
    )

    height_span = compute_height_span(grid)

    assert np.isnan(
        height_span[5, 5]
    )


def test_flat_surface_has_zero_slope():
    x = np.arange(
        0.05,
        1.0,
        0.10,
    )

    y = np.arange(
        0.05,
        1.0,
        0.10,
    )

    xx, yy = np.meshgrid(
        x,
        y,
    )

    xyz = np.column_stack([
        xx.ravel(),
        yy.ravel(),
        np.zeros(xx.size),
    ])

    grid = build_uniform_grid(
        xyz,
        resolution=0.10,
        x_range=(0.0, 1.0),
        y_range=(0.0, 1.0),
    )

    slope = compute_slope(grid)

    populated = np.isfinite(
        slope
    )

    assert np.allclose(
        slope[populated],
        0.0,
    )


def test_known_slope():
    # z = x
    # dz/dx = 1
    # slope = atan(1) = 45 degrees

    x = np.arange(
        0.05,
        1.0,
        0.10,
    )

    y = np.arange(
        0.05,
        1.0,
        0.10,
    )

    xx, yy = np.meshgrid(
        x,
        y,
    )

    xyz = np.column_stack([
        xx.ravel(),
        yy.ravel(),
        xx.ravel(),
    ])

    grid = build_uniform_grid(
        xyz,
        resolution=0.10,
        x_range=(0.0, 1.0),
        y_range=(0.0, 1.0),
    )

    slope = compute_slope(grid)

    populated = np.isfinite(
        slope
    )

    assert np.allclose(
        slope[populated],
        45.0,
        atol=1e-6,
    )


def test_flat_surface_has_zero_roughness():
    xyz = np.array([
        [0.05, 0.05, 2.0],
        [0.06, 0.06, 2.0],
        [0.07, 0.07, 2.0],
        [0.08, 0.08, 2.0],
    ])

    grid = build_uniform_grid(
        xyz,
        resolution=0.10,
        x_range=(0.0, 1.0),
        y_range=(0.0, 1.0),
    )

    roughness = compute_roughness(grid)

    assert np.isclose(
        roughness[0, 0],
        0.0,
    )


def test_known_roughness():
    xyz = np.array([
        [0.05, 0.05, 1.0],
        [0.06, 0.06, 2.0],
        [0.07, 0.07, 3.0],
    ])

    grid = build_uniform_grid(
        xyz,
        resolution=0.10,
        x_range=(0.0, 1.0),
        y_range=(0.0, 1.0),
    )

    roughness = compute_roughness(grid)

    # Population standard deviation of [1, 2, 3].
    expected = np.sqrt(
        2.0 / 3.0
    )

    assert np.isclose(
        roughness[0, 0],
        expected,
    )


def test_empty_cell_roughness_is_nan():
    xyz = np.array([
        [0.05, 0.05, 1.0],
    ])

    grid = build_uniform_grid(
        xyz,
        resolution=0.10,
        x_range=(0.0, 1.0),
        y_range=(0.0, 1.0),
    )

    roughness = compute_roughness(grid)

    assert np.isnan(
        roughness[5, 5]
    )


def test_flat_surface_has_zero_height_discontinuity():
    x = np.arange(
        0.05,
        1.0,
        0.10,
    )

    y = np.arange(
        0.05,
        1.0,
        0.10,
    )

    xx, yy = np.meshgrid(
        x,
        y,
    )

    xyz = np.column_stack([
        xx.ravel(),
        yy.ravel(),
        np.zeros(xx.size),
    ])

    grid = build_uniform_grid(
        xyz,
        resolution=0.10,
        x_range=(0.0, 1.0),
        y_range=(0.0, 1.0),
    )

    discontinuity = compute_height_discontinuity(
        grid
    )

    populated = np.isfinite(
        discontinuity
    )

    assert np.allclose(
        discontinuity[populated],
        0.0,
    )


def test_step_has_height_discontinuity():
    x = np.arange(
        0.05,
        1.0,
        0.10,
    )

    y = np.arange(
        0.05,
        1.0,
        0.10,
    )

    xx, yy = np.meshgrid(
        x,
        y,
    )

    # 1 metre step halfway across the surface.
    z = np.where(
        xx < 0.5,
        0.0,
        1.0,
    )

    xyz = np.column_stack([
        xx.ravel(),
        yy.ravel(),
        z.ravel(),
    ])

    grid = build_uniform_grid(
        xyz,
        resolution=0.10,
        x_range=(0.0, 1.0),
        y_range=(0.0, 1.0),
    )

    discontinuity = compute_height_discontinuity(
        grid
    )

    assert np.nanmax(
        discontinuity
    ) >= 1.0
def test_all_terrain_features_integrate_on_one_grid():
    xyz = np.array(
        [
            [0.05, 0.05, 1.0],
            [0.15, 0.05, 1.0],
            [0.05, 0.15, 1.0],
            [0.15, 0.15, 1.0],
            [1.05, 0.05, 2.0],
            [1.15, 0.05, 2.0],
            [1.05, 0.15, 2.0],
            [1.15, 0.15, 2.0],
        ],
        dtype=np.float64,
    )

    grid = build_uniform_grid(
        xyz,
        resolution=0.10,
        x_range=(0.0, 2.0),
        y_range=(0.0, 2.0),
    )

    height_span = compute_height_span(grid)
    slope = compute_slope(grid)
    roughness = compute_roughness(grid)
    discontinuity = compute_height_discontinuity(grid)

    expected_shape = grid.count.shape

    assert height_span.shape == expected_shape
    assert slope.shape == expected_shape
    assert roughness.shape == expected_shape
    assert discontinuity.shape == expected_shape

    populated = grid.count > 0

    assert np.all(np.isfinite(height_span[populated]))
    assert np.all(np.isfinite(slope[populated]))
    assert np.all(np.isfinite(roughness[populated]))
    assert np.all(np.isfinite(discontinuity[populated]))

    assert np.all(height_span[populated] >= 0.0)
    assert np.all(slope[populated] >= 0.0)
    assert np.all(roughness[populated] >= 0.0)
    assert np.all(discontinuity[populated] >= 0.0)