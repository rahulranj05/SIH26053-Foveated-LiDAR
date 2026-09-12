import numpy as np

from mapping.basic_grid import build_uniform_grid
from mapping.traversability import (
    _normalize_penalty,
    compute_traversability,
)


def test_normalize_penalty_safe_value():
    values = np.array([0.0, 5.0, 10.0])

    result = _normalize_penalty(
        values,
        safe_limit=5.0,
        unsafe_limit=10.0,
    )

    expected = np.array([0.0, 0.0, 1.0])

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_normalize_penalty_midpoint():
    values = np.array([7.5])

    result = _normalize_penalty(
        values,
        safe_limit=5.0,
        unsafe_limit=10.0,
    )

    np.testing.assert_allclose(
        result,
        [0.5],
    )


def test_normalize_penalty_clamps_values():
    values = np.array([-10.0, 20.0])

    result = _normalize_penalty(
        values,
        safe_limit=5.0,
        unsafe_limit=10.0,
    )

    expected = np.array([0.0, 1.0])

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_normalize_penalty_preserves_nan():
    values = np.array([5.0, np.nan, 10.0])

    result = _normalize_penalty(
        values,
        safe_limit=5.0,
        unsafe_limit=10.0,
    )

    assert np.isnan(result[1])


def test_normalize_penalty_rejects_invalid_limits():
    values = np.array([1.0, 2.0])

    try:
        _normalize_penalty(
            values,
            safe_limit=10.0,
            unsafe_limit=5.0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError for invalid limits."
        )


def test_traversability_range():
    xyz = np.array(
        [
            [0.05, 0.05, 0.0],
            [0.15, 0.05, 0.0],
            [0.05, 0.15, 0.0],
            [0.15, 0.15, 0.0],
        ],
        dtype=np.float64,
    )

    grid = build_uniform_grid(
        xyz,
        resolution=0.10,
        x_range=(0.0, 1.0),
        y_range=(0.0, 1.0),
    )

    traversability = compute_traversability(grid)

    valid = np.isfinite(traversability)

    assert np.all(
        traversability[valid] >= 0.0
    )

    assert np.all(
        traversability[valid] <= 1.0
    )


def test_flat_surface_is_highly_traversable():
    xyz = np.array(
        [
            [0.05, 0.05, 0.0],
            [0.15, 0.05, 0.0],
            [0.05, 0.15, 0.0],
            [0.15, 0.15, 0.0],
        ],
        dtype=np.float64,
    )

    grid = build_uniform_grid(
        xyz,
        resolution=0.10,
        x_range=(0.0, 1.0),
        y_range=(0.0, 1.0),
    )

    traversability = compute_traversability(grid)

    populated = grid.count > 0

    assert np.allclose(
        traversability[populated],
        1.0,
    )


def test_steep_surface_has_lower_traversability():
    xyz = np.array(
        [
            [0.05, 0.05, 0.0],
            [0.15, 0.05, 0.0],
            [0.05, 0.15, 1.0],
            [0.15, 0.15, 1.0],
        ],
        dtype=np.float64,
    )

    grid = build_uniform_grid(
        xyz,
        resolution=0.10,
        x_range=(0.0, 1.0),
        y_range=(0.0, 1.0),
    )

    traversability = compute_traversability(grid)

    populated = grid.count > 0

    assert np.nanmean(
        traversability[populated]
    ) < 1.0


def test_empty_cells_are_nan():
    xyz = np.array(
        [
            [0.05, 0.05, 0.0],
        ],
        dtype=np.float64,
    )

    grid = build_uniform_grid(
        xyz,
        resolution=0.10,
        x_range=(0.0, 1.0),
        y_range=(0.0, 1.0),
    )

    traversability = compute_traversability(grid)

    empty = grid.count == 0

    assert np.all(
        np.isnan(traversability[empty])
    )