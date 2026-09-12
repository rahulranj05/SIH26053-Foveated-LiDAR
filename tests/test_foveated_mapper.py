from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

from mapping.foveated_mapper import (
    LEVEL_RESOLUTIONS,
    RESOLUTION_TO_LEVEL,
    build_foveated_map,
    cell_indices,
    level_distribution,
    level_from_resolution,
    reason_distribution,
    resolution_distribution,
)


def test_level_resolution_mapping():
    resolution = np.array(
        [0.05, 0.10, 0.20, 0.40]
    )

    levels = level_from_resolution(
        resolution
    )

    assert np.array_equal(
        levels,
        [0, 1, 2, 3],
    )


def test_invalid_resolution():
    resolution = np.array(
        [0.05, 0.15, 0.40]
    )

    with pytest.raises(ValueError):
        level_from_resolution(
            resolution
        )


def test_cell_indices_positive():
    x = np.array(
        [0.00, 0.049, 0.051, 0.101]
    )

    y = np.zeros(4)

    resolution = np.full(
        4,
        0.05,
    )

    ix, iy = cell_indices(
        x,
        y,
        resolution,
    )

    assert np.array_equal(
        ix,
        [0, 0, 1, 2],
    )

    assert np.array_equal(
        iy,
        [0, 0, 0, 0],
    )


def test_cell_indices_negative():
    x = np.array(
        [-0.001, -0.049, -0.051]
    )

    y = np.zeros(3)

    resolution = np.full(
        3,
        0.05,
    )

    ix, _ = cell_indices(
        x,
        y,
        resolution,
    )

    assert np.array_equal(
        ix,
        [-1, -1, -2],
    )


def test_empty_map():
    xyz = np.empty(
        (0, 3),
        dtype=np.float64,
    )

    resolution = np.empty(
        0,
        dtype=np.float64,
    )

    reasons = np.empty(
        0,
        dtype=object,
    )

    result = build_foveated_map(
        xyz,
        resolution,
        reasons,
    )

    assert result.input_points == 0
    assert result.active_cells == 0
    assert result.cells == []


def test_single_point():
    xyz = np.array(
        [[1.2, 2.3, 4.5]]
    )

    resolution = np.array(
        [0.05]
    )

    reasons = np.array(
        ["DYNAMIC"],
        dtype=object,
    )

    result = build_foveated_map(
        xyz,
        resolution,
        reasons,
    )

    assert result.input_points == 1
    assert result.active_cells == 1

    cell = result.cells[0]

    assert cell.level == 0
    assert cell.resolution == 0.05
    assert cell.ix == 24
    assert cell.iy == 46
    assert cell.z_min == 4.5
    assert cell.z_max == 4.5
    assert cell.z_mean == 4.5
    assert cell.z_variance == 0.0
    assert cell.point_count == 1
    assert cell.dominant_reason == "DYNAMIC"


def test_points_in_same_cell_are_aggregated():
    xyz = np.array(
        [
            [0.01, 0.01, 1.0],
            [0.02, 0.02, 2.0],
            [0.04, 0.03, 3.0],
        ]
    )

    resolution = np.full(
        3,
        0.05,
    )

    reasons = np.array(
        [
            "DISTANCE",
            "DISTANCE",
            "DYNAMIC",
        ],
        dtype=object,
    )

    result = build_foveated_map(
        xyz,
        resolution,
        reasons,
    )

    assert result.active_cells == 1

    cell = result.cells[0]

    assert cell.point_count == 3
    assert cell.z_min == 1.0
    assert cell.z_max == 3.0
    assert np.isclose(
        cell.z_mean,
        2.0,
    )

    assert np.isclose(
        cell.z_variance,
        2.0 / 3.0,
    )

    assert cell.dominant_reason == "DISTANCE"


def test_different_resolutions_create_separate_hierarchy_cells():
    xyz = np.array(
        [
            [0.01, 0.01, 1.0],
            [0.02, 0.02, 2.0],
        ]
    )

    resolution = np.array(
        [
            0.05,
            0.10,
        ]
    )

    reasons = np.array(
        [
            "DISTANCE",
            "SEMANTIC",
        ],
        dtype=object,
    )

    result = build_foveated_map(
        xyz,
        resolution,
        reasons,
    )

    assert result.active_cells == 2

    levels = sorted(
        cell.level
        for cell in result.cells
    )

    assert levels == [0, 1]


def test_same_coordinate_different_levels_do_not_collide():
    xyz = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 2.0],
        ]
    )

    resolution = np.array(
        [
            0.05,
            0.40,
        ]
    )

    reasons = np.array(
        [
            "DYNAMIC",
            "DISTANCE",
        ],
        dtype=object,
    )

    result = build_foveated_map(
        xyz,
        resolution,
        reasons,
    )

    assert result.active_cells == 2

    keys = {
        (
            cell.level,
            cell.ix,
            cell.iy,
        )
        for cell in result.cells
    }

    assert len(keys) == 2


def test_resolution_distribution():
    xyz = np.array(
        [
            [0.01, 0.01, 1.0],
            [0.11, 0.01, 1.0],
            [0.21, 0.01, 1.0],
            [0.41, 0.01, 1.0],
        ]
    )

    resolution = np.array(
        [
            0.05,
            0.10,
            0.20,
            0.40,
        ]
    )

    reasons = np.array(
        [
            "DISTANCE",
            "KINEMATIC",
            "SEMANTIC",
            "DYNAMIC",
        ],
        dtype=object,
    )

    result = build_foveated_map(
        xyz,
        resolution,
        reasons,
    )

    distribution = (
        resolution_distribution(
            result
        )
    )

    assert distribution == {
        0.05: 1,
        0.10: 1,
        0.20: 1,
        0.40: 1,
    }


def test_level_distribution():
    xyz = np.array(
        [
            [0.01, 0.01, 1.0],
            [0.11, 0.01, 1.0],
            [0.21, 0.01, 1.0],
            [0.41, 0.01, 1.0],
        ]
    )

    resolution = np.array(
        [
            0.05,
            0.10,
            0.20,
            0.40,
        ]
    )

    reasons = np.array(
        [
            "DISTANCE",
            "KINEMATIC",
            "SEMANTIC",
            "DYNAMIC",
        ],
        dtype=object,
    )

    result = build_foveated_map(
        xyz,
        resolution,
        reasons,
    )

    distribution = (
        level_distribution(
            result
        )
    )

    assert distribution == {
        0: 1,
        1: 1,
        2: 1,
        3: 1,
    }


def test_reason_distribution():
    xyz = np.array(
        [
            [0.01, 0.01, 1.0],
            [0.11, 0.01, 1.0],
            [0.21, 0.01, 1.0],
        ]
    )

    resolution = np.full(
        3,
        0.05,
    )

    reasons = np.array(
        [
            "DYNAMIC",
            "DYNAMIC",
            "SEMANTIC",
        ],
        dtype=object,
    )

    result = build_foveated_map(
        xyz,
        resolution,
        reasons,
    )

    distribution = reason_distribution(
        result
    )

    assert distribution[
        "DYNAMIC"
    ] == 2

    assert distribution[
        "SEMANTIC"
    ] == 1


def test_all_cells_use_valid_hierarchy():
    rng = np.random.default_rng(
        42
    )

    xyz = rng.uniform(
        -10.0,
        10.0,
        size=(1000, 3),
    )

    resolution = rng.choice(
        np.array(
            [
                0.05,
                0.10,
                0.20,
                0.40,
            ]
        ),
        size=1000,
    )

    reasons = np.array(
        [
            "DISTANCE"
            if i % 2 == 0
            else "DYNAMIC"
            for i in range(1000)
        ],
        dtype=object,
    )

    result = build_foveated_map(
        xyz,
        resolution,
        reasons,
    )

    for cell in result.cells:
        assert cell.level in LEVEL_RESOLUTIONS

        assert np.isclose(
            cell.resolution,
            LEVEL_RESOLUTIONS[
                cell.level
            ],
        )

        assert (
            cell.resolution
            in RESOLUTION_TO_LEVEL
        )


def test_point_count_is_conserved():
    rng = np.random.default_rng(
        123
    )

    xyz = rng.uniform(
        -20.0,
        20.0,
        size=(5000, 3),
    )

    resolution = rng.choice(
        np.array(
            [
                0.05,
                0.10,
                0.20,
                0.40,
            ]
        ),
        size=5000,
    )

    reasons = np.full(
        5000,
        "DISTANCE",
        dtype=object,
    )

    result = build_foveated_map(
        xyz,
        resolution,
        reasons,
    )

    total_points = sum(
        cell.point_count
        for cell in result.cells
    )

    assert total_points == 5000


def test_nan_xyz_rejected():
    xyz = np.array(
        [
            [0.0, 0.0, np.nan]
        ]
    )

    resolution = np.array(
        [0.05]
    )

    reasons = np.array(
        ["DISTANCE"],
        dtype=object,
    )

    with pytest.raises(ValueError):
        build_foveated_map(
            xyz,
            resolution,
            reasons,
        )


def test_mismatched_resolution_length():
    xyz = np.zeros(
        (3, 3)
    )

    resolution = np.array(
        [0.05, 0.10]
    )

    reasons = np.array(
        [
            "DISTANCE",
            "DISTANCE",
            "DISTANCE",
        ],
        dtype=object,
    )

    with pytest.raises(ValueError):
        build_foveated_map(
            xyz,
            resolution,
            reasons,
        )


def test_mismatched_reason_length():
    xyz = np.zeros(
        (3, 3)
    )

    resolution = np.full(
        3,
        0.05,
    )

    reasons = np.array(
        ["DISTANCE"],
        dtype=object,
    )

    with pytest.raises(ValueError):
        build_foveated_map(
            xyz,
            resolution,
            reasons,
        )