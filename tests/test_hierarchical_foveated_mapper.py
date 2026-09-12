from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)


from mapping.hierarchical_foveated_mapper import (
    build_hierarchical_foveated_map,
    level_distribution,
    reason_distribution,
    resolution_distribution,
)


REASONS = np.array(
    [
        "DISTANCE",
        "KINEMATIC",
        "PREDICTED_PATH",
        "SEMANTIC",
        "DYNAMIC",
    ],
    dtype="U16",
)


def test_empty_input():

    result = build_hierarchical_foveated_map(
        np.empty(
            (0, 3),
            dtype=np.float64,
        ),
        np.empty(
            0,
            dtype=np.float64,
        ),
        np.empty(
            0,
            dtype="U16",
        ),
    )

    assert result.input_points == 0
    assert result.active_cells == 0


def test_single_coarse_point():

    xyz = np.array(
        [
            [1.0, 1.0, 2.0],
        ]
    )

    resolution = np.array(
        [0.40]
    )

    reasons = np.array(
        ["DISTANCE"],
        dtype="U16",
    )

    result = build_hierarchical_foveated_map(
        xyz,
        resolution,
        reasons,
    )

    assert result.active_cells == 1
    assert result.level[0] == 3
    assert result.resolution[0] == 0.40
    assert result.point_count[0] == 1


def test_single_fine_point():

    xyz = np.array(
        [
            [1.0, 1.0, 2.0],
        ]
    )

    resolution = np.array(
        [0.05]
    )

    reasons = np.array(
        ["SEMANTIC"],
        dtype="U16",
    )

    result = build_hierarchical_foveated_map(
        xyz,
        resolution,
        reasons,
    )

    assert result.active_cells == 1
    assert result.level[0] == 0
    assert result.resolution[0] == 0.05


def test_mixed_points_refine_parent():

    xyz = np.array(
        [
            [1.00, 1.00, 1.0],
            [1.30, 1.00, 1.2],
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
            "SEMANTIC",
            "DISTANCE",
        ],
        dtype="U16",
    )

    result = build_hierarchical_foveated_map(
        xyz,
        resolution,
        reasons,
    )

    assert result.active_cells >= 2

    assert np.all(
        result.level
        <= 3
    )


def test_point_count_conservation():

    rng = np.random.default_rng(
        123
    )

    xyz = rng.uniform(
        -10.0,
        10.0,
        size=(1000, 3),
    )

    resolution = rng.choice(
        [
            0.05,
            0.10,
            0.20,
            0.40,
        ],
        size=1000,
    )

    reasons = rng.choice(
        REASONS,
        size=1000,
    )

    result = build_hierarchical_foveated_map(
        xyz,
        resolution,
        reasons,
    )

    assert (
        np.sum(
            result.point_count
        )
        == 1000
    )


def test_only_valid_levels():

    rng = np.random.default_rng(
        456
    )

    xyz = rng.uniform(
        -20.0,
        20.0,
        size=(1000, 3),
    )

    resolution = rng.choice(
        [
            0.05,
            0.10,
            0.20,
            0.40,
        ],
        size=1000,
    )

    reasons = rng.choice(
        REASONS,
        size=1000,
    )

    result = build_hierarchical_foveated_map(
        xyz,
        resolution,
        reasons,
    )

    assert np.all(
        np.isin(
            result.level,
            [0, 1, 2, 3],
        )
    )


def test_resolution_matches_level():

    rng = np.random.default_rng(
        789
    )

    xyz = rng.uniform(
        -10.0,
        10.0,
        size=(500, 3),
    )

    resolution = rng.choice(
        [
            0.05,
            0.10,
            0.20,
            0.40,
        ],
        size=500,
    )

    reasons = rng.choice(
        REASONS,
        size=500,
    )

    result = build_hierarchical_foveated_map(
        xyz,
        resolution,
        reasons,
    )

    expected = np.array(
        [
            0.05,
            0.10,
            0.20,
            0.40,
        ]
    )

    assert np.allclose(
        result.resolution,
        expected[
            result.level
        ],
    )


def test_reason_distribution():

    xyz = np.array(
        [
            [0.1, 0.1, 1.0],
            [0.2, 0.2, 1.1],
            [5.0, 5.0, 2.0],
        ]
    )

    resolution = np.array(
        [
            0.05,
            0.05,
            0.40,
        ]
    )

    reasons = np.array(
        [
            "SEMANTIC",
            "SEMANTIC",
            "DISTANCE",
        ],
        dtype="U16",
    )

    result = build_hierarchical_foveated_map(
        xyz,
        resolution,
        reasons,
    )

    distribution = reason_distribution(
        result
    )

    assert distribution[
        "SEMANTIC"
    ] >= 1

    assert distribution[
        "DISTANCE"
    ] >= 1


def test_resolution_distribution():

    xyz = np.array(
        [
            [0.1, 0.1, 1.0],
            [1.0, 1.0, 1.0],
            [5.0, 5.0, 2.0],
        ]
    )

    resolution = np.array(
        [
            0.05,
            0.10,
            0.40,
        ]
    )

    reasons = np.array(
        [
            "SEMANTIC",
            "KINEMATIC",
            "DISTANCE",
        ],
        dtype="U16",
    )

    result = build_hierarchical_foveated_map(
        xyz,
        resolution,
        reasons,
    )

    distribution = (
        resolution_distribution(
            result
        )
    )

    assert sum(
        distribution.values()
    ) == result.active_cells


def test_level_distribution():

    rng = np.random.default_rng(
        999
    )

    xyz = rng.uniform(
        -10.0,
        10.0,
        size=(500, 3),
    )

    resolution = rng.choice(
        [
            0.05,
            0.10,
            0.20,
            0.40,
        ],
        size=500,
    )

    reasons = rng.choice(
        REASONS,
        size=500,
    )

    result = build_hierarchical_foveated_map(
        xyz,
        resolution,
        reasons,
    )

    distribution = level_distribution(
        result
    )

    assert sum(
        distribution.values()
    ) == result.active_cells


def test_nan_rejected():

    xyz = np.array(
        [
            [np.nan, 1.0, 2.0],
        ]
    )

    with pytest.raises(
        ValueError
    ):

        build_hierarchical_foveated_map(
            xyz,
            np.array(
                [0.05]
            ),
            np.array(
                ["SEMANTIC"],
                dtype="U16",
            ),
        )


def test_invalid_resolution_rejected():

    xyz = np.array(
        [
            [1.0, 1.0, 1.0],
        ]
    )

    with pytest.raises(
        ValueError
    ):

        build_hierarchical_foveated_map(
            xyz,
            np.array(
                [0.07]
            ),
            np.array(
                ["DISTANCE"],
                dtype="U16",
            ),
        )


def test_mismatched_resolution_rejected():

    xyz = np.zeros(
        (10, 3)
    )

    with pytest.raises(
        ValueError
    ):

        build_hierarchical_foveated_map(
            xyz,
            np.array(
                [0.05]
            ),
            np.array(
                ["DISTANCE"] * 10,
                dtype="U16",
            ),
        )


def test_mismatched_reason_rejected():

    xyz = np.zeros(
        (10, 3)
    )

    with pytest.raises(
        ValueError
    ):

        build_hierarchical_foveated_map(
            xyz,
            np.full(
                10,
                0.05,
            ),
            np.array(
                ["DISTANCE"],
                dtype="U16",
            ),
        )


def test_variance_non_negative():

    rng = np.random.default_rng(
        111
    )

    xyz = rng.uniform(
        -5.0,
        5.0,
        size=(500, 3),
    )

    resolution = np.full(
        500,
        0.10,
    )

    reasons = np.full(
        500,
        "DISTANCE",
        dtype="U16",
    )

    result = build_hierarchical_foveated_map(
        xyz,
        resolution,
        reasons,
    )

    assert np.all(
        result.z_variance >= 0.0
    )


def test_all_coarse_points_remain_coarse():

    rng = np.random.default_rng(
        222
    )

    xyz = rng.uniform(
        -20.0,
        20.0,
        size=(1000, 3),
    )

    resolution = np.full(
        1000,
        0.40,
    )

    reasons = np.full(
        1000,
        "DISTANCE",
        dtype="U16",
    )

    result = build_hierarchical_foveated_map(
        xyz,
        resolution,
        reasons,
    )

    assert np.all(
        result.level == 3
    )


def test_all_fine_points_are_fine():

    rng = np.random.default_rng(
        333
    )

    xyz = rng.uniform(
        -5.0,
        5.0,
        size=(500, 3),
    )

    resolution = np.full(
        500,
        0.05,
    )

    reasons = np.full(
        500,
        "SEMANTIC",
        dtype="U16",
    )

    result = build_hierarchical_foveated_map(
        xyz,
        resolution,
        reasons,
    )

    assert np.all(
        result.level == 0
    )