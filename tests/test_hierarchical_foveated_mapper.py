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
    cells_are_unique,
    has_parent_child_overlap,
    level_distribution,
    reason_distribution,
    resolution_distribution,
    validate_leaf_partition,
    validate_point_conservation,
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


def test_fine_requirement_cannot_be_blocked_by_coarse_requirement():

    # Both points begin inside the same 40 cm parent cell.
    #
    # One point requires 5 cm resolution.
    # The other only requires 40 cm resolution.
    #
    # The fine point must still be represented at a fine resolution.
    xyz = np.array(
        [
            [1.00, 1.00, 1.0],
            [1.10, 1.10, 1.2],
        ],
        dtype=np.float64,
    )

    resolution = np.array(
        [
            0.05,
            0.40,
        ],
        dtype=np.float64,
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

    assert np.any(
        result.level == 0
    )

    assert np.sum(
        result.point_count[
            result.level == 0
        ]
    ) >= 1

    assert validate_point_conservation(
        result
    )

    assert not has_parent_child_overlap(
        result
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


def test_no_duplicate_final_cells():

    rng = np.random.default_rng(
        1357
    )

    xyz = rng.uniform(
        -25.0,
        25.0,
        size=(2000, 3),
    )

    resolution = rng.choice(
        [
            0.05,
            0.10,
            0.20,
            0.40,
        ],
        size=2000,
    )

    reasons = rng.choice(
        REASONS,
        size=2000,
    )

    result = build_hierarchical_foveated_map(
        xyz,
        resolution,
        reasons,
    )

    assert cells_are_unique(
        result
    )


def test_no_parent_child_overlap():

    rng = np.random.default_rng(
        2468
    )

    xyz = rng.uniform(
        -20.0,
        20.0,
        size=(2000, 3),
    )

    resolution = rng.choice(
        [
            0.05,
            0.10,
            0.20,
            0.40,
        ],
        size=2000,
    )

    reasons = rng.choice(
        REASONS,
        size=2000,
    )

    result = build_hierarchical_foveated_map(
        xyz,
        resolution,
        reasons,
    )

    assert not has_parent_child_overlap(
        result
    )


def test_full_leaf_partition_validation():

    rng = np.random.default_rng(
        314159
    )

    xyz = rng.uniform(
        -50.0,
        50.0,
        size=(5000, 3),
    )

    resolution = rng.choice(
        [
            0.05,
            0.10,
            0.20,
            0.40,
        ],
        size=5000,
    )

    reasons = rng.choice(
        REASONS,
        size=5000,
    )

    result = build_hierarchical_foveated_map(
        xyz,
        resolution,
        reasons,
    )

    assert validate_leaf_partition(
        result
    )


def test_exact_resolution_boundaries():

    # Points deliberately placed around exact cell boundaries.
    xyz = np.array(
        [
            [0.00, 0.00, 1.0],
            [0.05, 0.00, 1.1],
            [0.10, 0.00, 1.2],
            [0.20, 0.00, 1.3],
            [0.40, 0.00, 1.4],
            [0.80, 0.00, 1.5],
        ],
        dtype=np.float64,
    )

    resolution = np.full(
        xyz.shape[0],
        0.05,
        dtype=np.float64,
    )

    reasons = np.full(
        xyz.shape[0],
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

    assert (
        np.sum(
            result.point_count
        )
        == xyz.shape[0]
    )

    assert cells_are_unique(
        result
    )


def test_negative_coordinate_alignment():

    xyz = np.array(
        [
            [-1.00, -1.00, 1.0],
            [-0.95, -1.00, 1.1],
            [-0.90, -1.00, 1.2],
            [-0.80, -1.00, 1.3],
            [-0.75, -1.00, 1.4],
            [-0.70, -1.00, 1.5],
        ],
        dtype=np.float64,
    )

    resolution = np.full(
        xyz.shape[0],
        0.05,
        dtype=np.float64,
    )

    reasons = np.full(
        xyz.shape[0],
        "DISTANCE",
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

    assert validate_leaf_partition(
        result
    )

    assert cells_are_unique(
        result
    )


def test_z_statistics():

    xyz = np.array(
        [
            [0.01, 0.01, 1.0],
            [0.02, 0.02, 2.0],
            [0.03, 0.03, 3.0],
            [0.04, 0.04, 4.0],
        ],
        dtype=np.float64,
    )

    resolution = np.full(
        4,
        0.05,
        dtype=np.float64,
    )

    reasons = np.full(
        4,
        "SEMANTIC",
        dtype="U16",
    )

    result = build_hierarchical_foveated_map(
        xyz,
        resolution,
        reasons,
    )

    assert result.active_cells == 1

    assert np.isclose(
        result.z_min[0],
        1.0,
    )

    assert np.isclose(
        result.z_max[0],
        4.0,
    )

    assert np.isclose(
        result.z_mean[0],
        2.5,
    )

    assert np.isclose(
        result.z_variance[0],
        1.25,
    )

    assert result.point_count[0] == 4


def test_dominant_reason_tie_is_deterministic():

    xyz = np.array(
        [
            [0.01, 0.01, 1.0],
            [0.02, 0.02, 2.0],
        ],
        dtype=np.float64,
    )

    resolution = np.full(
        2,
        0.05,
        dtype=np.float64,
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

    assert result.active_cells == 1

    # REASON_NAMES ordering is deterministic:
    # DISTANCE precedes SEMANTIC.
    assert result.dominant_reason[0] == "DISTANCE"


def test_large_coordinate_stability():

    xyz = np.array(
        [
            [1_000_000.00, 1_000_000.00, 10.0],
            [1_000_000.05, 1_000_000.00, 11.0],
            [1_000_000.10, 1_000_000.00, 12.0],
            [1_000_000.15, 1_000_000.00, 13.0],
        ],
        dtype=np.float64,
    )

    resolution = np.full(
        4,
        0.05,
        dtype=np.float64,
    )

    reasons = np.full(
        4,
        "DYNAMIC",
        dtype="U16",
    )

    result = build_hierarchical_foveated_map(
        xyz,
        resolution,
        reasons,
    )

    assert validate_leaf_partition(
        result
    )

    assert validate_point_conservation(
        result
    )

    assert cells_are_unique(
        result
    )

    assert np.all(
        np.isfinite(
            result.z_mean
        )
    )

    assert np.all(
        np.isfinite(
            result.z_variance
        )
    )


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