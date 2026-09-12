from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from mapping.hierarchical_foveated_mapper import (
    build_hierarchical_foveated_map,
    validate_leaf_partition,
)


VALID_REASONS = np.array(
    [
        "DISTANCE",
        "KINEMATIC",
        "PREDICTED_PATH",
        "SEMANTIC",
        "DYNAMIC",
    ],
    dtype=object,
)


def make_reasons(rng, n):
    return VALID_REASONS[
        rng.integers(0, len(VALID_REASONS), size=n)
    ]


def test_all_points_are_conserved():
    rng = np.random.default_rng(42)

    xyz = rng.uniform(
        low=[-20.0, -20.0, -2.0],
        high=[20.0, 20.0, 3.0],
        size=(5000, 3),
    )

    desired_resolution = np.array(
        [0.05, 0.10, 0.20, 0.40],
        dtype=np.float64,
    )[rng.integers(0, 4, size=len(xyz))]

    dominant_reason = make_reasons(rng, len(xyz))

    result = build_hierarchical_foveated_map(
        xyz=xyz,
        desired_resolution=desired_resolution,
        dominant_reason=dominant_reason,
    )

    assert result.input_points == len(xyz)
    assert int(result.point_count.sum()) == len(xyz)


def test_no_duplicate_leaf_cells():
    rng = np.random.default_rng(43)

    xyz = rng.uniform(
        low=[-15.0, -15.0, -1.0],
        high=[15.0, 15.0, 2.0],
        size=(5000, 3),
    )

    desired_resolution = np.array(
        [0.05, 0.10, 0.20, 0.40],
        dtype=np.float64,
    )[rng.integers(0, 4, size=len(xyz))]

    dominant_reason = make_reasons(rng, len(xyz))

    result = build_hierarchical_foveated_map(
        xyz=xyz,
        desired_resolution=desired_resolution,
        dominant_reason=dominant_reason,
    )

    keys = np.column_stack(
        (
            result.level,
            result.ix,
            result.iy,
        )
    )

    unique_keys = np.unique(keys, axis=0)

    assert len(unique_keys) == result.active_cells


def test_leaf_partition_is_valid():
    rng = np.random.default_rng(44)

    xyz = rng.uniform(
        low=[-25.0, -25.0, -2.0],
        high=[25.0, 25.0, 4.0],
        size=(10000, 3),
    )

    desired_resolution = np.array(
        [0.05, 0.10, 0.20, 0.40],
        dtype=np.float64,
    )[rng.integers(0, 4, size=len(xyz))]

    dominant_reason = make_reasons(rng, len(xyz))

    result = build_hierarchical_foveated_map(
        xyz=xyz,
        desired_resolution=desired_resolution,
        dominant_reason=dominant_reason,
    )

    assert validate_leaf_partition(result)


def test_all_coarse_points_remain_level_three():
    rng = np.random.default_rng(45)

    xyz = rng.uniform(
        low=[-10.0, -10.0, -1.0],
        high=[10.0, 10.0, 2.0],
        size=(2000, 3),
    )

    desired_resolution = np.full(
        len(xyz),
        0.40,
        dtype=np.float64,
    )

    dominant_reason = np.full(
        len(xyz),
        "DISTANCE",
        dtype=object,
    )

    result = build_hierarchical_foveated_map(
        xyz=xyz,
        desired_resolution=desired_resolution,
        dominant_reason=dominant_reason,
    )

    assert np.all(result.level == 3)
    assert int(result.point_count.sum()) == len(xyz)


def test_all_fine_points_end_at_level_zero():
    rng = np.random.default_rng(46)

    xyz = rng.uniform(
        low=[-10.0, -10.0, -1.0],
        high=[10.0, 10.0, 2.0],
        size=(2000, 3),
    )

    desired_resolution = np.full(
        len(xyz),
        0.05,
        dtype=np.float64,
    )

    dominant_reason = np.full(
        len(xyz),
        "SEMANTIC",
        dtype=object,
    )

    result = build_hierarchical_foveated_map(
        xyz=xyz,
        desired_resolution=desired_resolution,
        dominant_reason=dominant_reason,
    )

    assert np.all(result.level == 0)
    assert int(result.point_count.sum()) == len(xyz)


def test_refinement_never_creates_an_ancestor_leaf():
    rng = np.random.default_rng(47)

    xyz = rng.uniform(
        low=[-20.0, -20.0, -1.0],
        high=[20.0, 20.0, 3.0],
        size=(8000, 3),
    )

    desired_resolution = np.array(
        [0.05, 0.10, 0.20, 0.40],
        dtype=np.float64,
    )[rng.integers(0, 4, size=len(xyz))]

    dominant_reason = np.full(
        len(xyz),
        "DISTANCE",
        dtype=object,
    )

    result = build_hierarchical_foveated_map(
        xyz=xyz,
        desired_resolution=desired_resolution,
        dominant_reason=dominant_reason,
    )

    levels = result.level
    ix = result.ix
    iy = result.iy

    leaf_keys = {
        (int(level), int(x), int(y))
        for level, x, y in zip(levels, ix, iy)
    }

    for level, x, y in leaf_keys:
        if level == 3:
            continue

        parent_level = level + 1
        parent_x = x // 2
        parent_y = y // 2

        assert (
            parent_level,
            parent_x,
            parent_y,
        ) not in leaf_keys


def test_leaf_alignment_is_valid():
    rng = np.random.default_rng(48)

    xyz = rng.uniform(
        low=[-30.0, -30.0, -2.0],
        high=[30.0, 30.0, 4.0],
        size=(10000, 3),
    )

    desired_resolution = np.array(
        [0.05, 0.10, 0.20, 0.40],
        dtype=np.float64,
    )[rng.integers(0, 4, size=len(xyz))]

    dominant_reason = make_reasons(rng, len(xyz))

    result = build_hierarchical_foveated_map(
        xyz=xyz,
        desired_resolution=desired_resolution,
        dominant_reason=dominant_reason,
    )

    for level, x, y in zip(
        result.level,
        result.ix,
        result.iy,
    ):
        level = int(level)
        x = int(x)
        y = int(y)

        if level < 3:
            parent_x = x // 2
            parent_y = y // 2

            assert isinstance(parent_x, int)
            assert isinstance(parent_y, int)


def test_reason_labels_are_valid():
    rng = np.random.default_rng(49)

    xyz = rng.uniform(
        low=[-20.0, -20.0, -1.0],
        high=[20.0, 20.0, 3.0],
        size=(5000, 3),
    )

    desired_resolution = np.array(
        [0.05, 0.10, 0.20, 0.40],
        dtype=np.float64,
    )[rng.integers(0, 4, size=len(xyz))]

    dominant_reason = make_reasons(rng, len(xyz))

    result = build_hierarchical_foveated_map(
        xyz=xyz,
        desired_resolution=desired_resolution,
        dominant_reason=dominant_reason,
    )

    assert set(result.dominant_reason).issubset(
        set(VALID_REASONS)
    )