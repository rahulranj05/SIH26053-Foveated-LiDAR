from __future__ import annotations

import numpy as np
import pytest

from mapping.baselines import (
    DEFAULT_ADAPTIVE_RESOLUTIONS,
    DEFAULT_DISTANCE_BINS,
    distance_only_adaptive_baseline,
    uniform_2d_25d_baseline,
    uniform_3d_baseline,
)


def synthetic_cloud() -> np.ndarray:
    return np.array(
        [
            [0.01, 0.01, 0.10],
            [0.04, 0.01, 0.12],
            [0.06, 0.01, 0.15],
            [0.11, 0.01, 0.20],
            [0.19, 0.01, 0.21],
            [0.21, 0.01, 0.22],
            [1.01, 1.01, 0.50],
            [1.04, 1.01, 0.52],
            [11.0, 0.0, 1.0],
            [11.1, 0.0, 1.1],
            [30.0, 0.0, 2.0],
            [30.1, 0.0, 2.1],
            [60.0, 0.0, 3.0],
            [60.1, 0.0, 3.1],
        ],
        dtype=np.float64,
    )


def test_uniform_2d_25d_creates_cells():
    xyz = synthetic_cloud()

    result, latency = uniform_2d_25d_baseline(
        xyz,
        resolution_m=0.05,
    )

    assert result.name == "Uniform 2.5D"
    assert result.point_count == len(xyz)
    assert result.active_cells > 0
    assert latency >= 0.0


def test_uniform_2d_25d_aggregates_z_statistics():
    xyz = np.array(
        [
            [0.01, 0.01, 1.0],
            [0.02, 0.02, 3.0],
        ],
        dtype=np.float64,
    )

    result, _ = uniform_2d_25d_baseline(
        xyz,
        resolution_m=0.05,
    )

    assert result.active_cells == 1

    cell = result.cells[0]

    assert cell.point_count == 2
    assert cell.z_min == pytest.approx(1.0)
    assert cell.z_max == pytest.approx(3.0)
    assert cell.z_mean == pytest.approx(2.0)
    assert cell.z_variance == pytest.approx(1.0)


def test_uniform_2d_negative_coordinates_are_supported():
    xyz = np.array(
        [
            [-0.01, -0.01, 0.0],
            [-0.02, -0.02, 1.0],
        ],
        dtype=np.float64,
    )

    result, _ = uniform_2d_25d_baseline(
        xyz,
        resolution_m=0.05,
    )

    assert result.active_cells == 1


def test_uniform_2d_empty_cloud():
    xyz = np.empty((0, 3), dtype=np.float64)

    result, latency = uniform_2d_25d_baseline(xyz)

    assert result.point_count == 0
    assert result.active_cells == 0
    assert result.resolution_counts == {}
    assert latency >= 0.0


def test_uniform_3d_creates_occupancy_voxels():
    xyz = synthetic_cloud()

    result, latency = uniform_3d_baseline(
        xyz,
        voxel_size_m=0.05,
    )

    assert result.name == "Uniform 3D"
    assert result.point_count == len(xyz)
    assert result.active_cells > 0
    assert latency >= 0.0


def test_uniform_3d_collapses_identical_voxels():
    xyz = np.array(
        [
            [0.01, 0.01, 0.01],
            [0.02, 0.02, 0.02],
            [0.03, 0.03, 0.03],
        ],
        dtype=np.float64,
    )

    result, _ = uniform_3d_baseline(
        xyz,
        voxel_size_m=0.05,
    )

    assert result.active_cells == 1


def test_uniform_3d_empty_cloud():
    xyz = np.empty((0, 3), dtype=np.float64)

    result, latency = uniform_3d_baseline(xyz)

    assert result.point_count == 0
    assert result.active_cells == 0
    assert latency >= 0.0


def test_distance_only_adaptive_uses_range():
    xyz = np.array(
        [
            [1.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
            [60.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    result, _ = distance_only_adaptive_baseline(
        xyz,
        distance_bins=DEFAULT_DISTANCE_BINS,
        resolutions=DEFAULT_ADAPTIVE_RESOLUTIONS,
    )

    assert set(result.resolution_counts) == {
        0.05,
        0.10,
        0.20,
        0.40,
    }


def test_distance_only_adaptive_near_points_are_finer():
    xyz = np.array(
        [
            [1.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    result, _ = distance_only_adaptive_baseline(
        xyz,
        distance_bins=(0.0, 10.0, 25.0),
        resolutions=(0.05, 0.10),
    )

    resolutions = sorted(
        cell.resolution_m
        for cell in result.cells
    )

    assert resolutions == [0.05, 0.10]


def test_distance_only_adaptive_exact_boundary_uses_next_range():
    xyz = np.array(
        [
            [10.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    result, _ = distance_only_adaptive_baseline(
        xyz,
        distance_bins=(0.0, 10.0, 25.0),
        resolutions=(0.05, 0.10),
    )

    assert result.cells[0].resolution_m == pytest.approx(0.10)


def test_distance_only_adaptive_rejects_resolution_count_mismatch():
    xyz = synthetic_cloud()

    with pytest.raises(ValueError):
        distance_only_adaptive_baseline(
            xyz,
            distance_bins=(0.0, 10.0, 25.0),
            resolutions=(0.05,),
        )


def test_baseline_memory_is_positive_for_populated_maps():
    xyz = synthetic_cloud()

    uniform, _ = uniform_2d_25d_baseline(xyz)
    three_d, _ = uniform_3d_baseline(xyz)

    assert uniform.memory_bytes > 0
    assert three_d.memory_bytes > 0