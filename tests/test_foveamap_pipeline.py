"""
A20 — FoveaMap Full Pipeline Tests

Tests the integration boundary only.

A17 and A18.3 are not modified or mocked.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


# ============================================================================
# Repository setup
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from mapping.foveamap_pipeline import (  # noqa: E402
    FoveaMapPipelineResult,
    run_foveamap_pipeline,
    run_pipeline,
)


# ============================================================================
# Test data
# ============================================================================


def make_xyz(
    num_points: int = 256,
) -> np.ndarray:

    rng = np.random.default_rng(20260912)

    x = rng.uniform(
        -10.0,
        40.0,
        size=num_points,
    )

    y = rng.uniform(
        -10.0,
        10.0,
        size=num_points,
    )

    z = (
        0.05 * np.sin(x * 0.2)
        + 0.02 * np.cos(y * 0.3)
    )

    return np.column_stack(
        (x, y, z)
    ).astype(np.float64)


def make_signals(
    xyz: np.ndarray,
) -> dict[str, np.ndarray]:

    x = xyz[:, 0]
    y = xyz[:, 1]

    distance = np.sqrt(
        x * x + y * y
    )

    distance_importance = np.clip(
        1.0 - distance / 60.0,
        0.0,
        1.0,
    )

    kinematic_importance = np.exp(
        -0.5 * (y / 4.0) ** 2
    )

    predicted_path_importance = np.exp(
        -0.5
        * (
            (
                y
                - 0.02 * np.maximum(x, 0.0)
            )
            / 3.0
        ) ** 2
    )

    semantic_importance = np.exp(
        -0.5
        * (
            ((x - 15.0) / 4.0) ** 2
            + ((y - 2.0) / 2.5) ** 2
        )
    )

    dynamic_importance = (
        np.exp(
            -0.5
            * (
                (
                    y
                    + 0.02 * np.maximum(x, 0.0)
                )
                / 2.5
            ) ** 2
        )
        * np.clip(
            x / 35.0,
            0.0,
            1.0,
        )
    )

    return {
        "DISTANCE": distance_importance,
        "KINEMATIC": kinematic_importance,
        "PREDICTED_PATH": predicted_path_importance,
        "SEMANTIC": semantic_importance,
        "DYNAMIC": dynamic_importance,
    }


# ============================================================================
# Basic execution
# ============================================================================


def test_a20_returns_pipeline_result():

    xyz = make_xyz()
    signals = make_signals(xyz)

    result = run_foveamap_pipeline(
        xyz,
        signals,
    )

    assert isinstance(
        result,
        FoveaMapPipelineResult,
    )


def test_a20_exposes_foveation_output():

    xyz = make_xyz()
    signals = make_signals(xyz)

    result = run_foveamap_pipeline(
        xyz,
        signals,
    )

    assert set(
        result.foveation.keys()
    ) >= {
        "importance",
        "resolution",
        "dominant_reason",
    }


def test_a20_exposes_leaf_map():

    xyz = make_xyz()
    signals = make_signals(xyz)

    result = run_foveamap_pipeline(
        xyz,
        signals,
    )

    leaf_map = result.leaf_map

    assert leaf_map is not None
    assert leaf_map.active_cells >= 0


# ============================================================================
# A17 -> A18.3 propagation
# ============================================================================


def test_a20_resolution_reaches_mapper():

    xyz = make_xyz()
    signals = make_signals(xyz)

    result = run_foveamap_pipeline(
        xyz,
        signals,
    )

    resolutions = np.asarray(
        result.foveation["resolution"]
    )

    assert len(resolutions) == len(xyz)

    assert np.all(
        np.isin(
            resolutions,
            np.array(
                [0.05, 0.10, 0.20, 0.40]
            ),
        )
    )


def test_a20_dominant_reason_reaches_mapper():

    xyz = make_xyz()
    signals = make_signals(xyz)

    result = run_foveamap_pipeline(
        xyz,
        signals,
    )

    reasons = np.asarray(
        result.foveation["dominant_reason"]
    )

    assert len(reasons) == len(xyz)

    assert np.all(
        np.isin(
            reasons,
            np.array(
                [
                    "DISTANCE",
                    "KINEMATIC",
                    "PREDICTED_PATH",
                    "SEMANTIC",
                    "DYNAMIC",
                ]
            ),
        )
    )


# ============================================================================
# Point conservation
# ============================================================================


def test_a20_point_conservation():

    xyz = make_xyz()
    signals = make_signals(xyz)

    result = run_foveamap_pipeline(
        xyz,
        signals,
    )

    leaf_map = result.leaf_map

    point_count = np.asarray(
        leaf_map.point_count
    )

    assert int(
        leaf_map.input_points
    ) == len(xyz)

    assert int(
        np.sum(point_count)
    ) == len(xyz)

    assert np.all(
        point_count > 0
    )


# ============================================================================
# Hierarchy validity
# ============================================================================


def test_a20_valid_hierarchy_levels():

    xyz = make_xyz()
    signals = make_signals(xyz)

    result = run_foveamap_pipeline(
        xyz,
        signals,
    )

    levels = np.asarray(
        result.leaf_map.levels
    )

    assert np.all(
        np.isin(
            levels,
            np.array([0, 1, 2, 3]),
        )
    )


def test_a20_level_resolution_consistency():

    xyz = make_xyz()
    signals = make_signals(xyz)

    result = run_foveamap_pipeline(
        xyz,
        signals,
    )

    levels = np.asarray(
        result.leaf_map.levels
    )

    resolutions = np.asarray(
        result.leaf_map.resolutions
    )

    expected = {
        0: 0.05,
        1: 0.10,
        2: 0.20,
        3: 0.40,
    }

    for level, resolution in expected.items():

        mask = levels == level

        if np.any(mask):
            assert np.allclose(
                resolutions[mask],
                resolution,
            )


def test_a20_valid_resolutions():

    xyz = make_xyz()
    signals = make_signals(xyz)

    result = run_foveamap_pipeline(
        xyz,
        signals,
    )

    resolutions = np.asarray(
        result.leaf_map.resolutions
    )

    assert np.all(
        np.isin(
            resolutions,
            np.array(
                [0.05, 0.10, 0.20, 0.40]
            ),
        )
    )


# ============================================================================
# Leaf structure
# ============================================================================


def test_a20_leaf_arrays_have_consistent_lengths():

    xyz = make_xyz()
    signals = make_signals(xyz)

    result = run_foveamap_pipeline(
        xyz,
        signals,
    )

    leaf_map = result.leaf_map

    active_cells = int(
        leaf_map.active_cells
    )

    arrays = (
        leaf_map.levels,
        leaf_map.resolutions,
        leaf_map.ix,
        leaf_map.iy,
        leaf_map.z_min,
        leaf_map.z_max,
        leaf_map.z_mean,
        leaf_map.z_variance,
        leaf_map.point_count,
        leaf_map.dominant_reason,
    )

    for array in arrays:
        assert len(array) == active_cells


def test_a20_leaf_keys_are_unique():

    xyz = make_xyz()
    signals = make_signals(xyz)

    result = run_foveamap_pipeline(
        xyz,
        signals,
    )

    leaf_map = result.leaf_map

    levels = np.asarray(
        leaf_map.levels
    ).astype(np.int64)

    ix = np.asarray(
        leaf_map.ix
    ).astype(np.int64)

    iy = np.asarray(
        leaf_map.iy
    ).astype(np.int64)

    if len(levels) == 0:
        return

    keys = np.column_stack(
        (
            levels,
            ix,
            iy,
        )
    )

    assert len(
        np.unique(
            keys,
            axis=0,
        )
    ) == len(keys)


def test_a20_leaf_partition_is_valid():

    xyz = make_xyz()
    signals = make_signals(xyz)

    result = run_foveamap_pipeline(
        xyz,
        signals,
    )

    leaf_map = result.leaf_map

    levels = np.asarray(
        leaf_map.levels
    )

    ix = np.asarray(
        leaf_map.ix
    )

    iy = np.asarray(
        leaf_map.iy
    )

    resolutions = np.asarray(
        leaf_map.resolutions
    )

    if len(levels) == 0:
        return

    assert np.all(
        np.isfinite(ix)
    )

    assert np.all(
        np.isfinite(iy)
    )

    assert np.all(
        ix == np.floor(ix)
    )

    assert np.all(
        iy == np.floor(iy)
    )

    assert np.all(
        np.isfinite(resolutions)
    )


# ============================================================================
# Numeric validity
# ============================================================================


def test_a20_numeric_map_fields_are_valid():

    xyz = make_xyz()
    signals = make_signals(xyz)

    result = run_foveamap_pipeline(
        xyz,
        signals,
    )

    leaf_map = result.leaf_map

    z_min = np.asarray(
        leaf_map.z_min
    )

    z_max = np.asarray(
        leaf_map.z_max
    )

    z_mean = np.asarray(
        leaf_map.z_mean
    )

    z_variance = np.asarray(
        leaf_map.z_variance
    )

    assert np.all(
        np.isfinite(z_min)
    )

    assert np.all(
        np.isfinite(z_max)
    )

    assert np.all(
        np.isfinite(z_mean)
    )

    assert np.all(
        np.isfinite(z_variance)
    )

    assert np.all(
        z_variance >= 0.0
    )

    assert np.all(
        z_max >= z_min
    )


def test_a20_reasons_are_valid():

    xyz = make_xyz()
    signals = make_signals(xyz)

    result = run_foveamap_pipeline(
        xyz,
        signals,
    )

    reasons = np.asarray(
        result.leaf_map.dominant_reason
    )

    assert np.all(
        np.isin(
            reasons,
            np.array(
                [
                    "DISTANCE",
                    "KINEMATIC",
                    "PREDICTED_PATH",
                    "SEMANTIC",
                    "DYNAMIC",
                ]
            ),
        )
    )


# ============================================================================
# Timing
# ============================================================================


def test_a20_timings_are_present():

    xyz = make_xyz()
    signals = make_signals(xyz)

    result = run_foveamap_pipeline(
        xyz,
        signals,
    )

    timings = result.timings_ms

    assert set(timings) == {
        "controller_ms",
        "mapper_ms",
        "total_ms",
    }

    assert all(
        np.isfinite(value)
        for value in timings.values()
    )

    assert all(
        value >= 0.0
        for value in timings.values()
    )


def test_a20_total_time_covers_pipeline_stages():

    xyz = make_xyz()
    signals = make_signals(xyz)

    result = run_foveamap_pipeline(
        xyz,
        signals,
    )

    timings = result.timings_ms

    assert (
        timings["total_ms"]
        >= timings["controller_ms"]
    )

    assert (
        timings["total_ms"]
        >= timings["mapper_ms"]
    )


# ============================================================================
# Input validation
# ============================================================================


def test_a20_rejects_invalid_xyz_shape():

    xyz = np.zeros(
        (100, 2),
        dtype=np.float64,
    )

    signals = {
        "DISTANCE": np.zeros(100),
    }

    with pytest.raises(ValueError):
        run_foveamap_pipeline(
            xyz,
            signals,
        )


def test_a20_rejects_nonfinite_xyz():

    xyz = make_xyz()

    xyz[0, 0] = np.nan

    signals = make_signals(
        np.nan_to_num(xyz)
    )

    with pytest.raises(ValueError):
        run_foveamap_pipeline(
            xyz,
            signals,
        )


def test_a20_rejects_signal_length_mismatch():

    xyz = make_xyz(100)

    signals = {
        "DISTANCE": np.zeros(99),
    }

    with pytest.raises(ValueError):
        run_foveamap_pipeline(
            xyz,
            signals,
        )


def test_a20_rejects_nonfinite_signal():

    xyz = make_xyz()

    signals = make_signals(xyz)

    signals["DISTANCE"][0] = np.inf

    with pytest.raises(ValueError):
        run_foveamap_pipeline(
            xyz,
            signals,
        )


# ============================================================================
# Benchmark compatibility
# ============================================================================


def test_a20_benchmark_compatibility_wrapper():

    xyz = make_xyz()
    signals = make_signals(xyz)

    result, timings = run_pipeline(
        xyz,
        signals,
    )

    assert set(result) == {
        "foveation",
        "leaf_map",
    }

    assert set(timings) == {
        "controller_ms",
        "mapper_ms",
        "total_ms",
    }

    assert result["leaf_map"].active_cells >= 0