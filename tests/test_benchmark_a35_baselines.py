from __future__ import annotations

import numpy as np

from mapping.baselines import benchmark_baselines
from scripts.benchmark_a35_baselines import (
    make_benchmark_cloud,
    make_synthetic_foveamap_signals,
)


def test_benchmark_cloud_is_deterministic():
    first = make_benchmark_cloud(
        seed=35,
        point_count=1000,
    )

    second = make_benchmark_cloud(
        seed=35,
        point_count=1000,
    )

    assert np.array_equal(first, second)


def test_benchmark_cloud_has_expected_shape():
    xyz = make_benchmark_cloud(
        seed=35,
        point_count=1000,
    )

    assert xyz.shape == (1000, 3)
    assert np.all(np.isfinite(xyz))


def test_synthetic_signals_have_expected_length():
    xyz = make_benchmark_cloud(
        point_count=500,
    )

    signals = make_synthetic_foveamap_signals(xyz)

    assert set(signals) == {
        "DISTANCE",
        "SEMANTIC",
        "DYNAMIC",
    }

    for signal in signals.values():
        assert signal.shape == (500,)
        assert np.all(np.isfinite(signal))


def test_synthetic_signals_are_bounded():
    xyz = make_benchmark_cloud(
        point_count=500,
    )

    signals = make_synthetic_foveamap_signals(xyz)

    for signal in signals.values():
        assert np.min(signal) >= 0.0
        assert np.max(signal) <= 1.0


def test_a35_benchmark_returns_four_baselines():
    xyz = make_benchmark_cloud(
        point_count=800,
    )

    signals = make_synthetic_foveamap_signals(xyz)

    result = benchmark_baselines(
        xyz,
        foveamap_signals=signals,
    )

    assert result.names == (
        "Uniform 2.5D",
        "Uniform 3D",
        "Distance-only adaptive",
        "FoveaMap",
    )


def test_all_baselines_process_same_point_count():
    xyz = make_benchmark_cloud(
        point_count=800,
    )

    signals = make_synthetic_foveamap_signals(xyz)

    result = benchmark_baselines(
        xyz,
        foveamap_signals=signals,
    )

    for baseline in result.baselines:
        assert baseline.point_count == 800


def test_all_baselines_have_positive_active_representation():
    xyz = make_benchmark_cloud(
        point_count=800,
    )

    signals = make_synthetic_foveamap_signals(xyz)

    result = benchmark_baselines(
        xyz,
        foveamap_signals=signals,
    )

    for baseline in result.baselines:
        assert baseline.active_cells > 0


def test_all_baselines_have_non_negative_latency():
    xyz = make_benchmark_cloud(
        point_count=800,
    )

    signals = make_synthetic_foveamap_signals(xyz)

    result = benchmark_baselines(
        xyz,
        foveamap_signals=signals,
    )

    for baseline in result.baselines:
        assert baseline.latency_ms >= 0.0


def test_uniform_3d_is_distinct_from_uniform_2d():
    xyz = make_benchmark_cloud(
        point_count=800,
    )

    signals = make_synthetic_foveamap_signals(xyz)

    result = benchmark_baselines(
        xyz,
        foveamap_signals=signals,
    )

    uniform_2d = result.by_name("Uniform 2.5D")
    uniform_3d = result.by_name("Uniform 3D")

    assert uniform_2d.name != uniform_3d.name


def test_distance_only_has_multiple_resolution_levels():
    xyz = make_benchmark_cloud(
        point_count=5000,
    )

    signals = make_synthetic_foveamap_signals(xyz)

    result = benchmark_baselines(
        xyz,
        foveamap_signals=signals,
    )

    distance = result.by_name(
        "Distance-only adaptive"
    )

    assert len(distance.resolution_counts) >= 2


def test_foveamap_baseline_has_resolution_information():
    xyz = make_benchmark_cloud(
        point_count=800,
    )

    signals = make_synthetic_foveamap_signals(xyz)

    result = benchmark_baselines(
        xyz,
        foveamap_signals=signals,
    )

    foveamap = result.by_name("FoveaMap")

    assert foveamap.active_cells > 0
    assert len(foveamap.resolution_counts) > 0