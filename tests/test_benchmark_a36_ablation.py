from __future__ import annotations

import numpy as np

from evaluation.ablation import (
    ABLATION_VARIANTS,
    run_ablation_study,
)
from scripts.benchmark_a36_ablation import (
    build_demo_point_cloud,
    build_demo_signals,
)


def test_a36_demo_point_cloud_is_deterministic():
    first = build_demo_point_cloud(256)
    second = build_demo_point_cloud(256)

    np.testing.assert_array_equal(first, second)


def test_a36_demo_signals_match_point_count():
    xyz = build_demo_point_cloud(256)
    signals = build_demo_signals(xyz)

    assert set(signals) == {
        "DISTANCE",
        "KINEMATIC",
        "PREDICTED_PATH",
        "SEMANTIC",
        "DYNAMIC",
    }

    for values in signals.values():
        assert values.shape == (256,)
        assert np.all(np.isfinite(values))


def test_a36_benchmark_runs_all_variants():
    xyz = build_demo_point_cloud(256)
    signals = build_demo_signals(xyz)

    result = run_ablation_study(
        xyz,
        signals,
    )

    assert tuple(
        variant.name
        for variant in result.variants
    ) == ABLATION_VARIANTS


def test_a36_all_variants_preserve_point_count():
    xyz = build_demo_point_cloud(256)
    signals = build_demo_signals(xyz)

    result = run_ablation_study(
        xyz,
        signals,
    )

    for variant in result.variants:
        assert variant.point_count == len(xyz)


def test_a36_all_variants_have_positive_map_size():
    xyz = build_demo_point_cloud(256)
    signals = build_demo_signals(xyz)

    result = run_ablation_study(
        xyz,
        signals,
    )

    for variant in result.variants:
        assert variant.active_cells > 0
        assert variant.memory_bytes > 0


def test_a36_ablation_variants_change_resolution_distribution():
    xyz = build_demo_point_cloud(256)
    signals = build_demo_signals(xyz)

    result = run_ablation_study(
        xyz,
        signals,
    )

    distributions = {
        variant.name: tuple(
            sorted(variant.resolution_counts.items())
        )
        for variant in result.variants
    }

    assert len(set(distributions.values())) > 1


def test_a36_ablation_variants_change_dominant_reason_distribution():
    xyz = build_demo_point_cloud(256)
    signals = build_demo_signals(xyz)

    result = run_ablation_study(
        xyz,
        signals,
    )

    distance_only = result.by_name(
        "DISTANCE_ONLY"
    )

    full = result.by_name(
        "FULL_FOVEAMAP"
    )

    assert distance_only.dominant_reason_counts == {
        "DISTANCE": len(xyz)
    }

    assert len(full.dominant_reason_counts) > 1


def test_a36_full_foveamap_has_non_distance_contributions():
    xyz = build_demo_point_cloud(256)
    signals = build_demo_signals(xyz)

    result = run_ablation_study(
        xyz,
        signals,
    )

    full = result.full_foveamap

    non_distance_count = sum(
        count
        for reason, count in full.dominant_reason_counts.items()
        if reason != "DISTANCE"
    )

    assert non_distance_count > 0