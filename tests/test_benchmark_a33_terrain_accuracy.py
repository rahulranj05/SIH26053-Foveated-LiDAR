from __future__ import annotations

import math

from scripts.benchmark_a33_terrain_accuracy import (
    build_cases,
    run_benchmark,
    validate_results,
)


def test_a33_benchmark_builds_expected_terrain_cases():
    cases = build_cases()

    assert len(cases) == 4
    assert [case.name for case in cases] == [
        "Flat",
        "Linear slope",
        "Stepped terrain",
        "Uneven terrain",
    ]

    for case in cases:
        assert case.reference.shape == case.predicted.shape
        assert case.reference.ndim == 2
        assert case.resolution_m > 0.0


def test_a33_benchmark_results_are_complete_and_finite():
    results = run_benchmark()

    assert len(results) == 4

    for result in results.values():
        elevation = result.elevation.metrics
        slope = result.slope
        roughness = result.roughness

        values = [
            elevation.mae,
            elevation.rmse,
            elevation.bias,
            elevation.max_abs_error,
            elevation.p95_abs_error,
            slope.mae,
            slope.rmse,
            slope.bias,
            slope.max_abs_error,
            slope.p95_abs_error,
            roughness.mae,
            roughness.rmse,
            roughness.bias,
            roughness.max_abs_error,
            roughness.p95_abs_error,
        ]

        assert all(math.isfinite(float(value)) for value in values)
        assert result.elevation.coverage == 1.0
        assert elevation.valid_count > 0


def test_a33_benchmark_is_not_accidentally_perfect():
    results = run_benchmark()

    for result in results.values():
        assert result.elevation.metrics.rmse > 0.0


def test_a33_benchmark_validation_passes():
    results = run_benchmark()

    validate_results(results)