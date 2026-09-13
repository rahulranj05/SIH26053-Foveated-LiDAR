from __future__ import annotations

import numpy as np

from scripts.benchmark_a34_dynamic_objects import (
    DISTANCE_BINS,
    DynamicBenchmarkCase,
    combine_cases,
    create_benchmark_cases,
    evaluate_benchmark_cases,
    make_far_range_case,
    make_mid_range_case,
    make_near_range_case,
    run_a34_benchmark,
)


def test_individual_benchmark_cases_are_valid():
    cases = (
        make_near_range_case(),
        make_mid_range_case(),
        make_far_range_case(),
    )

    assert len(cases) == 3

    for case in cases:
        assert isinstance(
            case,
            DynamicBenchmarkCase,
        )

        assert (
            case.reference.ndim
            == 1
        )

        assert (
            case.probability.ndim
            == 1
        )

        assert (
            case.distances.ndim
            == 1
        )

        assert (
            case.reference.shape
            == case.probability.shape
        )

        assert (
            case.reference.shape
            == case.distances.shape
        )

        assert np.all(
            case.probability >= 0.0
        )

        assert np.all(
            case.probability <= 1.0
        )

        assert np.all(
            case.distances >= 0.0
        )


def test_create_benchmark_cases_returns_three_cases():
    cases = create_benchmark_cases()

    assert len(cases) == 3

    names = tuple(
        case.name
        for case in cases
    )

    assert names == (
        "Near-range moving objects",
        "Mid-range vehicles",
        "Far-range dynamic objects",
    )


def test_benchmark_distance_coverage():
    cases = create_benchmark_cases()

    (
        _reference,
        _probability,
        distances,
    ) = combine_cases(cases)

    assert np.any(
        distances < 10.0
    )

    assert np.any(
        (distances >= 10.0)
        & (distances < 25.0)
    )

    assert np.any(
        (distances >= 25.0)
        & (distances < 50.0)
    )

    assert np.any(
        (distances >= 50.0)
        & (distances < 75.0)
    )

    assert np.any(
        (distances >= 75.0)
        & (distances <= 100.0)
    )


def test_combine_cases_preserves_all_samples():
    cases = create_benchmark_cases()

    (
        reference,
        probability,
        distances,
    ) = combine_cases(cases)

    expected_count = sum(
        case.reference.size
        for case in cases
    )

    assert reference.size == expected_count
    assert probability.size == expected_count
    assert distances.size == expected_count

    assert reference.shape == probability.shape
    assert reference.shape == distances.shape


def test_each_case_can_be_evaluated():
    cases = create_benchmark_cases()

    results = evaluate_benchmark_cases(
        cases,
        threshold=0.5,
    )

    assert len(results) == len(cases)

    for (
        case,
        result,
    ) in results:
        assert result.total_points == (
            case.reference.size
        )

        assert result.valid_points == (
            case.reference.size
        )

        assert len(
            result.distance_bins
        ) == (
            len(DISTANCE_BINS) - 1
        )


def test_a34_benchmark_passes():
    report = run_a34_benchmark()

    assert report.passed is True


def test_a34_benchmark_metrics_are_strong():
    report = run_a34_benchmark()

    assert (
        report.overall_precision
        >= 0.90
    )

    assert (
        report.overall_recall
        >= 0.90
    )

    assert (
        report.overall_f1
        >= 0.90
    )

    assert (
        report.overall_false_positive_rate
        <= 0.10
    )


def test_a34_confusion_counts_are_consistent():
    report = run_a34_benchmark()

    total_confusion_count = (
        report.true_positive
        + report.false_positive
        + report.true_negative
        + report.false_negative
    )

    assert (
        total_confusion_count
        == report.total_points
    )


def test_a34_report_contains_all_cases():
    report = run_a34_benchmark()

    assert report.total_cases == 3

    assert len(
        report.cases
    ) == 3

    assert report.total_points > 0


def test_a34_threshold_can_be_changed():
    default_report = run_a34_benchmark(
        threshold=0.5,
    )

    strict_report = run_a34_benchmark(
        threshold=0.9,
    )

    assert default_report.total_points == (
        strict_report.total_points
    )

    assert (
        strict_report.overall_recall
        <= default_report.overall_recall
    )


def test_distance_bins_have_expected_boundaries():
    assert DISTANCE_BINS == (
        0.0,
        10.0,
        25.0,
        50.0,
        75.0,
        100.0,
    )


def test_all_benchmark_cases_contain_dynamic_objects():
    cases = create_benchmark_cases()

    for case in cases:
        assert np.any(
            case.reference == 1
        )


def test_all_benchmark_cases_contain_static_objects():
    cases = create_benchmark_cases()

    for case in cases:
        assert np.any(
            case.reference == 0
        )
        