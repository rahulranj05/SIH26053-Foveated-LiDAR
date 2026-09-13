from __future__ import annotations

import numpy as np
import pytest

from evaluation.dynamic_object_metrics import (
    BinaryDetectionMetrics,
    DistanceBinMetrics,
    DynamicObjectEvaluationResult,
    distance_binned_dynamic_metrics,
    dynamic_detection_metrics,
    evaluate_dynamic_objects,
)


def test_perfect_dynamic_detection():
    reference = np.array([0, 1, 0, 1, 1], dtype=np.int8)
    probability = np.array([0.1, 0.9, 0.2, 0.8, 1.0], dtype=np.float64)

    metrics = dynamic_detection_metrics(
        reference,
        probability,
        threshold=0.5,
    )

    assert isinstance(metrics, BinaryDetectionMetrics)
    assert metrics.true_positive == 3
    assert metrics.false_positive == 0
    assert metrics.true_negative == 2
    assert metrics.false_negative == 0

    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(1.0)
    assert metrics.false_positive_rate == pytest.approx(0.0)


def test_false_positive_and_false_negative_counts():
    reference = np.array([1, 1, 0, 0], dtype=np.int8)
    probability = np.array([0.2, 0.9, 0.8, 0.1], dtype=np.float64)

    metrics = dynamic_detection_metrics(
        reference,
        probability,
        threshold=0.5,
    )

    assert metrics.true_positive == 1
    assert metrics.false_positive == 1
    assert metrics.true_negative == 1
    assert metrics.false_negative == 1

    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(0.5)
    assert metrics.false_positive_rate == pytest.approx(0.5)


def test_all_dynamic_objects_missed():
    reference = np.array([1, 1, 1], dtype=np.int8)
    probability = np.array([0.1, 0.2, 0.4], dtype=np.float64)

    metrics = dynamic_detection_metrics(
        reference,
        probability,
        threshold=0.5,
    )

    assert metrics.true_positive == 0
    assert metrics.false_negative == 3
    assert metrics.false_positive == 0
    assert metrics.true_negative == 0

    assert metrics.precision == pytest.approx(0.0)
    assert metrics.recall == pytest.approx(0.0)
    assert metrics.f1 == pytest.approx(0.0)
    assert metrics.false_positive_rate == pytest.approx(0.0)


def test_all_samples_positive_has_zero_false_positive_rate():
    reference = np.array([1, 1, 1, 1], dtype=np.int8)
    probability = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float64)

    metrics = dynamic_detection_metrics(
        reference,
        probability,
        threshold=0.5,
    )

    assert metrics.true_positive == 4
    assert metrics.false_positive == 0
    assert metrics.true_negative == 0
    assert metrics.false_negative == 0

    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(1.0)
    assert metrics.false_positive_rate == pytest.approx(0.0)


def test_no_positive_reference_has_zero_recall():
    reference = np.array([0, 0, 0, 0], dtype=np.int8)
    probability = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64)

    metrics = dynamic_detection_metrics(
        reference,
        probability,
        threshold=0.5,
    )

    assert metrics.true_positive == 0
    assert metrics.false_negative == 0
    assert metrics.true_negative == 2
    assert metrics.false_positive == 2

    assert metrics.recall == pytest.approx(0.0)
    assert metrics.precision == pytest.approx(0.0)
    assert metrics.f1 == pytest.approx(0.0)
    assert metrics.false_positive_rate == pytest.approx(0.5)


def test_threshold_changes_detection():
    reference = np.array([1, 1, 0, 0], dtype=np.int8)

    # No probability is exactly equal to either threshold.
    probability = np.array(
        [0.30, 0.55, 0.45, 0.70],
        dtype=np.float64,
    )

    low_threshold = dynamic_detection_metrics(
        reference,
        probability,
        threshold=0.40,
    )

    high_threshold = dynamic_detection_metrics(
        reference,
        probability,
        threshold=0.60,
    )

    # threshold = 0.40
    #
    # reference:  [1, 1, 0, 0]
    # prediction: [0, 1, 1, 1]
    #
    # TP=1, FP=2, TN=0, FN=1
    assert low_threshold.true_positive == 1
    assert low_threshold.false_positive == 2
    assert low_threshold.true_negative == 0
    assert low_threshold.false_negative == 1

    # threshold = 0.60
    #
    # reference:  [1, 1, 0, 0]
    # prediction: [0, 0, 0, 1]
    #
    # TP=0, FP=1, TN=1, FN=2
    assert high_threshold.true_positive == 0
    assert high_threshold.false_positive == 1
    assert high_threshold.true_negative == 1
    assert high_threshold.false_negative == 2


def test_distance_binned_metrics():
    reference = np.array(
        [1, 0, 1, 0, 1, 0],
        dtype=np.int8,
    )

    probability = np.array(
        [0.9, 0.1, 0.8, 0.2, 0.9, 0.1],
        dtype=np.float64,
    )

    distances = np.array(
        [5.0, 7.0, 15.0, 20.0, 30.0, 40.0],
        dtype=np.float64,
    )

    results = distance_binned_dynamic_metrics(
        reference,
        probability,
        distances,
        bins=(0.0, 10.0, 25.0, 50.0),
        threshold=0.5,
    )

    assert len(results) == 3
    assert all(
        isinstance(item, DistanceBinMetrics)
        for item in results
    )

    assert results[0].min_distance_m == pytest.approx(0.0)
    assert results[0].max_distance_m == pytest.approx(10.0)
    assert results[0].sample_count == 2

    assert results[1].min_distance_m == pytest.approx(10.0)
    assert results[1].max_distance_m == pytest.approx(25.0)
    assert results[1].sample_count == 2

    assert results[2].min_distance_m == pytest.approx(25.0)
    assert results[2].max_distance_m == pytest.approx(50.0)
    assert results[2].sample_count == 2


def test_distance_bin_boundary_belongs_to_next_bin():
    reference = np.array(
        [1, 0],
        dtype=np.int8,
    )

    probability = np.array(
        [0.9, 0.1],
        dtype=np.float64,
    )

    distances = np.array(
        [10.0, 20.0],
        dtype=np.float64,
    )

    results = distance_binned_dynamic_metrics(
        reference,
        probability,
        distances,
        bins=(0.0, 10.0, 20.0),
    )

    # Distance bins use:
    #
    #   [0, 10)
    #   [10, 20]
    #
    # Therefore 10.0 belongs to the second bin.
    assert results[0].sample_count == 0
    assert results[1].sample_count == 2

    assert results[1].metrics.true_positive == 1
    assert results[1].metrics.true_negative == 1


def test_complete_evaluation_contains_overall_and_distance_bins():
    reference = np.array(
        [1, 0, 1, 0, 1, 0],
        dtype=np.int8,
    )

    probability = np.array(
        [0.9, 0.1, 0.8, 0.7, 0.2, 0.1],
        dtype=np.float64,
    )

    distances = np.array(
        [5.0, 8.0, 15.0, 30.0, 60.0, 90.0],
        dtype=np.float64,
    )

    result = evaluate_dynamic_objects(
        reference,
        probability,
        distances,
        threshold=0.5,
        bins=(0.0, 10.0, 25.0, 50.0, 75.0, 100.0),
    )

    assert isinstance(result, DynamicObjectEvaluationResult)
    assert isinstance(result.overall, BinaryDetectionMetrics)

    assert len(result.distance_bins) == 5
    assert result.threshold == pytest.approx(0.5)
    assert result.total_points == 6
    assert result.valid_points == 6

    assert sum(
        item.sample_count
        for item in result.distance_bins
    ) == 6


def test_as_dict_structure():
    reference = np.array(
        [1, 0, 1, 0],
        dtype=np.int8,
    )

    probability = np.array(
        [0.9, 0.1, 0.8, 0.2],
        dtype=np.float64,
    )

    distances = np.array(
        [5.0, 8.0, 15.0, 30.0],
        dtype=np.float64,
    )

    result = evaluate_dynamic_objects(
        reference,
        probability,
        distances,
    )

    payload = result.as_dict()

    assert isinstance(payload, dict)

    assert "overall" in payload
    assert "distance_bins" in payload
    assert "threshold" in payload
    assert "total_points" in payload
    assert "valid_points" in payload

    assert isinstance(payload["overall"], dict)
    assert isinstance(payload["distance_bins"], list)

    assert payload["threshold"] == pytest.approx(0.5)
    assert payload["total_points"] == 4
    assert payload["valid_points"] == 4

    assert set(payload["overall"]) >= {
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
    }


def test_mismatched_probability_length_is_rejected():
    reference = np.array(
        [1, 0, 1],
        dtype=np.int8,
    )

    probability = np.array(
        [0.9, 0.1],
        dtype=np.float64,
    )

    with pytest.raises(ValueError):
        dynamic_detection_metrics(
            reference,
            probability,
        )


def test_invalid_probability_range_is_rejected():
    reference = np.array(
        [1, 0],
        dtype=np.int8,
    )

    probability = np.array(
        [0.9, 1.2],
        dtype=np.float64,
    )

    with pytest.raises(ValueError):
        dynamic_detection_metrics(
            reference,
            probability,
        )


def test_invalid_threshold_is_rejected():
    reference = np.array(
        [1, 0],
        dtype=np.int8,
    )

    probability = np.array(
        [0.9, 0.1],
        dtype=np.float64,
    )

    with pytest.raises(ValueError):
        dynamic_detection_metrics(
            reference,
            probability,
            threshold=1.5,
        )


def test_invalid_distances_are_rejected():
    reference = np.array(
        [1, 0],
        dtype=np.int8,
    )

    probability = np.array(
        [0.9, 0.1],
        dtype=np.float64,
    )

    distances = np.array(
        [5.0, -1.0],
        dtype=np.float64,
    )

    with pytest.raises(ValueError):
        distance_binned_dynamic_metrics(
            reference,
            probability,
            distances,
        )


def test_invalid_distance_bins_are_rejected():
    reference = np.array(
        [1, 0],
        dtype=np.int8,
    )

    probability = np.array(
        [0.9, 0.1],
        dtype=np.float64,
    )

    distances = np.array(
        [5.0, 15.0],
        dtype=np.float64,
    )

    with pytest.raises(ValueError):
        distance_binned_dynamic_metrics(
            reference,
            probability,
            distances,
            bins=(0.0, 20.0, 10.0),
        )


def test_metrics_are_finite():
    reference = np.array(
        [1, 0, 1, 0],
        dtype=np.int8,
    )

    probability = np.array(
        [0.9, 0.1, 0.8, 0.2],
        dtype=np.float64,
    )

    metrics = dynamic_detection_metrics(
        reference,
        probability,
    )

    values = (
        metrics.precision,
        metrics.recall,
        metrics.f1,
        metrics.false_positive_rate,
    )

    assert all(
        np.isfinite(value)
        for value in values
    )