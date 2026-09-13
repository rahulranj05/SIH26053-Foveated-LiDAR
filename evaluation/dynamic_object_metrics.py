from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BinaryDetectionMetrics:
    """Binary dynamic-object detection metrics."""

    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    precision: float
    recall: float
    f1: float
    false_positive_rate: float

    support_positive: int
    support_negative: int


@dataclass(frozen=True)
class DistanceBinMetrics:
    """Dynamic-object detection metrics for one distance interval."""

    min_distance_m: float
    max_distance_m: float
    metrics: BinaryDetectionMetrics
    sample_count: int


@dataclass(frozen=True)
class DynamicObjectEvaluationResult:
    """Complete A34 dynamic-object evaluation result."""

    overall: BinaryDetectionMetrics
    distance_bins: tuple[DistanceBinMetrics, ...]
    threshold: float
    total_points: int
    valid_points: int

    def as_dict(self) -> dict[str, object]:
        return {
            "overall": {
                "true_positive": self.overall.true_positive,
                "false_positive": self.overall.false_positive,
                "true_negative": self.overall.true_negative,
                "false_negative": self.overall.false_negative,
                "precision": self.overall.precision,
                "recall": self.overall.recall,
                "f1": self.overall.f1,
                "false_positive_rate": self.overall.false_positive_rate,
                "support_positive": self.overall.support_positive,
                "support_negative": self.overall.support_negative,
            },
            "distance_bins": [
                {
                    "min_distance_m": item.min_distance_m,
                    "max_distance_m": item.max_distance_m,
                    "sample_count": item.sample_count,
                    "true_positive": item.metrics.true_positive,
                    "false_positive": item.metrics.false_positive,
                    "true_negative": item.metrics.true_negative,
                    "false_negative": item.metrics.false_negative,
                    "precision": item.metrics.precision,
                    "recall": item.metrics.recall,
                    "f1": item.metrics.f1,
                    "false_positive_rate": item.metrics.false_positive_rate,
                }
                for item in self.distance_bins
            ],
            "threshold": self.threshold,
            "total_points": self.total_points,
            "valid_points": self.valid_points,
        }


def _validate_binary_array(
    values: np.ndarray,
    name: str,
) -> np.ndarray:
    values = np.asarray(values)

    if values.ndim != 1:
        raise ValueError(
            f"{name} must be a one-dimensional array"
        )

    if values.size == 0:
        raise ValueError(
            f"{name} must not be empty"
        )

    if values.dtype.kind not in {"b", "i", "u"}:
        raise ValueError(
            f"{name} must contain boolean or integer values"
        )

    if not np.all((values == 0) | (values == 1)):
        raise ValueError(
            f"{name} must contain only 0/1 values"
        )

    return values.astype(bool, copy=False)


def _validate_probability_array(
    values: np.ndarray,
    name: str,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError(
            f"{name} must be a one-dimensional array"
        )

    if values.size == 0:
        raise ValueError(
            f"{name} must not be empty"
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"{name} must contain only finite values"
        )

    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(
            f"{name} must be within [0, 1]"
        )

    return values


def _validate_distances(
    distances: np.ndarray,
    expected_size: int,
) -> np.ndarray:
    distances = np.asarray(
        distances,
        dtype=np.float64,
    )

    if distances.ndim != 1:
        raise ValueError(
            "distances must be a one-dimensional array"
        )

    if distances.size != expected_size:
        raise ValueError(
            "distances must have the same length as "
            "the evaluation arrays"
        )

    if not np.all(np.isfinite(distances)):
        raise ValueError(
            "distances must contain only finite values"
        )

    if np.any(distances < 0.0):
        raise ValueError(
            "distances must be non-negative"
        )

    return distances


def _safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    if denominator == 0.0:
        return 0.0

    return float(numerator / denominator)


def _compute_metrics(
    reference: np.ndarray,
    predicted: np.ndarray,
) -> BinaryDetectionMetrics:
    reference = np.asarray(reference, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)

    true_positive = int(
        np.count_nonzero(reference & predicted)
    )

    false_positive = int(
        np.count_nonzero(~reference & predicted)
    )

    true_negative = int(
        np.count_nonzero(~reference & ~predicted)
    )

    false_negative = int(
        np.count_nonzero(reference & ~predicted)
    )

    support_positive = int(
        np.count_nonzero(reference)
    )

    support_negative = int(
        np.count_nonzero(~reference)
    )

    precision = _safe_divide(
        true_positive,
        true_positive + false_positive,
    )

    recall = _safe_divide(
        true_positive,
        true_positive + false_negative,
    )

    f1 = _safe_divide(
        2.0 * precision * recall,
        precision + recall,
    )

    false_positive_rate = _safe_divide(
        false_positive,
        false_positive + true_negative,
    )

    return BinaryDetectionMetrics(
        true_positive=true_positive,
        false_positive=false_positive,
        true_negative=true_negative,
        false_negative=false_negative,
        precision=precision,
        recall=recall,
        f1=f1,
        false_positive_rate=false_positive_rate,
        support_positive=support_positive,
        support_negative=support_negative,
    )


def dynamic_detection_metrics(
    reference_dynamic: np.ndarray,
    predicted_probability: np.ndarray,
    *,
    threshold: float = 0.5,
) -> BinaryDetectionMetrics:
    """
    Evaluate dynamic-object detection from reference labels and
    predicted dynamic probabilities.
    """

    reference_dynamic = _validate_binary_array(
        reference_dynamic,
        "reference_dynamic",
    )

    predicted_probability = _validate_probability_array(
        predicted_probability,
        "predicted_probability",
    )

    if reference_dynamic.size != predicted_probability.size:
        raise ValueError(
            "reference_dynamic and predicted_probability "
            "must have the same length"
        )

    if not np.isfinite(threshold):
        raise ValueError(
            "threshold must be finite"
        )

    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(
            "threshold must be within [0, 1]"
        )

    predicted_dynamic = (
        predicted_probability >= threshold
    )

    return _compute_metrics(
        reference_dynamic,
        predicted_dynamic,
    )


def distance_binned_dynamic_metrics(
    reference_dynamic: np.ndarray,
    predicted_probability: np.ndarray,
    distances: np.ndarray,
    *,
    threshold: float = 0.5,
    bins: tuple[float, ...] = (
        0.0,
        10.0,
        25.0,
        50.0,
        75.0,
        100.0,
    ),
) -> tuple[DistanceBinMetrics, ...]:
    """
    Evaluate dynamic-object detection independently in distance bins.
    """

    reference_dynamic = _validate_binary_array(
        reference_dynamic,
        "reference_dynamic",
    )

    predicted_probability = _validate_probability_array(
        predicted_probability,
        "predicted_probability",
    )

    if reference_dynamic.size != predicted_probability.size:
        raise ValueError(
            "reference_dynamic and predicted_probability "
            "must have the same length"
        )

    distances = _validate_distances(
        distances,
        reference_dynamic.size,
    )

    if not np.isfinite(threshold):
        raise ValueError(
            "threshold must be finite"
        )

    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(
            "threshold must be within [0, 1]"
        )

    edges = np.asarray(
        bins,
        dtype=np.float64,
    )

    if edges.ndim != 1 or edges.size < 2:
        raise ValueError(
            "bins must contain at least two boundaries"
        )

    if not np.all(np.isfinite(edges)):
        raise ValueError(
            "bins must contain only finite values"
        )

    if np.any(np.diff(edges) <= 0.0):
        raise ValueError(
            "bins must be strictly increasing"
        )

    predicted_dynamic = (
        predicted_probability >= threshold
    )

    results: list[DistanceBinMetrics] = []

    for index in range(edges.size - 1):
        lower = float(edges[index])
        upper = float(edges[index + 1])

        if index == edges.size - 2:
            mask = (
                (distances >= lower)
                & (distances <= upper)
            )
        else:
            mask = (
                (distances >= lower)
                & (distances < upper)
            )

        sample_count = int(
            np.count_nonzero(mask)
        )

        if sample_count == 0:
            metrics = _compute_metrics(
                np.array([False]),
                np.array([False]),
            )
        else:
            metrics = _compute_metrics(
                reference_dynamic[mask],
                predicted_dynamic[mask],
            )

        results.append(
            DistanceBinMetrics(
                min_distance_m=lower,
                max_distance_m=upper,
                metrics=metrics,
                sample_count=sample_count,
            )
        )

    return tuple(results)


def evaluate_dynamic_objects(
    reference_dynamic: np.ndarray,
    predicted_probability: np.ndarray,
    distances: np.ndarray,
    *,
    threshold: float = 0.5,
    bins: tuple[float, ...] = (
        0.0,
        10.0,
        25.0,
        50.0,
        75.0,
        100.0,
    ),
) -> DynamicObjectEvaluationResult:
    """
    Complete A34 dynamic-object evaluation.

    Evaluates:
      - precision
      - recall
      - F1
      - false-positive rate
      - distance-binned performance
    """

    reference_dynamic = _validate_binary_array(
        reference_dynamic,
        "reference_dynamic",
    )

    predicted_probability = _validate_probability_array(
        predicted_probability,
        "predicted_probability",
    )

    if reference_dynamic.size != predicted_probability.size:
        raise ValueError(
            "reference_dynamic and predicted_probability "
            "must have the same length"
        )

    distances = _validate_distances(
        distances,
        reference_dynamic.size,
    )

    overall = dynamic_detection_metrics(
        reference_dynamic,
        predicted_probability,
        threshold=threshold,
    )

    distance_bins = distance_binned_dynamic_metrics(
        reference_dynamic,
        predicted_probability,
        distances,
        threshold=threshold,
        bins=bins,
    )

    return DynamicObjectEvaluationResult(
        overall=overall,
        distance_bins=distance_bins,
        threshold=float(threshold),
        total_points=int(reference_dynamic.size),
        valid_points=int(reference_dynamic.size),
    )