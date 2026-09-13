from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from evaluation.dynamic_object_metrics import (
    DynamicObjectEvaluationResult,
    evaluate_dynamic_objects,
)


DISTANCE_BINS = (
    0.0,
    10.0,
    25.0,
    50.0,
    75.0,
    100.0,
)


@dataclass(frozen=True)
class DynamicBenchmarkCase:
    name: str
    reference: np.ndarray
    probability: np.ndarray
    distances: np.ndarray


@dataclass(frozen=True)
class DynamicBenchmarkReport:
    cases: tuple[
        tuple[
            DynamicBenchmarkCase,
            DynamicObjectEvaluationResult,
        ],
        ...,
    ]
    overall_precision: float
    overall_recall: float
    overall_f1: float
    overall_false_positive_rate: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    passed: bool

    @property
    def total_cases(self) -> int:
        return len(self.cases)

    @property
    def total_points(self) -> int:
        return sum(
            result.total_points
            for _case, result in self.cases
        )


def make_near_range_case() -> DynamicBenchmarkCase:
    return DynamicBenchmarkCase(
        name="Near-range moving objects",
        reference=np.array(
            [
                1,
                1,
                0,
                0,
                1,
                0,
            ],
            dtype=np.int8,
        ),
        probability=np.array(
            [
                0.95,
                0.88,
                0.08,
                0.12,
                0.91,
                0.20,
            ],
            dtype=np.float64,
        ),
        distances=np.array(
            [
                2.0,
                4.0,
                5.0,
                7.0,
                9.0,
                9.5,
            ],
            dtype=np.float64,
        ),
    )


def make_mid_range_case() -> DynamicBenchmarkCase:
    return DynamicBenchmarkCase(
        name="Mid-range vehicles",
        reference=np.array(
            [
                1,
                0,
                1,
                0,
                1,
                0,
                0,
                1,
            ],
            dtype=np.int8,
        ),
        probability=np.array(
            [
                0.92,
                0.18,
                0.84,
                0.23,
                0.79,
                0.15,
                0.31,
                0.86,
            ],
            dtype=np.float64,
        ),
        distances=np.array(
            [
                12.0,
                15.0,
                18.0,
                22.0,
                28.0,
                35.0,
                42.0,
                48.0,
            ],
            dtype=np.float64,
        ),
    )


def make_far_range_case() -> DynamicBenchmarkCase:
    return DynamicBenchmarkCase(
        name="Far-range dynamic objects",
        reference=np.array(
            [
                1,
                0,
                1,
                0,
                1,
                0,
                0,
                1,
            ],
            dtype=np.int8,
        ),
        probability=np.array(
            [
                0.74,
                0.19,
                0.68,
                0.28,
                0.62,
                0.22,
                0.36,
                0.71,
            ],
            dtype=np.float64,
        ),
        distances=np.array(
            [
                52.0,
                55.0,
                60.0,
                68.0,
                74.0,
                78.0,
                88.0,
                98.0,
            ],
            dtype=np.float64,
        ),
    )


def create_benchmark_cases() -> tuple[
    DynamicBenchmarkCase,
    ...,
]:
    return (
        make_near_range_case(),
        make_mid_range_case(),
        make_far_range_case(),
    )


def combine_cases(
    cases: tuple[
        DynamicBenchmarkCase,
        ...,
    ],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    reference = np.concatenate(
        [
            case.reference
            for case in cases
        ]
    )

    probability = np.concatenate(
        [
            case.probability
            for case in cases
        ]
    )

    distances = np.concatenate(
        [
            case.distances
            for case in cases
        ]
    )

    return (
        reference,
        probability,
        distances,
    )


def evaluate_benchmark_cases(
    cases: tuple[
        DynamicBenchmarkCase,
        ...,
    ],
    threshold: float = 0.5,
) -> tuple[
    tuple[
        DynamicBenchmarkCase,
        DynamicObjectEvaluationResult,
    ],
    ...,
]:
    results: list[
        tuple[
            DynamicBenchmarkCase,
            DynamicObjectEvaluationResult,
        ]
    ] = []

    for case in cases:
        result = evaluate_dynamic_objects(
            case.reference,
            case.probability,
            case.distances,
            threshold=threshold,
            bins=DISTANCE_BINS,
        )

        results.append(
            (
                case,
                result,
            )
        )

    return tuple(results)


def run_a34_benchmark(
    threshold: float = 0.5,
) -> DynamicBenchmarkReport:
    cases = create_benchmark_cases()

    case_results = evaluate_benchmark_cases(
        cases,
        threshold=threshold,
    )

    (
        reference,
        probability,
        distances,
    ) = combine_cases(cases)

    overall_result = evaluate_dynamic_objects(
        reference,
        probability,
        distances,
        threshold=threshold,
        bins=DISTANCE_BINS,
    )

    metrics = overall_result.overall

    passed = (
        metrics.precision >= 0.90
        and metrics.recall >= 0.90
        and metrics.f1 >= 0.90
        and metrics.false_positive_rate <= 0.10
    )

    return DynamicBenchmarkReport(
        cases=case_results,
        overall_precision=metrics.precision,
        overall_recall=metrics.recall,
        overall_f1=metrics.f1,
        overall_false_positive_rate=(
            metrics.false_positive_rate
        ),
        true_positive=metrics.true_positive,
        false_positive=metrics.false_positive,
        true_negative=metrics.true_negative,
        false_negative=metrics.false_negative,
        passed=passed,
    )


def print_case_result(
    case: DynamicBenchmarkCase,
    result: DynamicObjectEvaluationResult,
) -> None:
    metrics = result.overall

    print()
    print(case.name)
    print("-" * 78)

    print(
        f"Points              : "
        f"{result.total_points}"
    )

    print(
        f"Precision           : "
        f"{metrics.precision:.4f}"
    )

    print(
        f"Recall              : "
        f"{metrics.recall:.4f}"
    )

    print(
        f"F1 score            : "
        f"{metrics.f1:.4f}"
    )

    print(
        f"False-positive rate : "
        f"{metrics.false_positive_rate:.4f}"
    )

    print(
        f"TP / FP / TN / FN   : "
        f"{metrics.true_positive} / "
        f"{metrics.false_positive} / "
        f"{metrics.true_negative} / "
        f"{metrics.false_negative}"
    )

    print()
    print("Distance-binned metrics")

    for distance_bin in result.distance_bins:
        bin_metrics = distance_bin.metrics

        print(
            f"  "
            f"{distance_bin.min_distance_m:>5.1f}"
            f"–"
            f"{distance_bin.max_distance_m:<5.1f} m"
            f" | samples={distance_bin.sample_count:<3d}"
            f" | recall={bin_metrics.recall:.4f}"
            f" | precision={bin_metrics.precision:.4f}"
            f" | f1={bin_metrics.f1:.4f}"
        )


def print_benchmark_report(
    report: DynamicBenchmarkReport,
) -> None:
    print()
    print("=" * 78)
    print(
        "FOVEAMAP — A34 DYNAMIC-OBJECT "
        "DETECTION / RECALL EVALUATION"
    )
    print("=" * 78)

    print(
        f"Benchmark cases     : "
        f"{report.total_cases}"
    )

    print(
        f"Total points        : "
        f"{report.total_points}"
    )

    for case, result in report.cases:
        print_case_result(
            case,
            result,
        )

    print()
    print("=" * 78)
    print("OVERALL DYNAMIC-OBJECT METRICS")
    print("=" * 78)

    print(
        f"Precision           : "
        f"{report.overall_precision:.4f}"
    )

    print(
        f"Recall              : "
        f"{report.overall_recall:.4f}"
    )

    print(
        f"F1 score            : "
        f"{report.overall_f1:.4f}"
    )

    print(
        f"False-positive rate : "
        f"{report.overall_false_positive_rate:.4f}"
    )

    print(
        f"TP / FP / TN / FN   : "
        f"{report.true_positive} / "
        f"{report.false_positive} / "
        f"{report.true_negative} / "
        f"{report.false_negative}"
    )

    print()
    print("=" * 78)

    if report.passed:
        print("A34 STATUS: PASS")
    else:
        print("A34 STATUS: FAIL")

    print("=" * 78)


def main() -> None:
    report = run_a34_benchmark()

    print_benchmark_report(
        report
    )

    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()