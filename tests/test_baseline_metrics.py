from __future__ import annotations

import numpy as np
import pytest

from evaluation.baseline_metrics import (
    BaselineComparisonResult,
    BaselineMetrics,
    validate_distance_bins,
    validate_positive_resolution,
    validate_xyz,
)


def make_metrics(
    name: str,
    *,
    cells: int = 100,
    memory: int = 6400,
    latency: float = 10.0,
) -> BaselineMetrics:
    return BaselineMetrics(
        name=name,
        point_count=1000,
        active_cells=cells,
        resolution_counts={0.05: cells},
        memory_bytes=memory,
        latency_ms=latency,
    )


def test_baseline_metrics_properties():
    metrics = make_metrics("test")

    assert metrics.points_per_active_cell == pytest.approx(10.0)
    assert metrics.memory_mb == pytest.approx(6400 / (1024**2))
    assert metrics.update_rate_hz == pytest.approx(100.0)
    assert metrics.finest_resolution_m == pytest.approx(0.05)


def test_zero_cells_are_safe():
    metrics = make_metrics(
        "empty",
        cells=0,
        memory=0,
    )

    assert metrics.points_per_active_cell == 0.0
    assert metrics.memory_mb == 0.0


def test_as_dict_contains_expected_fields():
    metrics = make_metrics("test")

    payload = metrics.as_dict()

    assert payload["name"] == "test"
    assert payload["point_count"] == 1000
    assert payload["active_cells"] == 100
    assert payload["resolution_counts"] == {0.05: 100}
    assert "memory_mb" in payload
    assert "update_rate_hz" in payload


def test_comparison_lookup():
    comparison = BaselineComparisonResult(
        baselines=(
            make_metrics("A"),
            make_metrics("B"),
        )
    )

    assert comparison.by_name("A").name == "A"
    assert comparison.names == ("A", "B")


def test_comparison_unknown_name_rejected():
    comparison = BaselineComparisonResult(
        baselines=(make_metrics("A"),)
    )

    with pytest.raises(KeyError):
        comparison.by_name("missing")


def test_memory_reduction():
    comparison = BaselineComparisonResult(
        baselines=(
            make_metrics("reference", memory=1000),
            make_metrics("candidate", memory=600),
        )
    )

    assert comparison.memory_reduction_percent(
        "reference",
        "candidate",
    ) == pytest.approx(40.0)


def test_cell_reduction():
    comparison = BaselineComparisonResult(
        baselines=(
            make_metrics("reference", cells=1000),
            make_metrics("candidate", cells=750),
        )
    )

    assert comparison.active_cell_reduction_percent(
        "reference",
        "candidate",
    ) == pytest.approx(25.0)


def test_latency_reduction():
    comparison = BaselineComparisonResult(
        baselines=(
            make_metrics("reference", latency=100.0),
            make_metrics("candidate", latency=75.0),
        )
    )

    assert comparison.latency_reduction_percent(
        "reference",
        "candidate",
    ) == pytest.approx(25.0)


def test_zero_reference_reductions_are_safe():
    comparison = BaselineComparisonResult(
        baselines=(
            make_metrics(
                "reference",
                cells=0,
                memory=0,
                latency=0.0,
            ),
            make_metrics("candidate"),
        )
    )

    assert comparison.memory_reduction_percent(
        "reference",
        "candidate",
    ) == 0.0

    assert comparison.active_cell_reduction_percent(
        "reference",
        "candidate",
    ) == 0.0

    assert comparison.latency_reduction_percent(
        "reference",
        "candidate",
    ) == 0.0


def test_comparison_as_dict():
    comparison = BaselineComparisonResult(
        baselines=(make_metrics("A"),)
    )

    payload = comparison.as_dict()

    assert len(payload["baselines"]) == 1
    assert payload["baselines"][0]["name"] == "A"


def test_validate_xyz_accepts_valid_points():
    xyz = np.zeros((4, 3))

    result = validate_xyz(xyz)

    assert result.shape == (4, 3)
    assert result.dtype == np.float64


def test_validate_xyz_rejects_wrong_shape():
    with pytest.raises(ValueError):
        validate_xyz(np.zeros((4, 2)))


def test_validate_xyz_rejects_non_finite_values():
    xyz = np.zeros((4, 3))
    xyz[0, 0] = np.nan

    with pytest.raises(ValueError):
        validate_xyz(xyz)


def test_validate_resolution():
    assert validate_positive_resolution(0.05) == pytest.approx(0.05)

    with pytest.raises(ValueError):
        validate_positive_resolution(0.0)

    with pytest.raises(ValueError):
        validate_positive_resolution(-0.1)

    with pytest.raises(ValueError):
        validate_positive_resolution(np.nan)


def test_validate_distance_bins():
    assert validate_distance_bins(
        (0.0, 10.0, 25.0)
    ) == (0.0, 10.0, 25.0)

    with pytest.raises(ValueError):
        validate_distance_bins((0.0,))

    with pytest.raises(ValueError):
        validate_distance_bins((0.0, 10.0, 5.0))

    with pytest.raises(ValueError):
        validate_distance_bins((-1.0, 10.0))