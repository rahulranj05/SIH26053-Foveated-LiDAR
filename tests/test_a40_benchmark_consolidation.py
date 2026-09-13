from __future__ import annotations

import json
from pathlib import Path

from scripts.a40_benchmark_consolidation import (
    BENCHMARKS,
    BenchmarkResult,
    ConsolidatedReport,
    save_report,
)


ROOT = Path(__file__).resolve().parents[1]


def test_all_final_benchmark_scripts_exist():
    missing = [
        script
        for _, _, script in BENCHMARKS
        if not (ROOT / script).is_file()
    ]

    assert missing == []


def test_benchmark_inventory_contains_a33_to_a38():
    milestones = [milestone for milestone, _, _ in BENCHMARKS]

    assert milestones == [
        "A33",
        "A34",
        "A35",
        "A36",
        "A37",
        "A38",
    ]


def test_benchmark_names_are_unique():
    names = [name for _, name, _ in BENCHMARKS]

    assert len(names) == len(set(names))


def test_benchmark_scripts_are_unique():
    scripts = [script for _, _, script in BENCHMARKS]

    assert len(scripts) == len(set(scripts))


def test_consolidated_report_dataclass():
    report = ConsolidatedReport(
        generated_at_utc="2026-01-01T00:00:00+00:00",
        benchmark_count=1,
        passed_count=1,
        failed_count=0,
        overall_pass=True,
        benchmarks=(
            BenchmarkResult(
                milestone="A33",
                name="Terrain Accuracy",
                script="scripts/benchmark_a33_terrain_accuracy.py",
                return_code=0,
                passed=True,
                output="PASS",
            ),
        ),
    )

    assert report.benchmark_count == 1
    assert report.passed_count == 1
    assert report.failed_count == 0
    assert report.overall_pass is True
    assert len(report.benchmarks) == 1


def test_report_can_be_serialized(tmp_path):
    report = ConsolidatedReport(
        generated_at_utc="2026-01-01T00:00:00+00:00",
        benchmark_count=1,
        passed_count=1,
        failed_count=0,
        overall_pass=True,
        benchmarks=(
            BenchmarkResult(
                milestone="A33",
                name="Terrain Accuracy",
                script="scripts/benchmark_a33_terrain_accuracy.py",
                return_code=0,
                passed=True,
                output="PASS",
            ),
        ),
    )

    output = tmp_path / "a40_report.json"
    save_report(report, output)

    assert output.is_file()

    payload = json.loads(
        output.read_text(encoding="utf-8")
    )

    assert payload["overall_pass"] is True
    assert payload["benchmark_count"] == 1
    assert payload["failed_count"] == 0