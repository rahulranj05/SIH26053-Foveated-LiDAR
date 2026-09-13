from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

BENCHMARKS = (
    ("A33", "Terrain Accuracy", "scripts/benchmark_a33_terrain_accuracy.py"),
    ("A34", "Dynamic Objects", "scripts/benchmark_a34_dynamic_objects.py"),
    ("A35", "Baselines", "scripts/benchmark_a35_baselines.py"),
    ("A36", "Ablation", "scripts/benchmark_a36_ablation.py"),
    ("A37", "Performance", "scripts/benchmark_a37_performance.py"),
    ("A38", "Safety", "scripts/benchmark_a38_safety.py"),
)


@dataclass(frozen=True)
class BenchmarkResult:
    milestone: str
    name: str
    script: str
    return_code: int
    passed: bool
    output: str


@dataclass(frozen=True)
class ConsolidatedReport:
    generated_at_utc: str
    benchmark_count: int
    passed_count: int
    failed_count: int
    overall_pass: bool
    benchmarks: tuple[BenchmarkResult, ...]


def run_benchmark(
    milestone: str,
    name: str,
    script: str,
) -> BenchmarkResult:
    result = subprocess.run(
        [sys.executable, script],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    output = (result.stdout + "\n" + result.stderr).strip()

    return BenchmarkResult(
        milestone=milestone,
        name=name,
        script=script,
        return_code=result.returncode,
        passed=result.returncode == 0,
        output=output,
    )


def build_report() -> ConsolidatedReport:
    results = tuple(
        run_benchmark(milestone, name, script)
        for milestone, name, script in BENCHMARKS
    )

    passed = sum(result.passed for result in results)
    failed = len(results) - passed

    return ConsolidatedReport(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        benchmark_count=len(results),
        passed_count=passed,
        failed_count=failed,
        overall_pass=failed == 0,
        benchmarks=results,
    )


def save_report(
    report: ConsolidatedReport,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = asdict(report)

    output_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def print_report(report: ConsolidatedReport) -> None:
    print()
    print("=" * 78)
    print("FOVEAMAP — A40.2 FINAL BENCHMARK CONSOLIDATION")
    print("=" * 78)

    for result in report.benchmarks:
        status = "PASS" if result.passed else "FAIL"

        print()
        print("-" * 78)
        print(
            f"[{status}] {result.milestone} — {result.name}"
        )
        print(f"Script: {result.script}")
        print("-" * 78)
        print(result.output)

    print()
    print("=" * 78)
    print(
        f"Benchmarks : {report.benchmark_count}\n"
        f"Passed     : {report.passed_count}\n"
        f"Failed     : {report.failed_count}"
    )
    print(
        "OVERALL    : "
        + ("PASS" if report.overall_pass else "FAIL")
    )
    print("=" * 78)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run and consolidate all final FoveaMap benchmarks."
    )

    parser.add_argument(
        "--output",
        default="outputs/a40_final_benchmark_report.json",
        help="Path for the consolidated JSON report.",
    )

    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Suppress benchmark output and print only the JSON report.",
    )

    args = parser.parse_args()

    report = build_report()
    output_path = ROOT / args.output

    save_report(report, output_path)

    if args.json_only:
        print(
            json.dumps(
                asdict(report),
                indent=2,
            )
        )
    else:
        print_report(report)
        print()
        print(f"Saved consolidated report: {output_path}")

    return 0 if report.overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())