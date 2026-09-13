from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ValidationCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class ValidationReport:
    checks: tuple[ValidationCheck, ...]
    passed: int
    failed: int
    overall_pass: bool


REQUIRED_FILES = (
    "mapping/foveamap_pipeline.py",
    "mapping/temporal_foveamap_pipeline.py",
    "mapping/dataset_foveamap_runner.py",
    "mapping/safety_controller.py",
    "mapping/foveamap_dashboard.py",
    "mapping/foveamap_visual_dashboard.py",
    "mapping/temporal_motion.py",
    "evaluation/terrain_accuracy.py",
    "evaluation/dynamic_object_metrics.py",
    "evaluation/baseline_metrics.py",
    "evaluation/ablation.py",
    "scripts/run_foveamap_dashboard.py",
    "scripts/benchmark_a33_terrain_accuracy.py",
    "scripts/benchmark_a34_dynamic_objects.py",
    "scripts/benchmark_a35_baselines.py",
    "scripts/benchmark_a36_ablation.py",
    "scripts/benchmark_a37_performance.py",
    "scripts/benchmark_a38_safety.py",
)

BENCHMARK_SCRIPTS = (
    "scripts/benchmark_a33_terrain_accuracy.py",
    "scripts/benchmark_a34_dynamic_objects.py",
    "scripts/benchmark_a35_baselines.py",
    "scripts/benchmark_a36_ablation.py",
    "scripts/benchmark_a37_performance.py",
    "scripts/benchmark_a38_safety.py",
)

CORE_MODULES = (
    "mapping.foveamap_pipeline",
    "mapping.temporal_foveamap_pipeline",
    "mapping.dataset_foveamap_runner",
    "mapping.safety_controller",
    "mapping.foveamap_dashboard",
    "mapping.foveamap_visual_dashboard",
    "evaluation.terrain_accuracy",
    "evaluation.dynamic_object_metrics",
    "evaluation.baseline_metrics",
    "evaluation.ablation",
)


def _check_required_files() -> ValidationCheck:
    missing = [
        path
        for path in REQUIRED_FILES
        if not (ROOT / path).is_file()
    ]

    if missing:
        return ValidationCheck(
            name="required_files",
            passed=False,
            detail="Missing: " + ", ".join(missing),
        )

    return ValidationCheck(
        name="required_files",
        passed=True,
        detail=f"All {len(REQUIRED_FILES)} required files are present.",
    )


def _check_core_imports() -> ValidationCheck:
    failures: list[str] = []

    for module in CORE_MODULES:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                f"import {module}",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            stderr = result.stderr.strip().replace("\n", " ")
            failures.append(f"{module}: {stderr}")

    if failures:
        return ValidationCheck(
            name="core_imports",
            passed=False,
            detail="; ".join(failures),
        )

    return ValidationCheck(
        name="core_imports",
        passed=True,
        detail=f"All {len(CORE_MODULES)} core modules import successfully.",
    )


def _run_pytest(
    pytest_args: Sequence[str],
) -> tuple[bool, str]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        *pytest_args,
    ]

    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    output = (result.stdout + "\n" + result.stderr).strip()

    return result.returncode == 0, output


def _check_full_regression() -> ValidationCheck:
    passed, output = _run_pytest(("-q",))

    if not passed:
        tail = "\n".join(output.splitlines()[-12:])
        return ValidationCheck(
            name="full_regression",
            passed=False,
            detail=tail,
        )

    summary = "pytest completed successfully."

    for line in reversed(output.splitlines()):
        if "passed" in line and "in " in line:
            summary = line.strip()
            break

    return ValidationCheck(
        name="full_regression",
        passed=True,
        detail=summary,
    )


def _check_a39_live_dashboard_tests() -> ValidationCheck:
    test_path = ROOT / "tests" / "test_a39_3_live_dashboard.py"

    if not test_path.is_file():
        return ValidationCheck(
            name="a39_live_dashboard_tests",
            passed=False,
            detail="tests/test_a39_3_live_dashboard.py is missing.",
        )

    passed, output = _run_pytest(
        ("tests/test_a39_3_live_dashboard.py", "-q")
    )

    if not passed:
        tail = "\n".join(output.splitlines()[-12:])
        return ValidationCheck(
            name="a39_live_dashboard_tests",
            passed=False,
            detail=tail,
        )

    summary = "A39.3 targeted tests passed."

    for line in reversed(output.splitlines()):
        if "passed" in line and "in " in line:
            summary = line.strip()
            break

    return ValidationCheck(
        name="a39_live_dashboard_tests",
        passed=True,
        detail=summary,
    )


def _check_benchmark_inventory() -> ValidationCheck:
    missing = [
        path
        for path in BENCHMARK_SCRIPTS
        if not (ROOT / path).is_file()
    ]

    if missing:
        return ValidationCheck(
            name="benchmark_inventory",
            passed=False,
            detail="Missing benchmark scripts: " + ", ".join(missing),
        )

    return ValidationCheck(
        name="benchmark_inventory",
        passed=True,
        detail=(
            f"All {len(BENCHMARK_SCRIPTS)} final benchmark scripts "
            "are present."
        ),
    )


def build_report(
    *,
    run_full_regression: bool = True,
) -> ValidationReport:
    checks: list[ValidationCheck] = [
        _check_required_files(),
        _check_benchmark_inventory(),
        _check_core_imports(),
        _check_a39_live_dashboard_tests(),
    ]

    if run_full_regression:
        checks.append(_check_full_regression())

    passed = sum(check.passed for check in checks)
    failed = len(checks) - passed

    return ValidationReport(
        checks=tuple(checks),
        passed=passed,
        failed=failed,
        overall_pass=failed == 0,
    )


def _print_report(report: ValidationReport) -> None:
    print()
    print("=" * 72)
    print("FOVEAMAP — A40.1 FINAL VALIDATION")
    print("=" * 72)

    for check in report.checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.name}")
        print(f"       {check.detail}")

    print("-" * 72)
    print(
        f"Checks: {len(report.checks)} | "
        f"Passed: {report.passed} | "
        f"Failed: {report.failed}"
    )
    print(
        "OVERALL: "
        + ("PASS" if report.overall_pass else "FAIL")
    )
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run final A40.1 FoveaMap validation."
    )

    parser.add_argument(
        "--skip-regression",
        action="store_true",
        help="Skip the complete pytest regression.",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the validation report as JSON.",
    )

    args = parser.parse_args()

    report = build_report(
        run_full_regression=not args.skip_regression,
    )

    if args.json:
        payload = asdict(report)
        print(json.dumps(payload, indent=2))
    else:
        _print_report(report)

    return 0 if report.overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())