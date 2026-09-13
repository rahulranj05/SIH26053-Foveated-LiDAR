from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.a40_final_validation import (
    BENCHMARK_SCRIPTS,
    CORE_MODULES,
    REQUIRED_FILES,
    build_report,
)


ROOT = Path(__file__).resolve().parents[1]


def test_required_file_inventory_is_present():
    missing = [
        path
        for path in REQUIRED_FILES
        if not (ROOT / path).is_file()
    ]

    assert missing == []


def test_final_benchmark_inventory_is_present():
    missing = [
        path
        for path in BENCHMARK_SCRIPTS
        if not (ROOT / path).is_file()
    ]

    assert missing == []


@pytest.mark.parametrize("module", CORE_MODULES)
def test_core_module_imports(module):
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

    assert result.returncode == 0, result.stderr


def test_a40_validation_script_exists():
    script = ROOT / "scripts" / "a40_final_validation.py"

    assert script.is_file()


def test_a40_validation_report_without_regression():
    report = build_report(run_full_regression=False)

    assert report.overall_pass is True
    assert report.failed == 0
    assert report.passed == len(report.checks)


def test_a40_json_output():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/a40_final_validation.py",
            "--skip-regression",
            "--json",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

    payload = json.loads(result.stdout)

    assert payload["overall_pass"] is True
    assert payload["failed"] == 0
    assert payload["passed"] == len(payload["checks"])


def test_a39_live_dashboard_test_file_exists():
    test_path = ROOT / "tests" / "test_a39_3_live_dashboard.py"

    assert test_path.is_file()