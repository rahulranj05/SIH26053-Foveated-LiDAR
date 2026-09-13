from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_final_demo.py"


def test_final_demo_script_exists() -> None:
    assert SCRIPT.is_file()


def test_final_demo_help() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--help",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--frames" in result.stdout
    assert "--fps" in result.stdout
    assert "--seed" in result.stdout
    assert "--no-show" in result.stdout
    assert "--save-dir" in result.stdout


def test_final_demo_rejects_invalid_frame_count() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--frames",
            "0",
            "--no-show",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "--frames must be an integer >= 1." in result.stderr


def test_final_demo_rejects_invalid_fps() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--frames",
            "1",
            "--fps",
            "0",
            "--no-show",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "--fps must be greater than 0." in result.stderr


def test_final_demo_short_smoke_run() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--frames",
            "2",
            "--fps",
            "10",
            "--seed",
            "39",
            "--no-show",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        "Final demo failed.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )