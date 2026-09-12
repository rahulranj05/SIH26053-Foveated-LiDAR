from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


MILESTONES = {
    "A1": [
        "mapping/basic_grid.py",
        "tests",
    ],
    "A2": [
        "mapping/basic_grid.py",
    ],
    "A3": [
        "mapping/basic_grid.py",
    ],
    "A4": [
        "mapping/basic_grid.py",
    ],
    "A5": [
        "mapping/basic_grid.py",
    ],
    "A6": [
        "mapping/terrain_features.py",
    ],
    "A7": [
        "mapping/traversability.py",
    ],
    "A8": [
        "mapping/distance_adaptive.py",
    ],
    "A9": [
        "mapping/hierarchical_grid.py",
    ],
    "A10": [
        "tests",
    ],
    "A11": [
        "mapping/distance_adaptive.py",
    ],
    "A12": [
        "mapping/distance_adaptive.py",
    ],
    "A13": [
        "mapping/kinematic_foveation.py",
        "tests/test_kinematic_foveation.py",
    ],
    "A14": [
        "mapping/predicted_path_foveation.py",
        "tests/test_predicted_path_foveation.py",
    ],
    "A15": [
        "mapping/semantic_foveation.py",
        "tests/test_semantic_foveation.py",
    ],
    "A16": [
        "mapping/dynamic_foveation.py",
        "tests/test_dynamic_foveation.py",
    ],
    "A17": [
        "mapping/foveation_controller.py",
        "tests/test_foveation_controller.py",
    ],
    "A18": [
        "mapping/foveated_mapper.py",
    ],
    "A18.3": [
        "mapping/hierarchical_foveated_mapper.py",
        "tests/test_hierarchical_foveated_mapper.py",
        "tests/test_a18_3_structure.py",
    ],
    "A19": [
        "mapping/foveation_controller.py",
    ],
    "A20": [
        "mapping/foveamap_pipeline.py",
        "tests/test_foveamap_pipeline.py",
        "scripts/benchmark_a20_pipeline.py",
    ],
}


MILESTONE_DESCRIPTIONS = {
    "A1": "UniformGrid2D foundation",
    "A2": "Grid indexing / cell handling",
    "A3": "Basic grid validation",
    "A4": "Synthetic visualization",
    "A5": "SemanticKITTI real-frame validation",
    "A6": "Terrain features",
    "A7": "Traversability",
    "A8": "Distance-adaptive resolution",
    "A9": "Strict hierarchical grid",
    "A10": "Benchmark framework",
    "A11": "Vectorized aggregation",
    "A12": "Authoritative uniform/adaptive benchmark",
    "A13": "Kinematic foveation",
    "A14": "Predicted path foveation",
    "A15": "Semantic foveation",
    "A16": "Dynamic foveation",
    "A17": "Foveation controller",
    "A18": "Foveated mapper",
    "A18.3": "True hierarchical leaf mapper",
    "A19": "Foveation policy calibration",
    "A20": "Full pipeline integration",
}


def exists(relative_path: str) -> bool:
    """Return True if the expected project path exists."""
    return (ROOT / relative_path).exists()


def check_files() -> dict[str, bool]:
    """Check that every milestone's expected implementation/test files exist."""
    results: dict[str, bool] = {}

    print("=" * 70)
    print("A1–A20 FILE / MODULE CHECK")
    print("=" * 70)

    for milestone, required_files in MILESTONES.items():
        missing = [
            path
            for path in required_files
            if not exists(path)
        ]

        passed = len(missing) == 0
        results[milestone] = passed

        if passed:
            print(f"  {milestone:<5} PASS")
        else:
            print(f"  {milestone:<5} FAIL")

            for path in missing:
                print(f"         Missing: {path}")

    print()

    return results


def run_pytest() -> tuple[bool, str]:
    """Run the complete regression suite."""
    print("=" * 70)
    print("PYTEST REGRESSION CHECK")
    print("=" * 70)

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    output = (result.stdout + result.stderr).strip()

    print(output)
    print()

    return result.returncode == 0, output


def print_known_validation_status() -> None:
    """Print the current milestone definitions."""
    print("=" * 70)
    print("KNOWN VALIDATED MILESTONES")
    print("=" * 70)

    for milestone, description in MILESTONE_DESCRIPTIONS.items():
        print(f"  {milestone:<5} {description}")

    print()


def print_final_summary(
    file_results: dict[str, bool],
    pytest_passed: bool,
) -> None:
    """Print the final A1–A20 health-check result."""
    print("=" * 70)
    print("FOVEAMAP — A1–A20 HEALTH CHECK")
    print("=" * 70)

    print()

    for milestone in MILESTONES:
        result = file_results[milestone]

        print(
            f"  {milestone:<5}"
            f"{'PASS' if result else 'FAIL'}"
        )

    print()
    print("-" * 70)

    all_files_ok = all(file_results.values())

    print(
        f"  Pytest regression : "
        f"{'PASS' if pytest_passed else 'FAIL'}"
    )

    print(
        f"  Module structure  : "
        f"{'PASS' if all_files_ok else 'FAIL'}"
    )

    print("-" * 70)

    overall = all_files_ok and pytest_passed

    print()

    if overall:
        print("  A1–A20 HEALTH CHECK: PASS")
        print()
        print("  All expected milestone files are present.")
        print("  Full regression suite passes.")
    else:
        print("  A1–A20 HEALTH CHECK: FAIL")
        print()
        print("  Review the failures above.")

    print("=" * 70)


def main() -> int:
    print()
    print("=" * 70)
    print("FOVEAMAP — A1–A20 VALIDATION CHECK")
    print("=" * 70)
    print(f"Project root: {ROOT}")
    print()

    file_results = check_files()

    print_known_validation_status()

    pytest_passed, _ = run_pytest()

    print_final_summary(
        file_results=file_results,
        pytest_passed=pytest_passed,
    )

    return 0 if all(file_results.values()) and pytest_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())