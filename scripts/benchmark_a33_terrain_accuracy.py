from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

# Allow this script to be executed directly from the repository root:
#
#     python3 scripts/benchmark_a33_terrain_accuracy.py
#
# Without this, Python places `scripts/` rather than the repository root
# on sys.path, so `evaluation` cannot be imported.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from evaluation.terrain_accuracy import evaluate_terrain_accuracy


@dataclass(frozen=True)
class TerrainCase:
    """Synthetic terrain used for A33 evaluation."""

    name: str
    reference: np.ndarray
    predicted: np.ndarray
    resolution_m: float


def make_flat_terrain(size: int) -> np.ndarray:
    """Create a perfectly flat reference terrain."""

    if size < 2:
        raise ValueError("size must be at least 2")

    return np.zeros((size, size), dtype=np.float64)


def make_linear_slope(
    size: int,
    resolution_m: float,
) -> np.ndarray:
    """Create a deterministic planar terrain."""

    if size < 2:
        raise ValueError("size must be at least 2")

    if resolution_m <= 0.0 or not math.isfinite(resolution_m):
        raise ValueError("resolution_m must be finite and positive")

    y, x = np.indices(
        (size, size),
        dtype=np.float64,
    )

    x_m = x * resolution_m
    y_m = y * resolution_m

    return 0.20 * x_m + 0.08 * y_m


def make_stepped_terrain(size: int) -> np.ndarray:
    """Create a terrain with a deterministic elevation discontinuity."""

    if size < 2:
        raise ValueError("size must be at least 2")

    terrain = np.zeros(
        (size, size),
        dtype=np.float64,
    )

    midpoint = size // 2

    terrain[:, midpoint:] = 0.50

    return terrain


def make_uneven_terrain(
    size: int,
    resolution_m: float,
) -> np.ndarray:
    """Create a deterministic uneven terrain containing hills and waves."""

    if size < 2:
        raise ValueError("size must be at least 2")

    if resolution_m <= 0.0 or not math.isfinite(resolution_m):
        raise ValueError("resolution_m must be finite and positive")

    y, x = np.indices(
        (size, size),
        dtype=np.float64,
    )

    x_m = x * resolution_m
    y_m = y * resolution_m

    hill_1 = 0.45 * np.exp(
        -(
            ((x_m - 4.0) ** 2) / 4.0
            + ((y_m - 3.0) ** 2) / 3.0
        )
    )

    hill_2 = -0.30 * np.exp(
        -(
            ((x_m - 8.0) ** 2) / 5.0
            + ((y_m - 6.0) ** 2) / 4.0
        )
    )

    waves = (
        0.06
        * np.sin(x_m * 1.1)
        * np.cos(y_m * 0.8)
    )

    return hill_1 + hill_2 + waves


def perturb(
    terrain: np.ndarray,
    *,
    bias: float = 0.0,
    noise_amplitude: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a deterministic predicted terrain from a reference terrain.

    A controlled bias and Gaussian noise simulate reconstruction error.
    """

    terrain = np.asarray(
        terrain,
        dtype=np.float64,
    )

    if terrain.ndim != 2:
        raise ValueError("terrain must be a 2D array")

    if not math.isfinite(bias):
        raise ValueError("bias must be finite")

    if not math.isfinite(noise_amplitude):
        raise ValueError(
            "noise_amplitude must be finite"
        )

    if noise_amplitude < 0.0:
        raise ValueError(
            "noise_amplitude must be non-negative"
        )

    rng = np.random.default_rng(seed)

    noise = rng.normal(
        loc=0.0,
        scale=noise_amplitude,
        size=terrain.shape,
    )

    return terrain + bias + noise


def build_cases() -> list[TerrainCase]:
    """
    Build the deterministic A33 synthetic terrain benchmark suite.

    The benchmark intentionally contains:
      1. Flat terrain
      2. Linear slope
      3. Stepped terrain
      4. Uneven terrain

    The predicted terrain contains controlled reconstruction error so
    the evaluator is exercised with realistic non-zero errors.
    """

    size = 40
    resolution_m = 0.25

    flat = make_flat_terrain(size)

    slope = make_linear_slope(
        size,
        resolution_m,
    )

    stepped = make_stepped_terrain(size)

    uneven = make_uneven_terrain(
        size,
        resolution_m,
    )

    return [
        TerrainCase(
            name="Flat",
            reference=flat,
            predicted=perturb(
                flat,
                bias=0.015,
                noise_amplitude=0.003,
                seed=1,
            ),
            resolution_m=resolution_m,
        ),
        TerrainCase(
            name="Linear slope",
            reference=slope,
            predicted=perturb(
                slope,
                bias=-0.010,
                noise_amplitude=0.004,
                seed=2,
            ),
            resolution_m=resolution_m,
        ),
        TerrainCase(
            name="Stepped terrain",
            reference=stepped,
            predicted=perturb(
                stepped,
                bias=0.008,
                noise_amplitude=0.006,
                seed=3,
            ),
            resolution_m=resolution_m,
        ),
        TerrainCase(
            name="Uneven terrain",
            reference=uneven,
            predicted=perturb(
                uneven,
                bias=-0.005,
                noise_amplitude=0.010,
                seed=4,
            ),
            resolution_m=resolution_m,
        ),
    ]


def run_benchmark() -> dict[str, object]:
    """
    Run the complete A33 terrain accuracy benchmark.

    Returns:
        Dictionary mapping terrain names to TerrainAccuracyResult objects.
    """

    cases = build_cases()

    results: dict[str, object] = {}

    for case in cases:
        result = evaluate_terrain_accuracy(
            case.reference,
            case.predicted,
            resolution_m=case.resolution_m,
        )

        results[case.name] = result

    return results


def validate_results(
    results: dict[str, object],
) -> None:
    """
    Validate that the benchmark produced usable quantitative results.

    This is intentionally strict: a benchmark with NaN, missing coverage,
    or zero valid samples must not be reported as a successful evaluation.
    """

    if not results:
        raise RuntimeError(
            "A33 benchmark produced no results."
        )

    expected_cases = {
        "Flat",
        "Linear slope",
        "Stepped terrain",
        "Uneven terrain",
    }

    if set(results) != expected_cases:
        raise RuntimeError(
            "A33 benchmark produced unexpected terrain cases: "
            f"{sorted(results)}"
        )

    for name, result in results.items():
        elevation = result.elevation.metrics
        slope = result.slope
        roughness = result.roughness

        metrics = (
            elevation.mae,
            elevation.rmse,
            elevation.bias,
            elevation.max_abs_error,
            elevation.p95_abs_error,
            slope.mae,
            slope.rmse,
            slope.bias,
            slope.max_abs_error,
            slope.p95_abs_error,
            roughness.mae,
            roughness.rmse,
            roughness.bias,
            roughness.max_abs_error,
            roughness.p95_abs_error,
        )

        if not all(
            math.isfinite(float(value))
            for value in metrics
        ):
            raise RuntimeError(
                f"{name}: non-finite metric detected."
            )

        if not math.isfinite(
            float(result.elevation.coverage)
        ):
            raise RuntimeError(
                f"{name}: non-finite coverage."
            )

        if result.elevation.coverage < 1.0:
            raise RuntimeError(
                f"{name}: unexpected reference coverage "
                f"{result.elevation.coverage:.6f}"
            )

        if elevation.valid_count <= 0:
            raise RuntimeError(
                f"{name}: no valid elevation samples."
            )

        if elevation.rmse <= 0.0:
            raise RuntimeError(
                f"{name}: expected non-zero elevation "
                "error for perturbed prediction."
            )


def print_report(
    results: dict[str, object],
) -> None:
    """Print the human-readable A33 benchmark report."""

    print()
    print("=" * 78)
    print(
        "FOVEAMAP — A33 TERRAIN ACCURACY EVALUATION"
    )
    print("=" * 78)
    print()

    print(
        f"{'Terrain':<20}"
        f"{'Elevation RMSE':>18}"
        f"{'Slope RMSE':>16}"
        f"{'Roughness RMSE':>20}"
    )

    print("-" * 78)

    for name, result in results.items():
        print(
            f"{name:<20}"
            f"{result.elevation.metrics.rmse:>18.6f}"
            f"{result.slope.rmse:>16.6f}"
            f"{result.roughness.rmse:>20.6f}"
        )

    print()

    print("=" * 78)
    print("DETAILED ELEVATION METRICS")
    print("=" * 78)

    for name, result in results.items():
        metrics = result.elevation.metrics

        print()
        print(name)
        print(
            f"  MAE             : "
            f"{metrics.mae:.6f} m"
        )
        print(
            f"  RMSE            : "
            f"{metrics.rmse:.6f} m"
        )
        print(
            f"  Bias            : "
            f"{metrics.bias:.6f} m"
        )
        print(
            f"  Max abs error   : "
            f"{metrics.max_abs_error:.6f} m"
        )
        print(
            f"  P95 abs error   : "
            f"{metrics.p95_abs_error:.6f} m"
        )
        print(
            f"  Coverage        : "
            f"{result.elevation.coverage:.2%}"
        )
        print(
            f"  Valid cells     : "
            f"{metrics.valid_count}"
        )

    print()

    print("=" * 78)
    print("A33 STATUS: PASS")
    print("=" * 78)
    print()


def main() -> None:
    """Run, validate, and report the A33 benchmark."""

    results = run_benchmark()

    validate_results(results)

    print_report(results)


if __name__ == "__main__":
    main()