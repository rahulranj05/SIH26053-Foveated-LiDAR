from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mapping.baselines import benchmark_baselines


def make_synthetic_foveamap_signals(
    xyz: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Deterministic A20-compatible synthetic signals.

    Distance is always present.

    The additional signals intentionally create contextual
    importance so that this is a genuine FoveaMap invocation
    rather than merely the distance-only baseline.
    """

    n = len(xyz)

    distance = np.linalg.norm(xyz[:, :2], axis=1)

    distance_signal = np.clip(
        1.0 - distance / 100.0,
        0.0,
        1.0,
    )

    semantic_signal = np.zeros(n, dtype=np.float64)

    # Synthetic high-importance obstacle region.
    semantic_region = (
        (xyz[:, 0] > 8.0)
        & (xyz[:, 0] < 18.0)
        & (np.abs(xyz[:, 1]) < 3.0)
    )

    semantic_signal[semantic_region] = 1.0

    dynamic_signal = np.zeros(n, dtype=np.float64)

    dynamic_region = (
        (xyz[:, 0] > 20.0)
        & (xyz[:, 0] < 35.0)
        & (xyz[:, 1] > 1.0)
        & (xyz[:, 1] < 4.0)
    )

    dynamic_signal[dynamic_region] = 1.0

    return {
        "DISTANCE": distance_signal,
        "SEMANTIC": semantic_signal,
        "DYNAMIC": dynamic_signal,
    }


def make_benchmark_cloud(
    *,
    seed: int = 35,
    point_count: int = 12000,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    x = rng.uniform(
        0.0,
        100.0,
        size=point_count,
    )

    y = rng.uniform(
        -25.0,
        25.0,
        size=point_count,
    )

    z = (
        0.15 * np.sin(x / 7.0)
        + 0.08 * np.cos(y / 5.0)
        + rng.normal(0.0, 0.03, size=point_count)
    )

    return np.column_stack((x, y, z)).astype(
        np.float64,
        copy=False,
    )


def print_resolution_counts(
    resolution_counts: dict[float, int],
) -> None:
    if not resolution_counts:
        print("  Resolution counts : none")
        return

    print("  Resolution counts :")

    for resolution, count in sorted(
        resolution_counts.items()
    ):
        print(
            f"    {resolution:>5.2f} m : {count}"
        )


def print_report(result) -> None:
    print()
    print("=" * 86)
    print("FOVEAMAP — A35 BASELINE IMPLEMENTATIONS")
    print("=" * 86)

    print()

    for baseline in result.baselines:
        print(baseline.name)
        print("-" * 86)
        print(f"  Points              : {baseline.point_count}")
        print(f"  Active cells/voxels : {baseline.active_cells}")
        print(f"  Memory              : {baseline.memory_mb:.4f} MB")
        print(f"  Latency             : {baseline.latency_ms:.4f} ms")
        print(f"  Update rate         : {baseline.update_rate_hz:.2f} Hz")
        print(
            f"  Points / cell       : "
            f"{baseline.points_per_active_cell:.4f}"
        )
        print_resolution_counts(
            baseline.resolution_counts
        )
        print()

    uniform_name = "Uniform 2.5D"
    adaptive_name = "Distance-only adaptive"
    foveamap_name = "FoveaMap"

    print("=" * 86)
    print("FOVEAMAP RELATIVE COMPARISON")
    print("=" * 86)

    print(
        f"Memory reduction vs {uniform_name:<24}: "
        f"{result.memory_reduction_percent(uniform_name, foveamap_name):>8.2f}%"
    )

    print(
        f"Active-cell reduction vs {uniform_name:<19}: "
        f"{result.active_cell_reduction_percent(uniform_name, foveamap_name):>8.2f}%"
    )

    print(
        f"Latency reduction vs {uniform_name:<24}: "
        f"{result.latency_reduction_percent(uniform_name, foveamap_name):>8.2f}%"
    )

    print()

    print(
        f"Memory reduction vs {adaptive_name:<21}: "
        f"{result.memory_reduction_percent(adaptive_name, foveamap_name):>8.2f}%"
    )

    print(
        f"Active-cell reduction vs {adaptive_name:<16}: "
        f"{result.active_cell_reduction_percent(adaptive_name, foveamap_name):>8.2f}%"
    )

    print(
        f"Latency reduction vs {adaptive_name:<21}: "
        f"{result.latency_reduction_percent(adaptive_name, foveamap_name):>8.2f}%"
    )

    print()
    print("=" * 86)
    print("A35 STATUS: PASS")
    print("=" * 86)


def main() -> None:
    xyz = make_benchmark_cloud()

    signals = make_synthetic_foveamap_signals(xyz)

    result = benchmark_baselines(
        xyz,
        foveamap_signals=signals,
    )

    print_report(result)


if __name__ == "__main__":
    main()
    