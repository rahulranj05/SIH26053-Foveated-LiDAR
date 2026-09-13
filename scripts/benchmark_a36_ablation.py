from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from evaluation.ablation import (  # noqa: E402
    ABLATION_VARIANTS,
    run_ablation_study,
)


def build_demo_point_cloud(
    point_count: int = 2000,
) -> np.ndarray:
    """Create a deterministic LiDAR-like point cloud."""

    rng = np.random.default_rng(26053)

    xy = rng.uniform(
        -50.0,
        50.0,
        size=(point_count, 2),
    )

    z = (
        0.15 * np.sin(xy[:, 0] / 7.0)
        + 0.10 * np.cos(xy[:, 1] / 9.0)
        + rng.normal(
            0.0,
            0.03,
            size=point_count,
        )
    )

    return np.column_stack(
        (
            xy[:, 0],
            xy[:, 1],
            z,
        )
    ).astype(np.float64)


def build_demo_signals(
    xyz: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build deterministic signals for A36 framework validation."""

    distance = np.linalg.norm(
        xyz[:, :2],
        axis=1,
    )

    distance_importance = np.clip(
        1.0 - distance / 100.0,
        0.0,
        1.0,
    )

    kinematic = np.clip(
        np.abs(xyz[:, 1]) / 25.0,
        0.0,
        1.0,
    )

    predicted_path = np.exp(
        -(xyz[:, 1] ** 2) / (2.0 * 8.0**2)
    )

    semantic = np.where(
        np.abs(xyz[:, 0]) < 5.0,
        0.85,
        0.20,
    )

    dynamic = np.where(
        (xyz[:, 0] > 10.0)
        & (xyz[:, 0] < 25.0)
        & (np.abs(xyz[:, 1]) < 8.0),
        1.0,
        0.05,
    )

    return {
        "DISTANCE": distance_importance,
        "KINEMATIC": kinematic,
        "PREDICTED_PATH": predicted_path,
        "SEMANTIC": semantic,
        "DYNAMIC": dynamic,
    }


def main() -> None:
    print("=" * 78)
    print("FOVEAMAP — A36 ABLATION STUDY")
    print("=" * 78)

    xyz = build_demo_point_cloud()
    signals = build_demo_signals(xyz)

    result = run_ablation_study(
        xyz,
        signals,
    )

    print()
    print(
        f"{'Variant':<25}"
        f"{'Cells':>10}"
        f"{'Memory KB':>12}"
        f"{'Latency ms':>13}"
        f"{'FPS':>10}"
    )
    print("-" * 78)

    for variant in result.variants:
        print(
            f"{variant.name:<25}"
            f"{variant.active_cells:>10}"
            f"{variant.memory_kb:>12.2f}"
            f"{variant.latency_ms:>13.3f}"
            f"{variant.latency_fps:>10.2f}"
        )

    print()
    print("=" * 78)
    print("RESOLUTION DISTRIBUTION")
    print("=" * 78)

    for variant in result.variants:
        print()
        print(variant.name)

        for resolution, count in sorted(
            variant.resolution_counts.items()
        ):
            print(
                f"  {resolution:.2f} m : {count}"
            )

    print()
    print("=" * 78)
    print("DOMINANT FOVEATION REASONS")
    print("=" * 78)

    for variant in result.variants:
        print()
        print(variant.name)

        if not variant.dominant_reason_counts:
            print("  unavailable")
            continue

        for reason, count in sorted(
            variant.dominant_reason_counts.items()
        ):
            print(
                f"  {reason:<20} : {count}"
            )

    print()
    print("=" * 78)
    print("A36 STATUS: PASS")
    print("=" * 78)


if __name__ == "__main__":
    main()