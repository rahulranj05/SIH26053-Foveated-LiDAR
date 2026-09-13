from __future__ import annotations

import argparse
import sys
from pathlib import Path

# When this file is executed directly with:
#     python3 scripts/visualize_a24.py
# Python puts "scripts/" on sys.path rather than the repository root.
# Add the repository root explicitly so the project packages are importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from mapping.foveamap_pipeline import run_foveamap_pipeline
from mapping.map_visualizer import (
    MapVisualizationConfig,
    plot_foveamap_result,
    save_foveamap_figure,
)
from mapping.semantic_kitti_adapter import load_semantickitti_scan


def build_demo_distance_signal(
    xyz: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Build an explicit distance-based demonstration signal.

    This is NOT semantic or dynamic information.

    It exists only so the A24 visualizer can be exercised using the
    repository's available raw SemanticKITTI .bin test frame, which
    does not contain semantic labels.
    """
    ranges = np.linalg.norm(xyz[:, :2], axis=1)

    importance = np.clip(
        1.0 - (ranges / 100.0),
        0.0,
        1.0,
    )

    return {
        "DISTANCE": importance,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A24 FoveaMap 2.5D map visualization demo."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "datasets/TEST/000000.bin",
        help="SemanticKITTI .bin scan to visualize.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/a24_foveamap_map.png",
        help="Output PNG path.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the visualization interactively.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input LiDAR scan not found: {args.input}"
        )

    print("=" * 70)
    print("FOVEAMAP — A24 2.5D MAP VISUALIZATION")
    print("=" * 70)

    print(f"Input : {args.input}")

    xyz = load_semantickitti_scan(args.input)

    print(f"Points: {xyz.shape[0]:,}")

    # The repository's current test frame contains raw XYZ/intensity
    # but no semantic labels or dynamic probabilities.
    #
    # Therefore this demo supplies ONLY an explicit distance signal.
    # It does not fabricate semantic or dynamic information.
    signals = build_demo_distance_signal(xyz)

    print("Signal: DISTANCE only")
    print("Semantics: unavailable")
    print("Dynamics: unavailable")

    result = run_foveamap_pipeline(
        xyz,
        signals,
    )

    print()
    print("Pipeline complete.")
    print(f"Leaf cells: {result.leaf_map.active_cells:,}")

    unique_resolutions, resolution_counts = np.unique(
        result.leaf_map.resolutions,
        return_counts=True,
    )

    print()
    print("Leaf resolution distribution:")

    for resolution, count in zip(
        unique_resolutions,
        resolution_counts,
    ):
        print(
            f"  {resolution:.2f} m : {count:,} cells"
        )

    config = MapVisualizationConfig()

    figure = plot_foveamap_result(
        xyz,
        result,
        config,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_foveamap_figure(
        figure,
        args.output,
    )

    print()
    print(f"Saved visualization: {args.output}")

    if args.show:
        plt.show()
    else:
        plt.close(figure)

    print("=" * 70)


if __name__ == "__main__":
    main()