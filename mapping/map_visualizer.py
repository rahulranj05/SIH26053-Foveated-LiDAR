from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from mapping.foveamap_pipeline import FoveaMapPipelineResult
from mapping.hierarchical_foveated_mapper import (
    HierarchicalLeafMap,
    REASON_NAMES,
)


@dataclass(frozen=True)
class MapVisualizationConfig:
    point_size: float = 0.5
    point_alpha: float = 0.20
    cell_alpha: float = 0.60
    show_points: bool = True
    show_cell_edges: bool = True
    show_reason_labels: bool = True
    show_resolution_legend: bool = True
    equal_aspect: bool = True
    max_points: int = 100_000

    def __post_init__(self) -> None:
        if self.point_size <= 0:
            raise ValueError("point_size must be > 0")

        if not 0.0 <= self.point_alpha <= 1.0:
            raise ValueError("point_alpha must be in [0, 1]")

        if not 0.0 <= self.cell_alpha <= 1.0:
            raise ValueError("cell_alpha must be in [0, 1]")

        if self.max_points <= 0:
            raise ValueError("max_points must be > 0")


def _validate_xyz(xyz: np.ndarray) -> np.ndarray:
    points = np.asarray(xyz)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3)")

    if not np.issubdtype(points.dtype, np.number):
        raise ValueError("xyz must contain numeric values")

    if not np.all(np.isfinite(points)):
        raise ValueError("xyz must contain only finite values")

    return points


def _validate_leaf_map(leaf_map: HierarchicalLeafMap) -> None:
    if not isinstance(leaf_map, HierarchicalLeafMap):
        raise TypeError(
            "leaf_map must be a HierarchicalLeafMap"
        )

    arrays = (
        leaf_map.levels,
        leaf_map.resolutions,
        leaf_map.ix,
        leaf_map.iy,
        leaf_map.z_min,
        leaf_map.z_max,
        leaf_map.z_mean,
        leaf_map.z_variance,
        leaf_map.point_count,
        leaf_map.dominant_reason,
    )

    sizes = {np.asarray(array).size for array in arrays}

    if len(sizes) != 1:
        raise ValueError(
            "all leaf-map arrays must have the same length"
        )

    if leaf_map.input_points < 0:
        raise ValueError("input_points must be >= 0")

    if int(np.sum(leaf_map.point_count)) != leaf_map.input_points:
        raise ValueError(
            "leaf-map point counts must conserve input_points"
        )


def _validate_result(
    result: FoveaMapPipelineResult,
) -> None:
    if not isinstance(result, FoveaMapPipelineResult):
        raise TypeError(
            "result must be a FoveaMapPipelineResult"
        )

    _validate_leaf_map(result.leaf_map)


def leaf_cell_rectangles(
    leaf_map: HierarchicalLeafMap,
) -> list[Rectangle]:
    """
    Convert hierarchical leaf cells into XY rectangles.

    Cell coordinates follow the A18.3 convention:

        ix = floor(x / resolution)
        iy = floor(y / resolution)

    Therefore the lower-left corner is:

        (ix * resolution, iy * resolution)
    """
    _validate_leaf_map(leaf_map)

    rectangles: list[Rectangle] = []

    for ix, iy, resolution in zip(
        leaf_map.ix,
        leaf_map.iy,
        leaf_map.resolutions,
    ):
        resolution = float(resolution)

        rectangles.append(
            Rectangle(
                (
                    float(ix) * resolution,
                    float(iy) * resolution,
                ),
                resolution,
                resolution,
            )
        )

    return rectangles


def plot_lidar_points(
    ax,
    xyz: np.ndarray,
    config: MapVisualizationConfig | None = None,
):
    """
    Plot the original LiDAR points in XY.

    Point colour represents elevation Z.
    """
    points = _validate_xyz(xyz)

    if config is None:
        config = MapVisualizationConfig()

    if points.shape[0] > config.max_points:
        indices = np.linspace(
            0,
            points.shape[0] - 1,
            config.max_points,
            dtype=np.int64,
        )
        points = points[indices]

    collection = ax.scatter(
        points[:, 0],
        points[:, 1],
        c=points[:, 2],
        s=config.point_size,
        alpha=config.point_alpha,
        linewidths=0,
        rasterized=True,
    )

    return collection


def plot_leaf_map(
    ax,
    leaf_map: HierarchicalLeafMap,
    config: MapVisualizationConfig | None = None,
):
    """
    Plot hierarchical leaf cells as a 2D top-down map.

    Cell colour represents mean elevation.
    """
    _validate_leaf_map(leaf_map)

    if config is None:
        config = MapVisualizationConfig()

    rectangles = leaf_cell_rectangles(leaf_map)

    collection = PatchCollection(
        rectangles,
        cmap="viridis",
        alpha=config.cell_alpha,
        edgecolor="black" if config.show_cell_edges else "none",
        linewidth=0.4 if config.show_cell_edges else 0.0,
    )

    if leaf_map.z_mean.size:
        collection.set_array(
            np.asarray(leaf_map.z_mean, dtype=np.float64)
        )

    ax.add_collection(collection)

    return collection


def _resolution_counts(
    leaf_map: HierarchicalLeafMap,
) -> dict[float, int]:
    counts: dict[float, int] = {}

    for resolution in np.asarray(
        leaf_map.resolutions,
        dtype=np.float64,
    ):
        key = float(resolution)
        counts[key] = counts.get(key, 0) + 1

    return dict(sorted(counts.items()))


def _reason_counts(
    leaf_map: HierarchicalLeafMap,
) -> dict[str, int]:
    reasons = np.asarray(
        leaf_map.dominant_reason,
        dtype=object,
    )

    counts: dict[str, int] = {}

    for reason in reasons:
        name = str(reason)
        counts[name] = counts.get(name, 0) + 1

    return dict(sorted(counts.items()))


def plot_foveamap_result(
    xyz: np.ndarray,
    result: FoveaMapPipelineResult,
    config: MapVisualizationConfig | None = None,
):
    """
    Create a complete top-down visualization of a FoveaMap result.

    The visualization contains:

    - original LiDAR XY points coloured by elevation
    - hierarchical leaf-cell boundaries
    - leaf-cell elevation colouring
    - textual resolution statistics
    - dominant foveation-reason statistics
    """
    points = _validate_xyz(xyz)
    _validate_result(result)

    if points.shape[0] != result.leaf_map.input_points:
        raise ValueError(
            "xyz point count must match leaf_map.input_points"
        )

    if config is None:
        config = MapVisualizationConfig()

    figure, ax = plt.subplots(
        figsize=(12, 10),
    )

    if config.show_points and points.shape[0] > 0:
        point_collection = plot_lidar_points(
            ax,
            points,
            config,
        )

        figure.colorbar(
            point_collection,
            ax=ax,
            label="LiDAR elevation Z (m)",
            fraction=0.046,
            pad=0.04,
        )

    cell_collection = plot_leaf_map(
        ax,
        result.leaf_map,
        config,
    )

    if result.leaf_map.z_mean.size:
        figure.colorbar(
            cell_collection,
            ax=ax,
            label="Leaf-cell mean elevation Z (m)",
            fraction=0.046,
            pad=0.08,
        )

    if points.shape[0] > 0:
        x_min = float(np.min(points[:, 0]))
        x_max = float(np.max(points[:, 0]))
        y_min = float(np.min(points[:, 1]))
        y_max = float(np.max(points[:, 1]))

        x_margin = max((x_max - x_min) * 0.03, 0.5)
        y_margin = max((y_max - y_min) * 0.03, 0.5)

        ax.set_xlim(
            x_min - x_margin,
            x_max + x_margin,
        )
        ax.set_ylim(
            y_min - y_margin,
            y_max + y_margin,
        )

    if config.equal_aspect:
        ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_title(
        "FoveaMap — Adaptive Hierarchical 2.5D LiDAR Map"
    )

    resolution_counts = _resolution_counts(
        result.leaf_map
    )

    reason_counts = _reason_counts(
        result.leaf_map
    )

    summary_lines = [
        f"Input points: {result.leaf_map.input_points:,}",
        f"Active leaf cells: {result.leaf_map.active_cells:,}",
    ]

    if resolution_counts:
        summary_lines.append("")
        summary_lines.append("Leaf resolution:")

        for resolution, count in resolution_counts.items():
            summary_lines.append(
                f"  {resolution:.2f} m: {count:,}"
            )

    if reason_counts:
        summary_lines.append("")
        summary_lines.append("Dominant reason:")

        for reason in REASON_NAMES:
            count = reason_counts.get(reason, 0)

            if count:
                summary_lines.append(
                    f"  {reason}: {count:,}"
                )

        # Preserve any unexpected reason labels rather than
        # silently hiding them.
        known = set(REASON_NAMES)

        for reason, count in reason_counts.items():
            if reason not in known:
                summary_lines.append(
                    f"  {reason}: {count:,}"
                )

    ax.text(
        0.01,
        0.99,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "white",
            "alpha": 0.85,
        },
    )

    ax.grid(
        True,
        alpha=0.20,
        linewidth=0.5,
    )

    figure.tight_layout()

    return figure


def save_foveamap_figure(
    figure,
    path: str | Path,
    dpi: int = 200,
) -> Path:
    """
    Save a FoveaMap visualization to disk.
    """
    if dpi <= 0:
        raise ValueError("dpi must be > 0")

    output_path = Path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    return output_path


__all__ = [
    "MapVisualizationConfig",
    "leaf_cell_rectangles",
    "plot_lidar_points",
    "plot_leaf_map",
    "plot_foveamap_result",
    "save_foveamap_figure",
]