from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import numpy as np

from evaluation.baseline_metrics import (
    BaselineMetrics,
    BaselineComparisonResult,
    validate_distance_bins,
    validate_positive_resolution,
    validate_xyz,
)


DEFAULT_DISTANCE_BINS = (0.0, 10.0, 25.0, 50.0, 100.0)
DEFAULT_ADAPTIVE_RESOLUTIONS = (0.05, 0.10, 0.20, 0.40)


@dataclass(frozen=True)
class BaselineCell:
    """Compact 2.5D cell summary."""

    ix: int
    iy: int
    resolution_m: float
    z_min: float
    z_max: float
    z_mean: float
    z_variance: float
    point_count: int


@dataclass(frozen=True)
class BaselineMap:
    """Common representation returned by 2.5D baselines."""

    name: str
    cells: tuple[BaselineCell, ...]
    point_count: int
    resolution_counts: dict[float, int]

    @property
    def active_cells(self) -> int:
        return len(self.cells)

    @property
    def memory_bytes(self) -> int:
        # Explicit fixed-width estimate for a BaselineCell.
        # This is intentionally independent of Python object overhead.
        bytes_per_cell = (
            8  # ix
            + 8  # iy
            + 8  # resolution
            + 8  # z_min
            + 8  # z_max
            + 8  # z_mean
            + 8  # z_variance
            + 8  # point_count
        )
        return self.active_cells * bytes_per_cell


@dataclass(frozen=True)
class Voxel3D:
    """A fixed-resolution 3D occupancy voxel."""

    ix: int
    iy: int
    iz: int


@dataclass(frozen=True)
class Baseline3DMap:
    """Common representation for the uniform 3D-style baseline."""

    name: str
    voxels: tuple[Voxel3D, ...]
    point_count: int
    voxel_size_m: float

    @property
    def active_cells(self) -> int:
        return len(self.voxels)

    @property
    def memory_bytes(self) -> int:
        # int64 x 3 per occupied voxel.
        return self.active_cells * 24


def _aggregate_2d_cells(
    xyz: np.ndarray,
    resolutions: np.ndarray,
    *,
    name: str,
) -> BaselineMap:
    """Aggregate points into resolution-specific 2.5D cells."""

    xyz = validate_xyz(xyz)

    resolutions = np.asarray(resolutions, dtype=np.float64)

    if resolutions.shape != (len(xyz),):
        raise ValueError("resolutions must have shape (N,)")

    if not np.all(np.isfinite(resolutions)):
        raise ValueError("resolutions must be finite")

    if np.any(resolutions <= 0.0):
        raise ValueError("resolutions must be positive")

    cells: dict[tuple[float, int, int], list[float]] = {}

    for point, resolution in zip(xyz, resolutions):
        x, y, z = point
        resolution = float(resolution)

        ix = int(np.floor(x / resolution))
        iy = int(np.floor(y / resolution))

        key = (resolution, ix, iy)
        bucket = cells.setdefault(key, [])
        bucket.append(float(z))

    output: list[BaselineCell] = []

    for (resolution, ix, iy), values in sorted(
        cells.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            item[0][2],
        ),
    ):
        z_values = np.asarray(values, dtype=np.float64)

        output.append(
            BaselineCell(
                ix=ix,
                iy=iy,
                resolution_m=resolution,
                z_min=float(np.min(z_values)),
                z_max=float(np.max(z_values)),
                z_mean=float(np.mean(z_values)),
                z_variance=float(np.var(z_values)),
                point_count=int(len(z_values)),
            )
        )

    resolution_counts: dict[float, int] = {}

    for cell in output:
        resolution_counts[cell.resolution_m] = (
            resolution_counts.get(cell.resolution_m, 0) + 1
        )

    return BaselineMap(
        name=name,
        cells=tuple(output),
        point_count=len(xyz),
        resolution_counts=resolution_counts,
    )


def _timed_2d_baseline(
    xyz: np.ndarray,
    resolutions: np.ndarray,
    *,
    name: str,
) -> tuple[BaselineMap, float]:
    start = perf_counter()
    result = _aggregate_2d_cells(
        xyz,
        resolutions,
        name=name,
    )
    elapsed_ms = (perf_counter() - start) * 1000.0
    return result, elapsed_ms


def uniform_2d_25d_baseline(
    xyz: np.ndarray,
    *,
    resolution_m: float = 0.05,
) -> tuple[BaselineMap, float]:
    """
    Uniform-resolution 2.5D baseline.

    Every point uses exactly the same XY resolution.
    """

    xyz = validate_xyz(xyz)
    resolution = validate_positive_resolution(resolution_m)

    resolutions = np.full(
        len(xyz),
        resolution,
        dtype=np.float64,
    )

    return _timed_2d_baseline(
        xyz,
        resolutions,
        name="Uniform 2.5D",
    )


def uniform_3d_baseline(
    xyz: np.ndarray,
    *,
    voxel_size_m: float = 0.05,
) -> tuple[Baseline3DMap, float]:
    """
    Uniform-resolution 3D-style baseline.

    Every XYZ point is assigned to a fixed-size 3D voxel.
    """

    xyz = validate_xyz(xyz)
    voxel_size = validate_positive_resolution(voxel_size_m)

    start = perf_counter()

    indices = np.floor(xyz / voxel_size).astype(np.int64)

    if len(indices) == 0:
        unique = np.empty((0, 3), dtype=np.int64)
    else:
        unique = np.unique(indices, axis=0)

    voxels = tuple(
        Voxel3D(
            ix=int(index[0]),
            iy=int(index[1]),
            iz=int(index[2]),
        )
        for index in unique
    )

    result = Baseline3DMap(
        name="Uniform 3D",
        voxels=voxels,
        point_count=len(xyz),
        voxel_size_m=voxel_size,
    )

    elapsed_ms = (perf_counter() - start) * 1000.0

    return result, elapsed_ms


def _distance_resolution(
    xyz: np.ndarray,
    *,
    distance_bins: tuple[float, ...],
    resolutions: tuple[float, ...],
) -> np.ndarray:
    """
    Assign resolution using range only.

    This is deliberately context-free:
    no speed, yaw, path, semantic, or dynamic signals.
    """

    if len(resolutions) != len(distance_bins) - 1:
        raise ValueError(
            "resolutions must contain exactly len(distance_bins) - 1 values"
        )

    distance = np.linalg.norm(xyz[:, :2], axis=1)

    result = np.full(
        len(xyz),
        float(resolutions[-1]),
        dtype=np.float64,
    )

    for index, resolution in enumerate(resolutions):
        lower = distance_bins[index]
        upper = distance_bins[index + 1]

        mask = (distance >= lower) & (distance < upper)

        result[mask] = float(resolution)

    return result


def distance_only_adaptive_baseline(
    xyz: np.ndarray,
    *,
    distance_bins: tuple[float, ...] = DEFAULT_DISTANCE_BINS,
    resolutions: tuple[float, ...] = DEFAULT_ADAPTIVE_RESOLUTIONS,
) -> tuple[BaselineMap, float]:
    """
    Distance-only adaptive 2.5D baseline.

    This intentionally reproduces only the range-driven component.
    """

    xyz = validate_xyz(xyz)
    bins = validate_distance_bins(distance_bins)

    resolutions = tuple(
        validate_positive_resolution(value)
        for value in resolutions
    )

    if len(resolutions) != len(bins) - 1:
        raise ValueError(
            "resolutions must contain exactly len(distance_bins) - 1 values"
        )

    point_resolutions = _distance_resolution(
        xyz,
        distance_bins=bins,
        resolutions=resolutions,
    )

    return _timed_2d_baseline(
        xyz,
        point_resolutions,
        name="Distance-only adaptive",
    )


def _extract_resolution_counts_from_foveation(
    foveation: dict[str, Any],
) -> dict[float, int]:
    resolution = foveation.get("resolution")

    if resolution is None:
        resolution = foveation.get("desired_resolution")

    if resolution is None:
        raise ValueError(
            "FoveaMap foveation result does not contain a resolution array"
        )

    values = np.asarray(resolution, dtype=np.float64).reshape(-1)

    if not np.all(np.isfinite(values)):
        raise ValueError("FoveaMap resolution array contains non-finite values")

    counts: dict[float, int] = {}

    for value in values:
        key = float(value)
        counts[key] = counts.get(key, 0) + 1

    return counts


def _extract_leaf_count(leaf_map: Any) -> int:
    """
    Extract the number of final A18.3 hierarchical leaf cells.

    A18.3 stores one entry in ``levels`` for every final leaf cell.
    This is the authoritative representation-level count for the
    frozen hierarchical mapper.

    The other checks are retained for compatibility with alternate
    leaf-map representations used by tests or future adapters.
    """

    if hasattr(leaf_map, "active_leaf_count"):
        value = getattr(leaf_map, "active_leaf_count")
        return int(value() if callable(value) else value)

    if hasattr(leaf_map, "leaf_count"):
        value = getattr(leaf_map, "leaf_count")
        return int(value() if callable(value) else value)

    if hasattr(leaf_map, "leaves"):
        leaves = getattr(leaf_map, "leaves")
        return int(len(leaves))

    if hasattr(leaf_map, "cells"):
        cells = getattr(leaf_map, "cells")
        return int(len(cells))

    # A18.3 HierarchicalLeafMap stores one level value per final leaf.
    if hasattr(leaf_map, "levels"):
        levels = getattr(leaf_map, "levels")
        return int(len(levels))

    try:
        return int(len(leaf_map))
    except TypeError as exc:
        raise ValueError(
            "Unable to determine the number of FoveaMap leaf cells"
        ) from exc


def _extract_leaf_resolution_counts(
    leaf_map: Any,
) -> dict[float, int]:
    """
    Best-effort extraction of A18.3 leaf resolution counts.

    If the authoritative leaf map exposes a resolution-count API, use it.
    Otherwise inspect common cell containers.
    """

    for attribute_name in (
        "resolution_counts",
        "leaf_resolution_counts",
    ):
        if hasattr(leaf_map, attribute_name):
            value = getattr(leaf_map, attribute_name)
            value = value() if callable(value) else value

            return {
                float(key): int(count)
                for key, count in dict(value).items()
            }

    cells = None

    for attribute_name in ("leaves", "cells"):
        if hasattr(leaf_map, attribute_name):
            cells = getattr(leaf_map, attribute_name)
            break

    if cells is None:
        return {}

    counts: dict[float, int] = {}

    for cell in cells:
        resolution = None

        for attribute_name in (
            "resolution_m",
            "resolution",
            "cell_size",
        ):
            if hasattr(cell, attribute_name):
                resolution = getattr(cell, attribute_name)
                break

        if resolution is None:
            continue

        resolution = float(resolution)
        counts[resolution] = counts.get(resolution, 0) + 1

    return counts


def foveamap_baseline(
    xyz: np.ndarray,
    signals: dict[str, np.ndarray],
    *,
    pipeline: Callable[[np.ndarray, dict[str, np.ndarray]], Any] | None = None,
) -> tuple[BaselineMetrics, Any]:
    """
    Adapter for the authoritative A20 FoveaMap pipeline.

    The existing A20 implementation remains authoritative.
    This function only measures and normalizes its output for A35.
    """

    xyz = validate_xyz(xyz)

    if not isinstance(signals, dict):
        raise TypeError("signals must be a dictionary")

    if pipeline is None:
        from mapping.foveamap_pipeline import run_foveamap_pipeline

        pipeline = run_foveamap_pipeline

    start = perf_counter()

    result = pipeline(
        xyz,
        signals,
    )

    elapsed_ms = (perf_counter() - start) * 1000.0

    foveation = result.foveation
    leaf_map = result.leaf_map

    resolution_counts = _extract_resolution_counts_from_foveation(
        foveation
    )

    active_cells = _extract_leaf_count(leaf_map)

    # The leaf representation stores final map cells. For the common
    # comparison metric, use the fixed-width BaselineCell estimate.
    memory_bytes = active_cells * 64

    metrics = BaselineMetrics(
        name="FoveaMap",
        point_count=len(xyz),
        active_cells=active_cells,
        resolution_counts=(
            _extract_leaf_resolution_counts(leaf_map)
            or resolution_counts
        ),
        memory_bytes=memory_bytes,
        latency_ms=elapsed_ms,
    )

    return metrics, result


def benchmark_baselines(
    xyz: np.ndarray,
    *,
    foveamap_signals: dict[str, np.ndarray],
    uniform_resolution_m: float = 0.05,
    uniform_3d_resolution_m: float = 0.05,
    distance_bins: tuple[float, ...] = DEFAULT_DISTANCE_BINS,
    distance_resolutions: tuple[float, ...] = DEFAULT_ADAPTIVE_RESOLUTIONS,
) -> BaselineComparisonResult:
    """Run every A35 baseline on the same point cloud."""

    xyz = validate_xyz(xyz)

    uniform_map, uniform_latency = uniform_2d_25d_baseline(
        xyz,
        resolution_m=uniform_resolution_m,
    )

    uniform_3d_map, uniform_3d_latency = uniform_3d_baseline(
        xyz,
        voxel_size_m=uniform_3d_resolution_m,
    )

    distance_map, distance_latency = distance_only_adaptive_baseline(
        xyz,
        distance_bins=distance_bins,
        resolutions=distance_resolutions,
    )

    foveamap_metrics, _ = foveamap_baseline(
        xyz,
        foveamap_signals,
    )

    uniform_metrics = BaselineMetrics(
        name=uniform_map.name,
        point_count=uniform_map.point_count,
        active_cells=uniform_map.active_cells,
        resolution_counts=uniform_map.resolution_counts,
        memory_bytes=uniform_map.memory_bytes,
        latency_ms=uniform_latency,
    )

    uniform_3d_metrics = BaselineMetrics(
        name=uniform_3d_map.name,
        point_count=uniform_3d_map.point_count,
        active_cells=uniform_3d_map.active_cells,
        resolution_counts={
            uniform_3d_map.voxel_size_m: uniform_3d_map.active_cells,
        },
        memory_bytes=uniform_3d_map.memory_bytes,
        latency_ms=uniform_3d_latency,
    )

    distance_metrics = BaselineMetrics(
        name=distance_map.name,
        point_count=distance_map.point_count,
        active_cells=distance_map.active_cells,
        resolution_counts=distance_map.resolution_counts,
        memory_bytes=distance_map.memory_bytes,
        latency_ms=distance_latency,
    )

    return BaselineComparisonResult(
        baselines=(
            uniform_metrics,
            uniform_3d_metrics,
            distance_metrics,
            foveamap_metrics,
        )
    )