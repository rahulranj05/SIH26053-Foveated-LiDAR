from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BaselineMetrics:
    """Common measurable output for one mapping baseline."""

    name: str
    point_count: int
    active_cells: int
    resolution_counts: dict[float, int]
    memory_bytes: int
    latency_ms: float

    @property
    def points_per_active_cell(self) -> float:
        if self.active_cells == 0:
            return 0.0
        return self.point_count / self.active_cells

    @property
    def points_per_cell(self) -> float:
        return self.points_per_active_cell

    @property
    def memory_mb(self) -> float:
        return self.memory_bytes / (1024.0 * 1024.0)

    @property
    def update_rate_hz(self) -> float:
        if self.latency_ms <= 0.0:
            return 0.0
        return 1000.0 / self.latency_ms

    @property
    def finest_resolution_m(self) -> float:
        if not self.resolution_counts:
            return 0.0
        return min(self.resolution_counts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "point_count": self.point_count,
            "active_cells": self.active_cells,
            "resolution_counts": dict(self.resolution_counts),
            "memory_bytes": self.memory_bytes,
            "memory_mb": self.memory_mb,
            "latency_ms": self.latency_ms,
            "update_rate_hz": self.update_rate_hz,
            "points_per_active_cell": self.points_per_active_cell,
            "finest_resolution_m": self.finest_resolution_m,
        }


@dataclass(frozen=True)
class BaselineComparisonResult:
    """Comparison of all A35 mapping strategies."""

    baselines: tuple[BaselineMetrics, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.baselines)

    def by_name(self, name: str) -> BaselineMetrics:
        for baseline in self.baselines:
            if baseline.name == name:
                return baseline
        raise KeyError(f"Unknown baseline: {name!r}")

    def memory_reduction_percent(
        self,
        reference_name: str,
        candidate_name: str,
    ) -> float:
        reference = self.by_name(reference_name)
        candidate = self.by_name(candidate_name)

        if reference.memory_bytes == 0:
            return 0.0

        return (
            (reference.memory_bytes - candidate.memory_bytes)
            / reference.memory_bytes
            * 100.0
        )

    def active_cell_reduction_percent(
        self,
        reference_name: str,
        candidate_name: str,
    ) -> float:
        reference = self.by_name(reference_name)
        candidate = self.by_name(candidate_name)

        if reference.active_cells == 0:
            return 0.0

        return (
            (reference.active_cells - candidate.active_cells)
            / reference.active_cells
            * 100.0
        )

    def latency_reduction_percent(
        self,
        reference_name: str,
        candidate_name: str,
    ) -> float:
        reference = self.by_name(reference_name)
        candidate = self.by_name(candidate_name)

        if reference.latency_ms == 0.0:
            return 0.0

        return (
            (reference.latency_ms - candidate.latency_ms)
            / reference.latency_ms
            * 100.0
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "baselines": [baseline.as_dict() for baseline in self.baselines],
        }


def validate_xyz(xyz: np.ndarray) -> np.ndarray:
    """Validate and return an XYZ point cloud."""

    array = np.asarray(xyz, dtype=np.float64)

    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3)")

    if not np.all(np.isfinite(array)):
        raise ValueError("xyz must contain only finite values")

    return array


def validate_positive_resolution(resolution_m: float) -> float:
    resolution = float(resolution_m)

    if not np.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("resolution_m must be finite and positive")

    return resolution


def validate_distance_bins(
    distance_bins: tuple[float, ...],
) -> tuple[float, ...]:
    bins = tuple(float(value) for value in distance_bins)

    if len(bins) < 2:
        raise ValueError("distance_bins must contain at least two boundaries")

    if not np.all(np.isfinite(bins)):
        raise ValueError("distance_bins must contain finite values")

    if any(right <= left for left, right in zip(bins, bins[1:])):
        raise ValueError("distance_bins must be strictly increasing")

    if bins[0] < 0.0:
        raise ValueError("distance_bins cannot start below zero")

    return bins