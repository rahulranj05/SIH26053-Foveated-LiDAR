"""
A39 — FoveaMap Dashboard Data Layer

Provides a presentation-layer representation of the frozen FoveaMap
pipeline without modifying the underlying mapping, temporal, evaluation,
or safety implementations.

A39 is intentionally an adapter. It consumes already-computed FoveaMap
and safety results and exposes dashboard-ready statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


SAFETY_MODES = (
    "NORMAL",
    "DEGRADED",
    "SAFETY",
)


@dataclass(frozen=True)
class DashboardConfig:
    """Configuration for dashboard statistics."""

    max_points_for_preview: int = 50_000
    max_preview_range_m: float = 100.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_points_for_preview, int)
            or self.max_points_for_preview < 1
        ):
            raise ValueError(
                "max_points_for_preview must be an integer >= 1"
            )

        if (
            not np.isfinite(self.max_preview_range_m)
            or self.max_preview_range_m <= 0.0
        ):
            raise ValueError(
                "max_preview_range_m must be finite and > 0"
            )


@dataclass(frozen=True)
class DashboardFrame:
    """Dashboard-ready state for one processed frame."""

    frame_id: Any
    timestamp: float

    point_count: int
    valid_point_count: int
    invalid_point_count: int

    safety_mode: str
    safety_reasons: tuple[str, ...]

    semantic_available: bool
    dynamic_available: bool

    leaf_count: int
    resolution_counts: Mapping[float, int]

    dominant_reason_counts: Mapping[str, int]

    processing_latency_ms: float
    fps: float

    preview_xyz: np.ndarray

    @property
    def valid_fraction(self) -> float:
        if self.point_count == 0:
            return 0.0

        return float(
            self.valid_point_count / self.point_count
        )

    @property
    def invalid_fraction(self) -> float:
        if self.point_count == 0:
            return 1.0

        return float(
            self.invalid_point_count / self.point_count
        )

    @property
    def is_safe(self) -> bool:
        return self.safety_mode != "SAFETY"

    @property
    def is_degraded(self) -> bool:
        return self.safety_mode == "DEGRADED"

    @property
    def is_safety_mode(self) -> bool:
        return self.safety_mode == "SAFETY"

    @property
    def is_normal(self) -> bool:
        return self.safety_mode == "NORMAL"


def _validate_xyz(xyz: np.ndarray) -> np.ndarray:
    array = np.asarray(
        xyz,
        dtype=np.float64,
    )

    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(
            f"xyz must have shape (N, 3), got {array.shape}"
        )

    return array


def _validate_latency(
    latency_ms: float,
) -> float:
    value = float(latency_ms)

    if not np.isfinite(value) or value < 0.0:
        raise ValueError(
            "latency_ms must be finite and >= 0"
        )

    return value


def _validate_safety_mode(
    mode: str,
) -> str:
    if mode not in SAFETY_MODES:
        raise ValueError(
            f"invalid safety mode: {mode!r}"
        )

    return mode


def _extract_leaf_count(
    leaf_map: Any,
) -> int:
    if leaf_map is None:
        return 0

    levels = getattr(
        leaf_map,
        "levels",
        None,
    )

    if levels is not None:
        try:
            return int(
                sum(
                    len(level)
                    for level in levels.values()
                )
            )
        except AttributeError:
            pass

    if hasattr(leaf_map, "leaf_count"):
        return int(leaf_map.leaf_count)

    if hasattr(leaf_map, "__len__"):
        return int(len(leaf_map))

    raise TypeError(
        "unable to determine leaf count from leaf_map"
    )


def _extract_resolution_counts(
    leaf_map: Any,
) -> dict[float, int]:
    if leaf_map is None:
        return {}

    levels = getattr(
        leaf_map,
        "levels",
        None,
    )

    if levels is None:
        return {}

    result: dict[float, int] = {}

    for resolution, cells in levels.items():
        result[float(resolution)] = int(
            len(cells)
        )

    return dict(
        sorted(result.items())
    )


def _extract_dominant_reason_counts(
    foveation: Mapping[str, Any] | None,
) -> dict[str, int]:
    if not foveation:
        return {}

    dominant_reason = foveation.get(
        "dominant_reason"
    )

    if dominant_reason is None:
        return {}

    values = np.asarray(
        dominant_reason
    )

    if values.ndim == 0:
        values = values.reshape(1)

    counts: dict[str, int] = {}

    for value in values.reshape(-1):
        key = str(value)
        counts[key] = counts.get(key, 0) + 1

    return dict(
        sorted(counts.items())
    )


def _build_preview(
    xyz: np.ndarray,
    *,
    config: DashboardConfig,
) -> np.ndarray:
    finite_mask = np.all(
        np.isfinite(xyz),
        axis=1,
    )

    points = xyz[finite_mask]

    if points.shape[0] == 0:
        return np.empty(
            (0, 3),
            dtype=np.float64,
        )

    distances = np.linalg.norm(
        points,
        axis=1,
    )

    points = points[
        distances <= config.max_preview_range_m
    ]

    if points.shape[0] <= config.max_points_for_preview:
        return points.copy()

    indices = np.linspace(
        0,
        points.shape[0] - 1,
        config.max_points_for_preview,
        dtype=np.int64,
    )

    return points[indices].copy()


def build_dashboard_frame(
    *,
    frame_id: Any,
    timestamp: float,
    xyz: np.ndarray,
    foveation: Mapping[str, Any] | None,
    leaf_map: Any,
    safety_assessment: Any,
    latency_ms: float,
    config: DashboardConfig | None = None,
) -> DashboardFrame:
    """
    Build a dashboard-ready frame from existing FoveaMap outputs.

    This function does not run mapping or safety logic itself.
    It only summarizes already-computed results.
    """

    if config is None:
        config = DashboardConfig()

    points = _validate_xyz(xyz)

    timestamp_value = float(timestamp)

    if not np.isfinite(timestamp_value):
        raise ValueError(
            "timestamp must be finite"
        )

    latency = _validate_latency(
        latency_ms
    )

    if safety_assessment is None:
        raise ValueError(
            "safety_assessment must not be None"
        )

    safety_mode = _validate_safety_mode(
        str(safety_assessment.mode)
    )

    reasons = tuple(
        str(reason)
        for reason in safety_assessment.reasons
    )

    valid_mask = np.all(
        np.isfinite(points),
        axis=1,
    )

    point_count = int(
        points.shape[0]
    )

    valid_point_count = int(
        np.sum(valid_mask)
    )

    invalid_point_count = (
        point_count - valid_point_count
    )

    semantic_available = bool(
        getattr(
            safety_assessment,
            "semantic_available",
            False,
        )
    )

    dynamic_available = bool(
        getattr(
            safety_assessment,
            "dynamic_available",
            False,
        )
    )

    leaf_count = _extract_leaf_count(
        leaf_map
    )

    resolution_counts = (
        _extract_resolution_counts(
            leaf_map
        )
    )

    dominant_reason_counts = (
        _extract_dominant_reason_counts(
            foveation
        )
    )

    if latency > 0.0:
        fps = 1000.0 / latency
    else:
        fps = float("inf")

    preview = _build_preview(
        points,
        config=config,
    )

    preview.setflags(
        write=False
    )

    return DashboardFrame(
        frame_id=frame_id,
        timestamp=timestamp_value,
        point_count=point_count,
        valid_point_count=valid_point_count,
        invalid_point_count=invalid_point_count,
        safety_mode=safety_mode,
        safety_reasons=reasons,
        semantic_available=semantic_available,
        dynamic_available=dynamic_available,
        leaf_count=leaf_count,
        resolution_counts=resolution_counts,
        dominant_reason_counts=dominant_reason_counts,
        processing_latency_ms=latency,
        fps=fps,
        preview_xyz=preview,
    )