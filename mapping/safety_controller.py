"""
A38 — Robustness and Safety Controller

Provides a standalone robustness/safety layer for FoveaMap.

A38 does not modify the frozen mapping, foveation, temporal, baseline,
or evaluation implementations. It evaluates runtime health signals and
selects one of three deterministic operating modes:

    NORMAL
    DEGRADED
    SAFETY

The controller is intentionally dependency-light and uses only NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


SAFETY_MODES = (
    "NORMAL",
    "DEGRADED",
    "SAFETY",
)


@dataclass(frozen=True)
class SafetyConfig:
    """
    Runtime thresholds used by the A38 safety controller.
    """

    min_points_degraded: int = 2_000
    min_points_safety: int = 500

    max_invalid_fraction_degraded: float = 0.05
    max_invalid_fraction_safety: float = 0.25

    max_latency_ms_degraded: float = 100.0
    max_latency_ms_safety: float = 250.0

    max_frame_gap_s_degraded: float = 0.20
    max_frame_gap_s_safety: float = 0.50

    degraded_max_range_m: float = 50.0
    safety_max_range_m: float = 25.0

    degraded_min_resolution_m: float = 0.10
    safety_min_resolution_m: float = 0.20

    def __post_init__(self) -> None:
        if (
            not isinstance(self.min_points_degraded, int)
            or self.min_points_degraded < 1
        ):
            raise ValueError(
                "min_points_degraded must be an integer >= 1"
            )

        if (
            not isinstance(self.min_points_safety, int)
            or self.min_points_safety < 0
        ):
            raise ValueError(
                "min_points_safety must be an integer >= 0"
            )

        if self.min_points_safety > self.min_points_degraded:
            raise ValueError(
                "min_points_safety must be <= min_points_degraded"
            )

        fractions = (
            self.max_invalid_fraction_degraded,
            self.max_invalid_fraction_safety,
        )

        for value in fractions:
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    "invalid fractions must be finite and in [0, 1]"
                )

        if (
            self.max_invalid_fraction_safety
            < self.max_invalid_fraction_degraded
        ):
            raise ValueError(
                "max_invalid_fraction_safety must be >= "
                "max_invalid_fraction_degraded"
            )

        latencies = (
            self.max_latency_ms_degraded,
            self.max_latency_ms_safety,
        )

        for value in latencies:
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    "latency thresholds must be finite and >= 0"
                )

        if self.max_latency_ms_safety < self.max_latency_ms_degraded:
            raise ValueError(
                "max_latency_ms_safety must be >= "
                "max_latency_ms_degraded"
            )

        gaps = (
            self.max_frame_gap_s_degraded,
            self.max_frame_gap_s_safety,
        )

        for value in gaps:
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    "frame-gap thresholds must be finite and >= 0"
                )

        if self.max_frame_gap_s_safety < self.max_frame_gap_s_degraded:
            raise ValueError(
                "max_frame_gap_s_safety must be >= "
                "max_frame_gap_s_degraded"
            )

        ranges = (
            self.degraded_max_range_m,
            self.safety_max_range_m,
        )

        for value in ranges:
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    "safety ranges must be finite and > 0"
                )

        if self.safety_max_range_m > self.degraded_max_range_m:
            raise ValueError(
                "safety_max_range_m must be <= degraded_max_range_m"
            )

        resolutions = (
            self.degraded_min_resolution_m,
            self.safety_min_resolution_m,
        )

        for value in resolutions:
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    "minimum resolutions must be finite and > 0"
                )


@dataclass(frozen=True)
class SafetyAssessment:
    """
    Result of one runtime health assessment.
    """

    mode: str
    reasons: tuple[str, ...]
    point_count: int
    invalid_fraction: float
    latency_ms: float
    frame_gap_s: float | None
    semantic_available: bool
    dynamic_available: bool

    @property
    def is_safe(self) -> bool:
        return self.mode != "SAFETY"

    @property
    def is_degraded(self) -> bool:
        return self.mode == "DEGRADED"

    @property
    def is_safety_mode(self) -> bool:
        return self.mode == "SAFETY"


@dataclass(frozen=True)
class SafetyPolicy:
    """
    Operational constraints selected for the current safety mode.
    """

    mode: str
    max_range_m: float | None
    minimum_resolution_m: float | None
    allow_full_foveation: bool
    reasons: tuple[str, ...]

    @property
    def near_field_priority(self) -> bool:
        return self.mode in {"DEGRADED", "SAFETY"}


def _validate_mode(mode: str) -> str:
    if mode not in SAFETY_MODES:
        raise ValueError(
            f"mode must be one of {SAFETY_MODES}, got {mode!r}"
        )

    return mode


def _validate_xyz(xyz: np.ndarray) -> np.ndarray:
    array = np.asarray(xyz, dtype=np.float64)

    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(
            f"xyz must have shape (N, 3), got {array.shape}"
        )

    return array


def _invalid_fraction(xyz: np.ndarray) -> float:
    if xyz.shape[0] == 0:
        return 1.0

    valid = np.all(np.isfinite(xyz), axis=1)

    return float(
        1.0 - np.mean(valid)
    )


def _validate_latency(latency_ms: float) -> float:
    value = float(latency_ms)

    if not np.isfinite(value) or value < 0.0:
        raise ValueError(
            "latency_ms must be finite and >= 0"
        )

    return value


def _validate_frame_gap(
    frame_gap_s: float | None,
) -> float | None:
    if frame_gap_s is None:
        return None

    value = float(frame_gap_s)

    if not np.isfinite(value) or value < 0.0:
        raise ValueError(
            "frame_gap_s must be finite and >= 0"
        )

    return value


def _determine_mode(
    *,
    point_count: int,
    invalid_fraction: float,
    latency_ms: float,
    frame_gap_s: float | None,
    config: SafetyConfig,
) -> tuple[str, tuple[str, ...]]:
    safety_reasons: list[str] = []
    degraded_reasons: list[str] = []

    if point_count <= config.min_points_safety:
        safety_reasons.append("LOW_POINT_COUNT")
    elif point_count < config.min_points_degraded:
        degraded_reasons.append("LOW_POINT_COUNT")

    if invalid_fraction > config.max_invalid_fraction_safety:
        safety_reasons.append("HIGH_INVALID_POINT_FRACTION")
    elif invalid_fraction > config.max_invalid_fraction_degraded:
        degraded_reasons.append("HIGH_INVALID_POINT_FRACTION")

    if latency_ms > config.max_latency_ms_safety:
        safety_reasons.append("HIGH_LATENCY")
    elif latency_ms > config.max_latency_ms_degraded:
        degraded_reasons.append("HIGH_LATENCY")

    if frame_gap_s is not None:
        if frame_gap_s > config.max_frame_gap_s_safety:
            safety_reasons.append("STALE_FRAME")
        elif frame_gap_s > config.max_frame_gap_s_degraded:
            degraded_reasons.append("STALE_FRAME")

    if safety_reasons:
        return (
            "SAFETY",
            tuple(safety_reasons),
        )

    if degraded_reasons:
        return (
            "DEGRADED",
            tuple(degraded_reasons),
        )

    return (
        "NORMAL",
        (),
    )


def assess_safety(
    xyz: np.ndarray,
    *,
    latency_ms: float = 0.0,
    frame_gap_s: float | None = None,
    semantic_available: bool = False,
    dynamic_available: bool = False,
    config: SafetyConfig | None = None,
) -> SafetyAssessment:
    """
    Assess runtime health and select NORMAL, DEGRADED, or SAFETY.
    """

    if config is None:
        config = SafetyConfig()

    points = _validate_xyz(xyz)
    latency = _validate_latency(latency_ms)
    gap = _validate_frame_gap(frame_gap_s)

    invalid_fraction = _invalid_fraction(points)

    point_count = int(points.shape[0])

    mode, reasons = _determine_mode(
        point_count=point_count,
        invalid_fraction=invalid_fraction,
        latency_ms=latency,
        frame_gap_s=gap,
        config=config,
    )

    if invalid_fraction >= 1.0 and "HIGH_INVALID_POINT_FRACTION" not in reasons:
        mode = "SAFETY"
        reasons = reasons + ("HIGH_INVALID_POINT_FRACTION",)

    return SafetyAssessment(
        mode=_validate_mode(mode),
        reasons=reasons,
        point_count=point_count,
        invalid_fraction=invalid_fraction,
        latency_ms=latency,
        frame_gap_s=gap,
        semantic_available=bool(semantic_available),
        dynamic_available=bool(dynamic_available),
    )


def safety_policy(
    assessment: SafetyAssessment,
    config: SafetyConfig | None = None,
) -> SafetyPolicy:
    """
    Convert a safety assessment into deterministic runtime constraints.
    """

    if not isinstance(assessment, SafetyAssessment):
        raise TypeError(
            "assessment must be a SafetyAssessment"
        )

    if config is None:
        config = SafetyConfig()

    mode = _validate_mode(assessment.mode)

    if mode == "NORMAL":
        return SafetyPolicy(
            mode=mode,
            max_range_m=None,
            minimum_resolution_m=None,
            allow_full_foveation=True,
            reasons=assessment.reasons,
        )

    if mode == "DEGRADED":
        return SafetyPolicy(
            mode=mode,
            max_range_m=config.degraded_max_range_m,
            minimum_resolution_m=config.degraded_min_resolution_m,
            allow_full_foveation=True,
            reasons=assessment.reasons,
        )

    return SafetyPolicy(
        mode=mode,
        max_range_m=config.safety_max_range_m,
        minimum_resolution_m=config.safety_min_resolution_m,
        allow_full_foveation=False,
        reasons=assessment.reasons,
    )


def apply_safety_policy(
    xyz: np.ndarray,
    assessment: SafetyAssessment,
    config: SafetyConfig | None = None,
) -> np.ndarray:
    """
    Apply the selected safety range constraint to a point cloud.

    NORMAL:
        returns all points.

    DEGRADED:
        retains points within the degraded maximum range.

    SAFETY:
        retains only points within the safety maximum range.

    The returned array is always a new array.
    """

    points = _validate_xyz(xyz)

    policy = safety_policy(
        assessment,
        config=config,
    )

    if policy.max_range_m is None:
        return points.copy()

    finite_mask = np.all(
        np.isfinite(points),
        axis=1,
    )

    distances = np.linalg.norm(
        points,
        axis=1,
    )

    keep = (
        finite_mask
        & (distances <= policy.max_range_m)
    )

    return points[keep].copy()


def validate_safety_reasons(
    reasons: Iterable[str],
) -> tuple[str, ...]:
    """
    Validate and normalize a sequence of safety reason codes.
    """

    normalized = tuple(str(reason) for reason in reasons)

    for reason in normalized:
        if not reason:
            raise ValueError(
                "safety reason codes must not be empty"
            )

    return normalized