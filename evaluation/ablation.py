from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import numpy as np


ABLATION_VARIANTS = (
    "DISTANCE_ONLY",
    "DISTANCE_KINEMATIC",
    "DISTANCE_PATH",
    "DISTANCE_SEMANTIC",
    "DISTANCE_DYNAMIC",
    "FULL_FOVEAMAP",
)

ABLATION_SIGNALS = {
    "DISTANCE_ONLY": ("DISTANCE",),
    "DISTANCE_KINEMATIC": ("DISTANCE", "KINEMATIC"),
    "DISTANCE_PATH": ("DISTANCE", "PREDICTED_PATH"),
    "DISTANCE_SEMANTIC": ("DISTANCE", "SEMANTIC"),
    "DISTANCE_DYNAMIC": ("DISTANCE", "DYNAMIC"),
    "FULL_FOVEAMAP": (
        "DISTANCE",
        "KINEMATIC",
        "PREDICTED_PATH",
        "SEMANTIC",
        "DYNAMIC",
    ),
}


@dataclass(frozen=True)
class AblationVariantResult:
    """Measured result for one A36 ablation configuration."""

    name: str
    enabled_signals: tuple[str, ...]
    point_count: int
    active_cells: int
    memory_bytes: int
    latency_ms: float
    resolution_counts: dict[float, int]
    dominant_reason_counts: dict[str, int]

    @property
    def memory_kb(self) -> float:
        return self.memory_bytes / 1024.0

    @property
    def latency_fps(self) -> float:
        if self.latency_ms <= 0.0:
            return float("inf")
        return 1000.0 / self.latency_ms


@dataclass(frozen=True)
class AblationComparisonResult:
    """Complete A36 comparison across all ablation variants."""

    variants: tuple[AblationVariantResult, ...]

    def by_name(self, name: str) -> AblationVariantResult:
        for variant in self.variants:
            if variant.name == name:
                return variant

        raise KeyError(f"Unknown ablation variant: {name}")

    @property
    def full_foveamap(self) -> AblationVariantResult:
        return self.by_name("FULL_FOVEAMAP")

    def as_dict(self) -> dict[str, Any]:
        return {
            "variants": [
                {
                    "name": variant.name,
                    "enabled_signals": list(variant.enabled_signals),
                    "point_count": variant.point_count,
                    "active_cells": variant.active_cells,
                    "memory_bytes": variant.memory_bytes,
                    "memory_kb": variant.memory_kb,
                    "latency_ms": variant.latency_ms,
                    "latency_fps": variant.latency_fps,
                    "resolution_counts": dict(
                        variant.resolution_counts
                    ),
                    "dominant_reason_counts": dict(
                        variant.dominant_reason_counts
                    ),
                }
                for variant in self.variants
            ]
        }


def validate_ablation_signals(
    signals: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Validate and normalize the complete A36 signal dictionary."""

    if not isinstance(signals, dict):
        raise TypeError("signals must be a dictionary")

    normalized: dict[str, np.ndarray] = {}

    for name, values in signals.items():
        canonical_name = str(name).upper()
        array = np.asarray(values)

        if array.ndim != 1:
            raise ValueError(
                f"Signal {name!r} must have shape (N,)"
            )

        if not np.all(np.isfinite(array.astype(np.float64))):
            raise ValueError(
                f"Signal {name!r} contains non-finite values"
            )

        normalized[canonical_name] = array

    return normalized


def select_ablation_signals(
    signals: dict[str, np.ndarray],
    variant: str,
) -> dict[str, np.ndarray]:
    """
    Return only the signals enabled by one A36 experiment.

    The input dictionary is never modified.
    """

    if variant not in ABLATION_VARIANTS:
        raise ValueError(
            f"Unknown ablation variant: {variant!r}. "
            f"Expected one of {ABLATION_VARIANTS}"
        )

    normalized = validate_ablation_signals(signals)

    required = ABLATION_SIGNALS[variant]

    missing = [
        signal
        for signal in required
        if signal not in normalized
    ]

    if missing:
        raise ValueError(
            f"Missing required signals for {variant}: {missing}"
        )

    return {
        signal: normalized[signal].copy()
        for signal in required
    }


def _extract_leaf_count(leaf_map: Any) -> int:
    """Extract final-cell count without modifying A18.3."""

    if hasattr(leaf_map, "levels"):
        return int(len(getattr(leaf_map, "levels")))

    if hasattr(leaf_map, "active_leaf_count"):
        value = getattr(leaf_map, "active_leaf_count")
        return int(value() if callable(value) else value)

    if hasattr(leaf_map, "leaf_count"):
        value = getattr(leaf_map, "leaf_count")
        return int(value() if callable(value) else value)

    if hasattr(leaf_map, "leaves"):
        return int(len(getattr(leaf_map, "leaves")))

    if hasattr(leaf_map, "cells"):
        return int(len(getattr(leaf_map, "cells")))

    try:
        return int(len(leaf_map))
    except TypeError as exc:
        raise ValueError(
            "Unable to determine FoveaMap leaf count"
        ) from exc


def _extract_resolution_counts(
    foveation: dict[str, Any],
    leaf_map: Any,
) -> dict[float, int]:
    """Prefer authoritative leaf-map resolution counts."""

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

    resolutions = None

    for attribute_name in ("resolutions", "resolution"):
        if hasattr(leaf_map, attribute_name):
            resolutions = getattr(leaf_map, attribute_name)
            break

    if resolutions is not None:
        values = np.asarray(resolutions, dtype=np.float64).reshape(-1)

        counts: dict[float, int] = {}

        for value in values:
            key = float(value)
            counts[key] = counts.get(key, 0) + 1

        if counts:
            return counts

    resolution = foveation.get("resolution")

    if resolution is None:
        resolution = foveation.get("desired_resolution")

    if resolution is None:
        return {}

    values = np.asarray(resolution, dtype=np.float64).reshape(-1)

    counts = {}

    for value in values:
        key = float(value)
        counts[key] = counts.get(key, 0) + 1

    return counts


def _extract_dominant_reasons(
    foveation: dict[str, Any],
) -> dict[str, int]:
    """Count dominant-reason labels produced by A17."""

    value = foveation.get("dominant_reason")

    if value is None:
        value = foveation.get("dominant_reasons")

    if value is None:
        return {}

    values = np.asarray(value).reshape(-1)

    counts: dict[str, int] = {}

    for item in values:
        key = str(item)
        counts[key] = counts.get(key, 0) + 1

    return counts


def _run_variant(
    xyz: np.ndarray,
    signals: dict[str, np.ndarray],
    variant: str,
    pipeline: Callable[[np.ndarray, dict[str, np.ndarray]], Any],
) -> AblationVariantResult:
    selected = select_ablation_signals(
        signals,
        variant,
    )

    start = perf_counter()

    result = pipeline(
        xyz,
        selected,
    )

    elapsed_ms = (perf_counter() - start) * 1000.0

    leaf_map = result.leaf_map
    foveation = result.foveation

    active_cells = _extract_leaf_count(leaf_map)

    resolution_counts = _extract_resolution_counts(
        foveation,
        leaf_map,
    )

    dominant_reason_counts = _extract_dominant_reasons(
        foveation,
    )

    memory_bytes = active_cells * 64

    return AblationVariantResult(
        name=variant,
        enabled_signals=ABLATION_SIGNALS[variant],
        point_count=len(xyz),
        active_cells=active_cells,
        memory_bytes=memory_bytes,
        latency_ms=elapsed_ms,
        resolution_counts=resolution_counts,
        dominant_reason_counts=dominant_reason_counts,
    )


def run_ablation_study(
    xyz: np.ndarray,
    signals: dict[str, np.ndarray],
    *,
    pipeline: Callable[[np.ndarray, dict[str, np.ndarray]], Any]
    | None = None,
) -> AblationComparisonResult:
    """
    Run the complete A36 ablation matrix.

    The authoritative A20 pipeline is used unless a test or experiment
    supplies a custom pipeline.
    """

    xyz = np.asarray(xyz, dtype=np.float64)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3)")

    if not np.all(np.isfinite(xyz)):
        raise ValueError("xyz contains non-finite values")

    normalized = validate_ablation_signals(signals)

    point_count = len(xyz)

    for name, values in normalized.items():
        if len(values) != point_count:
            raise ValueError(
                f"Signal {name!r} has length {len(values)}, "
                f"expected {point_count}"
            )

    if pipeline is None:
        from mapping.foveamap_pipeline import run_foveamap_pipeline

        pipeline = run_foveamap_pipeline

    results = tuple(
        _run_variant(
            xyz,
            normalized,
            variant,
            pipeline,
        )
        for variant in ABLATION_VARIANTS
    )

    return AblationComparisonResult(
        variants=results,
    )