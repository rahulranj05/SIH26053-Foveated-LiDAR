"""
A20 — FoveaMap Full Pipeline

Orchestration layer connecting:

    A17 Foveation Controller
        |
        v
    A18.3 Hierarchical Foveated Mapper
        |
        v
    Hierarchical Leaf Map

A20 intentionally contains no new foveation or mapping logic.
It only validates inputs, invokes the validated A17 controller,
passes its outputs into the validated A18.3 mapper, and records timing.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .foveation_controller import compute_foveation
from .hierarchical_foveated_mapper import (
    HierarchicalLeafMap,
    build_hierarchical_foveated_map,
)


# ============================================================================
# Result container
# ============================================================================


@dataclass(frozen=True)
class FoveaMapPipelineResult:
    """
    Complete A20 pipeline result.

    Attributes
    ----------
    foveation:
        Exact output dictionary produced by A17.

    leaf_map:
        Exact HierarchicalLeafMap produced by A18.3.

    timings_ms:
        Timing breakdown containing:
            controller_ms
            mapper_ms
            total_ms
    """

    foveation: dict[str, np.ndarray]
    leaf_map: HierarchicalLeafMap
    timings_ms: dict[str, float]


# ============================================================================
# Input validation
# ============================================================================


def _validate_xyz(xyz: np.ndarray) -> np.ndarray:
    """
    Validate and normalize the input point cloud.

    Expected shape:
        (N, 3)
    """

    array = np.asarray(xyz, dtype=np.float64)

    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(
            f"xyz must have shape (N, 3); got {array.shape}"
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            "xyz must contain only finite values"
        )

    return array


def _validate_signals(
    signals: dict[str, np.ndarray],
    num_points: int,
) -> dict[str, np.ndarray]:
    """
    Validate A17-compatible importance signals.

    A17 expects one-dimensional arrays with exactly one value
    per input point.
    """

    if not isinstance(signals, dict):
        raise TypeError(
            "signals must be a dictionary"
        )

    validated: dict[str, np.ndarray] = {}

    for name, values in signals.items():

        array = np.asarray(
            values,
            dtype=np.float64,
        )

        if array.ndim != 1:
            raise ValueError(
                f"signal '{name}' must be one-dimensional; "
                f"got shape {array.shape}"
            )

        if len(array) != num_points:
            raise ValueError(
                f"signal '{name}' has {len(array)} values; "
                f"expected {num_points}"
            )

        if not np.all(np.isfinite(array)):
            raise ValueError(
                f"signal '{name}' must contain only finite values"
            )

        validated[name] = array

    if not validated:
        raise ValueError(
            "signals must contain at least one importance signal"
        )

    return validated


# ============================================================================
# Pipeline
# ============================================================================


def run_foveamap_pipeline(
    xyz: np.ndarray,
    signals: dict[str, np.ndarray],
) -> FoveaMapPipelineResult:
    """
    Execute the complete A20 FoveaMap pipeline.

    Flow
    ----
    1. Validate the point cloud.
    2. Validate A17 importance signals.
    3. Run A17 Foveation Controller.
    4. Pass A17 resolution and dominant-reason outputs directly
       into A18.3.
    5. Return the foveation result, hierarchical leaf map, and timings.

    Notes
    -----
    A17 and A18.3 are treated as validated black-box modules.
    This function does not alter their behavior.
    """

    points = _validate_xyz(xyz)

    validated_signals = _validate_signals(
        signals,
        len(points),
    )

    total_start = time.perf_counter()

    # ------------------------------------------------------------------------
    # A17 — Foveation Controller
    # ------------------------------------------------------------------------

    controller_start = time.perf_counter()

    foveation = compute_foveation(
        validated_signals
    )

    controller_end = time.perf_counter()

    # ------------------------------------------------------------------------
    # Validate A17 output before passing it downstream.
    # ------------------------------------------------------------------------

    if not isinstance(foveation, dict):
        raise TypeError(
            "A17 compute_foveation() must return a dictionary"
        )

    required_outputs = (
        "importance",
        "resolution",
        "dominant_reason",
    )

    for key in required_outputs:
        if key not in foveation:
            raise ValueError(
                f"A17 output is missing required field '{key}'"
            )

        value = np.asarray(
            foveation[key]
        )

        if len(value) != len(points):
            raise ValueError(
                f"A17 output '{key}' has {len(value)} values; "
                f"expected {len(points)}"
            )

    # ------------------------------------------------------------------------
    # A18.3 — Hierarchical Foveated Mapper
    # ------------------------------------------------------------------------

    mapper_start = time.perf_counter()

    leaf_map = build_hierarchical_foveated_map(
        points,
        foveation["resolution"],
        foveation["dominant_reason"],
    )

    mapper_end = time.perf_counter()

    total_end = time.perf_counter()

    timings_ms = {
        "controller_ms": (
            controller_end
            - controller_start
        ) * 1000.0,

        "mapper_ms": (
            mapper_end
            - mapper_start
        ) * 1000.0,

        "total_ms": (
            total_end
            - total_start
        ) * 1000.0,
    }

    return FoveaMapPipelineResult(
        foveation=foveation,
        leaf_map=leaf_map,
        timings_ms=timings_ms,
    )


# ============================================================================
# Compatibility alias
# ============================================================================


def run_pipeline(
    xyz: np.ndarray,
    signals: dict[str, np.ndarray],
) -> tuple[dict[str, object], dict[str, float]]:
    """
    Compatibility wrapper matching the A20 benchmark interface.

    The benchmark historically expects:

        result, timings = run_pipeline(...)

    while the public A20 API uses FoveaMapPipelineResult.
    """

    result = run_foveamap_pipeline(
        xyz,
        signals,
    )

    return (
        {
            "foveation": result.foveation,
            "leaf_map": result.leaf_map,
        },
        result.timings_ms,
    )