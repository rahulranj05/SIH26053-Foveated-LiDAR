from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SEMANTIC_CLASSES = {
    "UNKNOWN": 0,
    "DRIVABLE": 1,
    "NON_DRIVABLE": 2,
    "STATIC_OBSTACLE": 3,
    "VEHICLE": 4,
    "VULNERABLE_USER": 5,
}


DEFAULT_SEMANTIC_IMPORTANCE = {
    0: 0.35,  # UNKNOWN
    1: 0.00,  # DRIVABLE
    2: 0.40,  # NON_DRIVABLE
    3: 0.75,  # STATIC_OBSTACLE
    4: 0.85,  # VEHICLE
    5: 1.00,  # VULNERABLE_USER
}


@dataclass(frozen=True)
class SemanticFoveationConfig:
    semantic_weight: float = 1.0

    unknown_importance: float = 0.35
    drivable_importance: float = 0.00
    non_drivable_importance: float = 0.40
    static_obstacle_importance: float = 0.75
    vehicle_importance: float = 0.85
    vulnerable_user_importance: float = 1.00

    fine_threshold: float = 0.75
    medium_threshold: float = 0.50
    coarse_threshold: float = 0.25

    fine_resolution: float = 0.05
    medium_resolution: float = 0.10
    coarse_resolution: float = 0.20
    far_resolution: float = 0.40

    def __post_init__(self) -> None:
        if not 0.0 <= self.semantic_weight <= 1.0:
            raise ValueError(
                "semantic_weight must be between 0.0 and 1.0."
            )

        importance_values = (
            self.unknown_importance,
            self.drivable_importance,
            self.non_drivable_importance,
            self.static_obstacle_importance,
            self.vehicle_importance,
            self.vulnerable_user_importance,
        )

        if any(not 0.0 <= value <= 1.0 for value in importance_values):
            raise ValueError(
                "All semantic importance values must be between 0.0 and 1.0."
            )

        thresholds = (
            self.fine_threshold,
            self.medium_threshold,
            self.coarse_threshold,
        )

        if any(not 0.0 <= value <= 1.0 for value in thresholds):
            raise ValueError(
                "All importance thresholds must be between 0.0 and 1.0."
            )

        if not (
            self.fine_threshold
            >= self.medium_threshold
            >= self.coarse_threshold
        ):
            raise ValueError(
                "Importance thresholds must satisfy "
                "fine >= medium >= coarse."
            )

        resolutions = (
            self.fine_resolution,
            self.medium_resolution,
            self.coarse_resolution,
            self.far_resolution,
        )

        if any(value <= 0.0 for value in resolutions):
            raise ValueError(
                "All resolutions must be positive."
            )


def validate_semantic_labels(
    semantic_label: np.ndarray,
    expected_length: int | None = None,
) -> bool:
    """
    Validate a semantic-label array.

    Semantic labels must be a one-dimensional integer array containing
    the supported ontology IDs 0 through 5.
    """

    labels = np.asarray(semantic_label)

    if labels.ndim != 1:
        raise ValueError(
            "semantic_label must be a 1D array."
        )

    if expected_length is not None and len(labels) != expected_length:
        raise ValueError(
            "semantic_label length must match the expected number of points."
        )

    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(
            "semantic_label must contain integer labels."
        )

    valid_ids = set(SEMANTIC_CLASSES.values())

    if labels.size > 0:
        unique_labels = np.unique(labels)

        if any(int(label) not in valid_ids for label in unique_labels):
            raise ValueError(
                "semantic_label contains unsupported semantic class IDs."
            )

    return True


def _importance_lookup(
    config: SemanticFoveationConfig,
) -> np.ndarray:
    """Build a lookup table for the semantic ontology."""

    return np.array(
        [
            config.unknown_importance,
            config.drivable_importance,
            config.non_drivable_importance,
            config.static_obstacle_importance,
            config.vehicle_importance,
            config.vulnerable_user_importance,
        ],
        dtype=np.float64,
    )


def semantic_importance(
    semantic_label: np.ndarray,
    config: SemanticFoveationConfig | None = None,
) -> np.ndarray:
    """
    Convert semantic class IDs into normalized semantic importance.

    Higher importance means the mapper should preserve finer spatial
    resolution around that semantic category.
    """

    if config is None:
        config = SemanticFoveationConfig()

    labels = np.asarray(semantic_label)

    validate_semantic_labels(labels)

    lookup = _importance_lookup(config)

    return lookup[labels.astype(np.int64)]


def fuse_semantic_importance(
    base_importance: np.ndarray,
    semantic_importance_values: np.ndarray,
    config: SemanticFoveationConfig | None = None,
) -> np.ndarray:
    """
    Fuse geometric/path importance with semantic importance.

    Semantic information may refine an existing geometric importance,
    but it must never reduce it.

    semantic_weight = 0:
        use geometric/base importance unchanged.

    semantic_weight = 1:
        use the stronger of base and semantic importance.
    """

    if config is None:
        config = SemanticFoveationConfig()

    base = np.asarray(base_importance, dtype=np.float64)
    semantic = np.asarray(
        semantic_importance_values,
        dtype=np.float64,
    )

    if base.ndim != 1 or semantic.ndim != 1:
        raise ValueError(
            "base_importance and semantic_importance must be 1D arrays."
        )

    if base.shape != semantic.shape:
        raise ValueError(
            "base_importance and semantic_importance must have the same shape."
        )

    if not np.all(np.isfinite(base)):
        raise ValueError(
            "base_importance must contain only finite values."
        )

    if not np.all(np.isfinite(semantic)):
        raise ValueError(
            "semantic_importance must contain only finite values."
        )

    if np.any(base < 0.0) or np.any(base > 1.0):
        raise ValueError(
            "base_importance values must be between 0.0 and 1.0."
        )

    if np.any(semantic < 0.0) or np.any(semantic > 1.0):
        raise ValueError(
            "semantic_importance values must be between 0.0 and 1.0."
        )

    stronger_importance = np.maximum(base, semantic)

    fused = (
        (1.0 - config.semantic_weight) * base
        + config.semantic_weight * stronger_importance
    )

    return np.clip(fused, 0.0, 1.0)


def resolution_from_semantic_importance(
    importance: np.ndarray,
    config: SemanticFoveationConfig | None = None,
) -> np.ndarray:
    """
    Convert semantic importance into hierarchy-safe map resolution.

    0.75+ -> 5 cm
    0.50-0.75 -> 10 cm
    0.25-0.50 -> 20 cm
    below 0.25 -> 40 cm
    """

    if config is None:
        config = SemanticFoveationConfig()

    values = np.asarray(importance, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError(
            "importance must be a 1D array."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "importance must contain only finite values."
        )

    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(
            "importance values must be between 0.0 and 1.0."
        )

    resolution = np.full(
        values.shape,
        config.far_resolution,
        dtype=np.float64,
    )

    resolution[
        values >= config.coarse_threshold
    ] = config.coarse_resolution

    resolution[
        values >= config.medium_threshold
    ] = config.medium_resolution

    resolution[
        values >= config.fine_threshold
    ] = config.fine_resolution

    return resolution


def semantic_resolution(
    semantic_label: np.ndarray,
    config: SemanticFoveationConfig | None = None,
) -> np.ndarray:
    """
    Convenience function:

    semantic labels
        -> semantic importance
        -> hierarchy-safe resolution
    """

    if config is None:
        config = SemanticFoveationConfig()

    importance = semantic_importance(
        semantic_label,
        config=config,
    )

    return resolution_from_semantic_importance(
        importance,
        config=config,
    )