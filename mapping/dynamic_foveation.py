from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DynamicFoveationConfig:
    """
    Configuration for dynamic-object-aware foveation.

    Dynamic probability and object velocity are converted into
    an additional importance value.

    The resulting importance is mapped to the same
    alignment-safe resolution hierarchy used by FoveaMap:

        5 cm
        10 cm
        20 cm
        40 cm
    """

    dynamic_weight: float = 1.0
    velocity_weight: float = 0.5

    velocity_reference: float = 10.0

    fine_threshold: float = 0.75
    medium_threshold: float = 0.50
    coarse_threshold: float = 0.25

    fine_resolution: float = 0.05
    medium_resolution: float = 0.10
    coarse_resolution: float = 0.20
    far_resolution: float = 0.40

    def __post_init__(self) -> None:
        if self.dynamic_weight < 0.0:
            raise ValueError(
                "dynamic_weight must be non-negative."
            )

        if self.velocity_weight < 0.0:
            raise ValueError(
                "velocity_weight must be non-negative."
            )

        if self.velocity_reference <= 0.0:
            raise ValueError(
                "velocity_reference must be positive."
            )

        if not (
            0.0
            <= self.coarse_threshold
            <= self.medium_threshold
            <= self.fine_threshold
            <= 1.0
        ):
            raise ValueError(
                "Importance thresholds must satisfy "
                "0 <= coarse <= medium <= fine <= 1."
            )

        resolutions = (
            self.fine_resolution,
            self.medium_resolution,
            self.coarse_resolution,
            self.far_resolution,
        )

        if any(resolution <= 0.0 for resolution in resolutions):
            raise ValueError(
                "All resolutions must be positive."
            )


def validate_dynamic_inputs(
    dynamic_probability: np.ndarray,
    velocity: np.ndarray,
    expected_length: int | None = None,
) -> bool:
    """
    Validate dynamic probability and velocity arrays.

    Args:
        dynamic_probability:
            1D array containing dynamic probabilities [0, 1].

        velocity:
            1D array containing object/point velocity magnitudes
            in m/s.

        expected_length:
            Optional expected number of elements.

    Returns:
        True when inputs are valid.
    """

    dynamic_probability = np.asarray(
        dynamic_probability
    )

    velocity = np.asarray(velocity)

    if dynamic_probability.ndim != 1:
        raise ValueError(
            "dynamic_probability must be a 1D array."
        )

    if velocity.ndim != 1:
        raise ValueError(
            "velocity must be a 1D array."
        )

    if len(dynamic_probability) != len(velocity):
        raise ValueError(
            "dynamic_probability and velocity must have "
            "the same length."
        )

    if expected_length is not None:
        if len(dynamic_probability) != expected_length:
            raise ValueError(
                "Input length does not match expected_length."
            )

    if not np.issubdtype(
        dynamic_probability.dtype,
        np.number,
    ):
        raise ValueError(
            "dynamic_probability must contain numeric values."
        )

    if not np.issubdtype(
        velocity.dtype,
        np.number,
    ):
        raise ValueError(
            "velocity must contain numeric values."
        )

    if not np.all(
        np.isfinite(dynamic_probability)
    ):
        raise ValueError(
            "dynamic_probability must contain finite values."
        )

    if not np.all(np.isfinite(velocity)):
        raise ValueError(
            "velocity must contain finite values."
        )

    if np.any(
        (dynamic_probability < 0.0)
        | (dynamic_probability > 1.0)
    ):
        raise ValueError(
            "dynamic_probability must be in [0, 1]."
        )

    if np.any(velocity < 0.0):
        raise ValueError(
            "velocity must be non-negative."
        )

    return True


def normalize_velocity(
    velocity: np.ndarray,
    config: DynamicFoveationConfig | None = None,
) -> np.ndarray:
    """
    Normalize velocity relative to a configurable reference speed.

    A velocity equal to velocity_reference maps to 1.0.
    Larger velocities are clipped to 1.0.
    """

    if config is None:
        config = DynamicFoveationConfig()

    velocity = np.asarray(
        velocity,
        dtype=np.float64,
    )

    return np.clip(
        velocity / config.velocity_reference,
        0.0,
        1.0,
    )


def dynamic_importance(
    dynamic_probability: np.ndarray,
    velocity: np.ndarray,
    config: DynamicFoveationConfig | None = None,
) -> np.ndarray:
    """
    Convert dynamic probability and velocity into dynamic importance.

    Dynamic probability is the primary signal.

    Velocity provides an additional refinement signal so that
    faster-moving objects receive more attention.

    The result is always clipped to [0, 1].
    """

    if config is None:
        config = DynamicFoveationConfig()

    dynamic_probability = np.asarray(
        dynamic_probability,
        dtype=np.float64,
    )

    velocity = np.asarray(
        velocity,
        dtype=np.float64,
    )

    validate_dynamic_inputs(
        dynamic_probability,
        velocity,
    )

    normalized_velocity = normalize_velocity(
        velocity,
        config,
    )

    importance = (
        config.dynamic_weight
        * dynamic_probability
        + config.velocity_weight
        * dynamic_probability
        * normalized_velocity
    )

    return np.clip(
        importance,
        0.0,
        1.0,
    )


def fuse_dynamic_importance(
    base_importance: np.ndarray,
    dynamic_importance_values: np.ndarray,
    config: DynamicFoveationConfig | None = None,
) -> np.ndarray:
    """
    Fuse dynamic importance with an existing importance field.

    Dynamic information can refine the map but must never reduce
    an existing geometric, kinematic, path, or semantic priority.

    Therefore the fused importance is the element-wise maximum
    of the base importance and dynamic importance.
    """

    if config is None:
        config = DynamicFoveationConfig()

    base_importance = np.asarray(
        base_importance,
        dtype=np.float64,
    )

    dynamic_importance_values = np.asarray(
        dynamic_importance_values,
        dtype=np.float64,
    )

    if base_importance.ndim != 1:
        raise ValueError(
            "base_importance must be a 1D array."
        )

    if dynamic_importance_values.ndim != 1:
        raise ValueError(
            "dynamic_importance_values must be a 1D array."
        )

    if len(base_importance) != len(
        dynamic_importance_values
    ):
        raise ValueError(
            "base_importance and dynamic_importance_values "
            "must have the same length."
        )

    if not np.all(np.isfinite(base_importance)):
        raise ValueError(
            "base_importance must contain finite values."
        )

    if not np.all(
        np.isfinite(dynamic_importance_values)
    ):
        raise ValueError(
            "dynamic_importance_values must contain "
            "finite values."
        )

    if np.any(
        (base_importance < 0.0)
        | (base_importance > 1.0)
    ):
        raise ValueError(
            "base_importance must be in [0, 1]."
        )

    if np.any(
        (dynamic_importance_values < 0.0)
        | (dynamic_importance_values > 1.0)
    ):
        raise ValueError(
            "dynamic_importance_values must be in [0, 1]."
        )

    if config.dynamic_weight == 0.0:
        return base_importance.copy()

    return np.maximum(
        base_importance,
        dynamic_importance_values,
    )


def resolution_from_dynamic_importance(
    importance: np.ndarray,
    config: DynamicFoveationConfig | None = None,
) -> np.ndarray:
    """
    Convert dynamic importance into hierarchical map resolution.

    >= fine threshold   -> 5 cm
    >= medium threshold -> 10 cm
    >= coarse threshold -> 20 cm
    otherwise            -> 40 cm
    """

    if config is None:
        config = DynamicFoveationConfig()

    importance = np.asarray(
        importance,
        dtype=np.float64,
    )

    if importance.ndim != 1:
        raise ValueError(
            "importance must be a 1D array."
        )

    if not np.all(np.isfinite(importance)):
        raise ValueError(
            "importance must contain finite values."
        )

    if np.any(
        (importance < 0.0)
        | (importance > 1.0)
    ):
        raise ValueError(
            "importance must be in [0, 1]."
        )

    resolution = np.full(
        importance.shape,
        config.far_resolution,
        dtype=np.float64,
    )

    medium_mask = (
        importance >= config.coarse_threshold
    )

    resolution[medium_mask] = (
        config.coarse_resolution
    )

    fine_mask = (
        importance >= config.medium_threshold
    )

    resolution[fine_mask] = (
        config.medium_resolution
    )

    very_fine_mask = (
        importance >= config.fine_threshold
    )

    resolution[very_fine_mask] = (
        config.fine_resolution
    )

    return resolution


def dynamic_resolution(
    dynamic_probability: np.ndarray,
    velocity: np.ndarray,
    config: DynamicFoveationConfig | None = None,
) -> np.ndarray:
    """
    Convenience function:

        dynamic probability + velocity
                    ↓
             dynamic importance
                    ↓
               resolution
    """

    importance = dynamic_importance(
        dynamic_probability,
        velocity,
        config,
    )

    return resolution_from_dynamic_importance(
        importance,
        config,
    )