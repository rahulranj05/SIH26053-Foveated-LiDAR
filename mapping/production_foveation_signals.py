"""
Production foveation signal construction.

This module converts the canonical SensorFrame into the three
production foveation signals owned by the perception layer:

    DISTANCE
    KINEMATIC
    PREDICTED_PATH

It deliberately does NOT:
    - fuse signals
    - choose the final resolution
    - select the dominant reason
    - invoke the hierarchical mapper

Those responsibilities remain with A17 and A18.3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mapping.kinematic_foveation import (
    KinematicFoveationConfig,
    kinematic_importance,
)
from mapping.predicted_path_foveation import path_importance
from mapping.sensor_frame import SensorFrame


@dataclass(frozen=True)
class ProductionFoveationSignalConfig:
    """Configuration for production foveation signal construction."""

    maximum_distance: float = 100.0
    path_horizon: float = 5.0
    path_step: float = 0.25
    path_corridor_width: float = 3.0
    kinematic: KinematicFoveationConfig = KinematicFoveationConfig()

    def __post_init__(self) -> None:
        if not np.isfinite(self.maximum_distance):
            raise ValueError("maximum_distance must be finite")
        if self.maximum_distance <= 0.0:
            raise ValueError("maximum_distance must be positive")

        if not np.isfinite(self.path_horizon):
            raise ValueError("path_horizon must be finite")
        if self.path_horizon <= 0.0:
            raise ValueError("path_horizon must be positive")

        if not np.isfinite(self.path_step):
            raise ValueError("path_step must be finite")
        if self.path_step <= 0.0:
            raise ValueError("path_step must be positive")

        if not np.isfinite(self.path_corridor_width):
            raise ValueError("path_corridor_width must be finite")
        if self.path_corridor_width <= 0.0:
            raise ValueError("path_corridor_width must be positive")


def _validate_xyz(xyz: np.ndarray) -> np.ndarray:
    """Validate and normalize a point cloud to float64."""

    points = np.asarray(xyz, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3)")

    if not np.all(np.isfinite(points)):
        raise ValueError("xyz must contain only finite values")

    return points


def _validate_signal(
    signal: np.ndarray,
    expected_length: int,
    *,
    name: str,
) -> np.ndarray:
    """Validate one normalized foveation signal."""

    values = np.asarray(signal, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")

    if values.shape[0] != expected_length:
        raise ValueError(
            f"{name} length must match point count "
            f"({expected_length}), got {values.shape[0]}"
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")

    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(f"{name} must be in [0, 1]")

    return values


def distance_importance(
    xyz: np.ndarray,
    *,
    maximum_distance: float = 100.0,
) -> np.ndarray:
    """
    Compute distance-based foveation importance.

    Importance is 1.0 at zero horizontal distance and decreases
    linearly to 0.0 at maximum_distance.
    """

    if not np.isfinite(maximum_distance):
        raise ValueError("maximum_distance must be finite")

    if maximum_distance <= 0.0:
        raise ValueError("maximum_distance must be positive")

    points = _validate_xyz(xyz)

    distance = np.linalg.norm(points[:, :2], axis=1)

    importance = np.clip(
        1.0 - (distance / maximum_distance),
        0.0,
        1.0,
    )

    return importance


def build_production_foveation_signals(
    frame: SensorFrame,
    *,
    config: ProductionFoveationSignalConfig | None = None,
) -> dict[str, np.ndarray]:
    """
    Build the canonical production foveation signals for one SensorFrame.

    Returns exactly:
        DISTANCE
        KINEMATIC
        PREDICTED_PATH
    """

    if not isinstance(frame, SensorFrame):
        raise TypeError("frame must be a SensorFrame")

    if config is None:
        config = ProductionFoveationSignalConfig()

    xyz = _validate_xyz(frame.xyz)
    point_count = xyz.shape[0]

    distance = distance_importance(
        xyz,
        maximum_distance=config.maximum_distance,
    )

    kinematic = kinematic_importance(
        xyz,
        frame.vehicle_state,
        config=config.kinematic,
    )

    predicted_path = path_importance(
        xyz,
        frame.vehicle_state,
        horizon=config.path_horizon,
        step=config.path_step,
        corridor_width=config.path_corridor_width,
    )

    signals = {
        "DISTANCE": _validate_signal(
            distance,
            point_count,
            name="DISTANCE",
        ),
        "KINEMATIC": _validate_signal(
            kinematic,
            point_count,
            name="KINEMATIC",
        ),
        "PREDICTED_PATH": _validate_signal(
            predicted_path,
            point_count,
            name="PREDICTED_PATH",
        ),
    }

    return signals


def make_production_foveation_provider(
    *,
    config: ProductionFoveationSignalConfig | None = None,
):
    """
    Create a callable production signal provider.

    The returned provider accepts a SensorFrame and returns the
    canonical production foveation signals.
    """

    if config is None:
        config = ProductionFoveationSignalConfig()

    def provider(frame: SensorFrame) -> dict[str, np.ndarray]:
        return build_production_foveation_signals(
            frame,
            config=config,
        )

    return provider