"""
A21 — Canonical Sensor Frame Interface

Defines the simulator- and dataset-independent perception frame used by
FoveaMap.

A SensorFrame can originate from:
    - SemanticKITTI
    - CARLA
    - Webots
    - Gazebo
    - real LiDAR hardware

The frame is intentionally independent of any simulator or dataset.

A21 does NOT modify the A17/A18.3/A20 implementations. It provides a
canonical boundary through which those components can receive perception
data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

import numpy as np

from mapping.kinematic_foveation import VehicleState


@dataclass(frozen=True)
class SensorFrame:
    """
    Canonical single-frame perception representation.

    Parameters
    ----------
    xyz:
        LiDAR point coordinates with shape (N, 3), in metres.

    semantic_labels:
        Optional per-point semantic labels with shape (N,).
        If supplied, labels are integer-valued.

    dynamic_probability:
        Optional per-point dynamic probability in [0, 1], shape (N,).

    vehicle_state:
        Ego-vehicle state used by kinematic foveation.

    timestamp:
        Frame timestamp in seconds.

    frame_id:
        Source-specific frame identifier.

    Notes
    -----
    A SensorFrame is immutable after construction. Input arrays are copied
    and marked read-only so downstream stages cannot accidentally mutate
    the canonical frame.
    """

    xyz: np.ndarray
    vehicle_state: VehicleState
    timestamp: float
    frame_id: Hashable
    semantic_labels: np.ndarray | None = None
    dynamic_probability: np.ndarray | None = None

    def __post_init__(self) -> None:
        xyz = np.asarray(self.xyz, dtype=np.float64)

        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(
                f"xyz must have shape (N, 3), got {xyz.shape}"
            )

        if not np.all(np.isfinite(xyz)):
            raise ValueError("xyz must contain only finite values")

        if not np.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")

        if not isinstance(self.vehicle_state, VehicleState):
            raise TypeError("vehicle_state must be a VehicleState")

        semantic_labels = self._validate_semantic_labels(
            self.semantic_labels,
            xyz.shape[0],
        )

        dynamic_probability = self._validate_dynamic_probability(
            self.dynamic_probability,
            xyz.shape[0],
        )

        xyz = np.array(xyz, dtype=np.float64, copy=True)
        xyz.setflags(write=False)

        if semantic_labels is not None:
            semantic_labels = np.array(
                semantic_labels,
                dtype=np.int64,
                copy=True,
            )
            semantic_labels.setflags(write=False)

        if dynamic_probability is not None:
            dynamic_probability = np.array(
                dynamic_probability,
                dtype=np.float64,
                copy=True,
            )
            dynamic_probability.setflags(write=False)

        object.__setattr__(self, "xyz", xyz)
        object.__setattr__(self, "semantic_labels", semantic_labels)
        object.__setattr__(
            self,
            "dynamic_probability",
            dynamic_probability,
        )

    @staticmethod
    def _validate_semantic_labels(
        labels: np.ndarray | None,
        num_points: int,
    ) -> np.ndarray | None:
        if labels is None:
            return None

        labels = np.asarray(labels)

        if labels.ndim != 1:
            raise ValueError(
                "semantic_labels must be a 1D array"
            )

        if labels.shape[0] != num_points:
            raise ValueError(
                "semantic_labels length must match xyz point count"
            )

        if not np.all(np.isfinite(labels)):
            raise ValueError(
                "semantic_labels must contain only finite values"
            )

        if not np.all(labels == np.floor(labels)):
            raise ValueError(
                "semantic_labels must contain integer values"
            )

        return labels

    @staticmethod
    def _validate_dynamic_probability(
        probability: np.ndarray | None,
        num_points: int,
    ) -> np.ndarray | None:
        if probability is None:
            return None

        probability = np.asarray(probability, dtype=np.float64)

        if probability.ndim != 1:
            raise ValueError(
                "dynamic_probability must be a 1D array"
            )

        if probability.shape[0] != num_points:
            raise ValueError(
                "dynamic_probability length must match xyz point count"
            )

        if not np.all(np.isfinite(probability)):
            raise ValueError(
                "dynamic_probability must contain only finite values"
            )

        if np.any(probability < 0.0) or np.any(probability > 1.0):
            raise ValueError(
                "dynamic_probability values must be in [0, 1]"
            )

        return probability

    @property
    def num_points(self) -> int:
        """Return the number of LiDAR points in the frame."""
        return int(self.xyz.shape[0])

    @property
    def has_semantics(self) -> bool:
        """Return whether semantic labels are available."""
        return self.semantic_labels is not None

    @property
    def has_dynamic_probability(self) -> bool:
        """Return whether dynamic probabilities are available."""
        return self.dynamic_probability is not None

    def to_foveamap_signals(self) -> dict[str, np.ndarray]:
        """
        Convert available frame annotations into FoveaMap signal arrays.

        The returned dictionary is deliberately compatible with the signal
        interface consumed by A20.

        Currently exposed signals:
            semantic
            dynamic

        Missing optional annotations are simply omitted.

        Kinematic state is retained in ``vehicle_state`` and is intentionally
        not converted into an array here because A13/A17's existing
        interfaces operate differently from per-point signals.
        """
        signals: dict[str, np.ndarray] = {}

        if self.semantic_labels is not None:
            signals["semantic"] = self.semantic_labels.astype(
                np.float64,
                copy=True,
            )

        if self.dynamic_probability is not None:
            signals["dynamic"] = self.dynamic_probability.copy()

        return signals

    def copy(self) -> "SensorFrame":
        """
        Return an independent copy of the canonical frame.
        """
        return SensorFrame(
            xyz=self.xyz.copy(),
            semantic_labels=(
                None
                if self.semantic_labels is None
                else self.semantic_labels.copy()
            ),
            dynamic_probability=(
                None
                if self.dynamic_probability is None
                else self.dynamic_probability.copy()
            ),
            vehicle_state=self.vehicle_state,
            timestamp=self.timestamp,
            frame_id=self.frame_id,
        )