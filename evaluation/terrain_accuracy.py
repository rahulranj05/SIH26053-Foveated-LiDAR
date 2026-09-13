"""
A33 — Elevation + Terrain Accuracy Evaluation

Evaluates the numerical accuracy of a 2.5D elevation representation
against a reference terrain representation.

A33 is intentionally independent from the frozen FoveaMap pipeline.

It does NOT modify:
    - A20 FoveaMap pipeline
    - A21 SensorFrame
    - A25 temporal motion
    - A26 ego-motion
    - A29 sequential dataset
    - A30 signal integration
    - A31 temporal FoveaMap pipeline
    - A32 sequential dataset runner

The evaluator operates on dense 2D elevation rasters so that the same
evaluation machinery can later be used for:
    - uniform maps
    - adaptive maps rasterized onto a common reference grid
    - SemanticKITTI
    - RELLIS-3D
    - CARLA ground truth
    - held-out terrain evaluation

Metrics
-------

Elevation:
    - MAE
    - RMSE
    - bias
    - maximum absolute error
    - 95th percentile absolute error
    - valid-cell coverage

Terrain:
    - slope MAE
    - slope RMSE
    - slope bias
    - roughness MAE
    - roughness RMSE
    - roughness bias

Terrain features are computed directly from the elevation raster.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ScalarErrorMetrics:
    """
    Standard error statistics for one scalar terrain quantity.
    """

    mae: float
    rmse: float
    bias: float
    max_abs_error: float
    p95_abs_error: float
    valid_count: int

    def __post_init__(self) -> None:
        if self.valid_count < 0:
            raise ValueError(
                "valid_count must be non-negative"
            )

        values = (
            self.mae,
            self.rmse,
            self.bias,
            self.max_abs_error,
            self.p95_abs_error,
        )

        if not all(
            np.isfinite(value)
            for value in values
        ):
            raise ValueError(
                "error metrics must be finite"
            )


@dataclass(frozen=True)
class ElevationAccuracyResult:
    """
    Elevation accuracy evaluation result.
    """

    metrics: ScalarErrorMetrics
    coverage: float
    total_reference_cells: int
    valid_reference_cells: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.coverage <= 1.0:
            raise ValueError(
                "coverage must be in [0, 1]"
            )

        if self.total_reference_cells < 0:
            raise ValueError(
                "total_reference_cells must be non-negative"
            )

        if self.valid_reference_cells < 0:
            raise ValueError(
                "valid_reference_cells must be non-negative"
            )


@dataclass(frozen=True)
class TerrainAccuracyResult:
    """
    Complete A33 terrain accuracy result.
    """

    elevation: ElevationAccuracyResult
    slope: ScalarErrorMetrics
    roughness: ScalarErrorMetrics
    resolution_m: float

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.resolution_m)
            or self.resolution_m <= 0.0
        ):
            raise ValueError(
                "resolution_m must be positive and finite"
            )

    def as_dict(self) -> dict[str, Any]:
        """
        Return a JSON-friendly summary dictionary.
        """

        return {
            "resolution_m": self.resolution_m,
            "elevation": {
                "mae": self.elevation.metrics.mae,
                "rmse": self.elevation.metrics.rmse,
                "bias": self.elevation.metrics.bias,
                "max_abs_error": (
                    self.elevation.metrics.max_abs_error
                ),
                "p95_abs_error": (
                    self.elevation.metrics.p95_abs_error
                ),
                "valid_count": (
                    self.elevation.metrics.valid_count
                ),
                "coverage": self.elevation.coverage,
            },
            "slope": {
                "mae": self.slope.mae,
                "rmse": self.slope.rmse,
                "bias": self.slope.bias,
                "max_abs_error": (
                    self.slope.max_abs_error
                ),
                "p95_abs_error": (
                    self.slope.p95_abs_error
                ),
                "valid_count": self.slope.valid_count,
            },
            "roughness": {
                "mae": self.roughness.mae,
                "rmse": self.roughness.rmse,
                "bias": self.roughness.bias,
                "max_abs_error": (
                    self.roughness.max_abs_error
                ),
                "p95_abs_error": (
                    self.roughness.p95_abs_error
                ),
                "valid_count": self.roughness.valid_count,
            },
        }


def _validate_elevation_raster(
    elevation: np.ndarray,
    name: str,
) -> np.ndarray:
    """
    Validate and normalize a 2D elevation raster.
    """

    array = np.asarray(
        elevation,
        dtype=np.float64,
    )

    if array.ndim != 2:
        raise ValueError(
            f"{name} must be a 2D array"
        )

    if array.size == 0:
        raise ValueError(
            f"{name} must not be empty"
        )

    return array


def _validate_same_shape(
    reference: np.ndarray,
    predicted: np.ndarray,
) -> None:
    """
    Require reference and prediction to share a raster shape.
    """

    if reference.shape != predicted.shape:
        raise ValueError(
            "reference and predicted elevation rasters "
            "must have the same shape"
        )


def _finite_pair_mask(
    reference: np.ndarray,
    predicted: np.ndarray,
) -> np.ndarray:
    """
    Return cells where both rasters contain finite values.
    """

    return (
        np.isfinite(reference)
        & np.isfinite(predicted)
    )


def _compute_scalar_metrics(
    reference: np.ndarray,
    predicted: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> ScalarErrorMetrics:
    """
    Compute standard scalar error metrics.
    """

    reference = np.asarray(
        reference,
        dtype=np.float64,
    )

    predicted = np.asarray(
        predicted,
        dtype=np.float64,
    )

    if reference.shape != predicted.shape:
        raise ValueError(
            "reference and predicted must have "
            "the same shape"
        )

    mask = _finite_pair_mask(
        reference,
        predicted,
    )

    if valid_mask is not None:
        valid_mask = np.asarray(
            valid_mask,
            dtype=bool,
        )

        if valid_mask.shape != reference.shape:
            raise ValueError(
                "valid_mask must have the same shape "
                "as reference and predicted"
            )

        mask &= valid_mask

    if not np.any(mask):
        raise ValueError(
            "no valid cells are available for evaluation"
        )

    error = (
        predicted[mask]
        - reference[mask]
    )

    absolute_error = np.abs(error)

    return ScalarErrorMetrics(
        mae=float(
            np.mean(absolute_error)
        ),
        rmse=float(
            np.sqrt(
                np.mean(error ** 2)
            )
        ),
        bias=float(
            np.mean(error)
        ),
        max_abs_error=float(
            np.max(absolute_error)
        ),
        p95_abs_error=float(
            np.percentile(
                absolute_error,
                95.0,
            )
        ),
        valid_count=int(
            error.size
        ),
    )


def elevation_error_metrics(
    reference: np.ndarray,
    predicted: np.ndarray,
) -> ElevationAccuracyResult:
    """
    Evaluate predicted elevation against reference elevation.

    Parameters
    ----------
    reference:
        Ground-truth elevation raster.

    predicted:
        FoveaMap elevation raster.

    Returns
    -------
    ElevationAccuracyResult
    """

    reference = _validate_elevation_raster(
        reference,
        "reference",
    )

    predicted = _validate_elevation_raster(
        predicted,
        "predicted",
    )

    _validate_same_shape(
        reference,
        predicted,
    )

    valid_mask = _finite_pair_mask(
        reference,
        predicted,
    )

    metrics = _compute_scalar_metrics(
        reference,
        predicted,
    )

    total_cells = int(
        reference.size
    )

    valid_cells = int(
        np.count_nonzero(
            np.isfinite(reference)
        )
    )

    coverage = (
        float(
            np.count_nonzero(valid_mask)
        )
        / float(valid_cells)
        if valid_cells > 0
        else 0.0
    )

    return ElevationAccuracyResult(
        metrics=metrics,
        coverage=coverage,
        total_reference_cells=total_cells,
        valid_reference_cells=valid_cells,
    )


def compute_slope(
    elevation: np.ndarray,
    resolution_m: float,
) -> np.ndarray:
    """
    Compute terrain slope magnitude in degrees.

    The raster is interpreted as a regular XY elevation grid.

    Slope is:

        atan(
            sqrt(
                dz/dx^2 + dz/dy^2
            )
        )

    expressed in degrees.
    """

    elevation = _validate_elevation_raster(
        elevation,
        "elevation",
    )

    if (
        not np.isfinite(resolution_m)
        or resolution_m <= 0.0
    ):
        raise ValueError(
            "resolution_m must be positive and finite"
        )

    if min(elevation.shape) < 2:
        raise ValueError(
            "elevation raster must have at least "
            "two rows and two columns"
        )

    dz_dy, dz_dx = np.gradient(
        elevation,
        resolution_m,
        resolution_m,
    )

    gradient_magnitude = np.sqrt(
        dz_dx ** 2
        + dz_dy ** 2
    )

    return np.degrees(
        np.arctan(
            gradient_magnitude
        )
    )


def compute_roughness(
    elevation: np.ndarray,
) -> np.ndarray:
    """
    Compute local terrain roughness as absolute deviation
    from the 3x3 neighborhood mean.

    Boundary cells use the available neighborhood.
    """

    elevation = _validate_elevation_raster(
        elevation,
        "elevation",
    )

    if min(elevation.shape) < 3:
        raise ValueError(
            "elevation raster must have at least "
            "three rows and three columns"
        )

    padded = np.pad(
        elevation,
        pad_width=1,
        mode="edge",
    )

    local_sum = np.zeros_like(
        elevation,
        dtype=np.float64,
    )

    for row_offset in range(3):
        for col_offset in range(3):
            local_sum += padded[
                row_offset : row_offset
                + elevation.shape[0],
                col_offset : col_offset
                + elevation.shape[1],
            ]

    local_mean = (
        local_sum / 9.0
    )

    return np.abs(
        elevation - local_mean
    )


def evaluate_terrain_accuracy(
    reference: np.ndarray,
    predicted: np.ndarray,
    *,
    resolution_m: float,
) -> TerrainAccuracyResult:
    """
    Perform complete A33 terrain accuracy evaluation.

    This evaluates:

        1. elevation
        2. slope
        3. roughness

    on a common raster resolution.
    """

    reference = _validate_elevation_raster(
        reference,
        "reference",
    )

    predicted = _validate_elevation_raster(
        predicted,
        "predicted",
    )

    _validate_same_shape(
        reference,
        predicted,
    )

    elevation_result = elevation_error_metrics(
        reference,
        predicted,
    )

    reference_slope = compute_slope(
        reference,
        resolution_m,
    )

    predicted_slope = compute_slope(
        predicted,
        resolution_m,
    )

    reference_roughness = compute_roughness(
        reference,
    )

    predicted_roughness = compute_roughness(
        predicted,
    )

    slope_metrics = _compute_scalar_metrics(
        reference_slope,
        predicted_slope,
    )

    roughness_metrics = _compute_scalar_metrics(
        reference_roughness,
        predicted_roughness,
    )

    return TerrainAccuracyResult(
        elevation=elevation_result,
        slope=slope_metrics,
        roughness=roughness_metrics,
        resolution_m=float(
            resolution_m
        ),
    )


__all__ = [
    "ScalarErrorMetrics",
    "ElevationAccuracyResult",
    "TerrainAccuracyResult",
    "elevation_error_metrics",
    "compute_slope",
    "compute_roughness",
    "evaluate_terrain_accuracy",
]