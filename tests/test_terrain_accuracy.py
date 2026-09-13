import numpy as np
import pytest

from evaluation.terrain_accuracy import (
    compute_roughness,
    compute_slope,
    elevation_error_metrics,
    evaluate_terrain_accuracy,
)


def test_perfect_elevation_has_zero_error():
    elevation = np.ones(
        (5, 5),
        dtype=np.float64,
    )

    result = elevation_error_metrics(
        elevation,
        elevation,
    )

    assert result.metrics.mae == pytest.approx(0.0)
    assert result.metrics.rmse == pytest.approx(0.0)
    assert result.metrics.bias == pytest.approx(0.0)
    assert result.metrics.max_abs_error == pytest.approx(0.0)
    assert result.metrics.p95_abs_error == pytest.approx(0.0)
    assert result.coverage == pytest.approx(1.0)


def test_constant_offset_has_expected_error():
    reference = np.zeros(
        (4, 4),
        dtype=np.float64,
    )

    predicted = np.full(
        (4, 4),
        0.20,
        dtype=np.float64,
    )

    result = elevation_error_metrics(
        reference,
        predicted,
    )

    assert result.metrics.mae == pytest.approx(0.20)
    assert result.metrics.rmse == pytest.approx(0.20)
    assert result.metrics.bias == pytest.approx(0.20)
    assert result.metrics.max_abs_error == pytest.approx(0.20)
    assert result.metrics.p95_abs_error == pytest.approx(0.20)


def test_positive_and_negative_errors_cancel_in_bias():
    reference = np.zeros(
        (2, 2),
        dtype=np.float64,
    )

    predicted = np.array(
        [
            [1.0, -1.0],
            [1.0, -1.0],
        ],
        dtype=np.float64,
    )

    result = elevation_error_metrics(
        reference,
        predicted,
    )

    assert result.metrics.bias == pytest.approx(0.0)
    assert result.metrics.mae == pytest.approx(1.0)
    assert result.metrics.rmse == pytest.approx(1.0)


def test_nan_prediction_reduces_coverage():
    reference = np.ones(
        (3, 3),
        dtype=np.float64,
    )

    predicted = np.ones(
        (3, 3),
        dtype=np.float64,
    )

    predicted[1, 1] = np.nan

    result = elevation_error_metrics(
        reference,
        predicted,
    )

    assert result.coverage == pytest.approx(
        8.0 / 9.0
    )

    assert result.metrics.valid_count == 8


def test_slope_of_flat_plane_is_zero():
    elevation = np.zeros(
        (5, 5),
        dtype=np.float64,
    )

    slope = compute_slope(
        elevation,
        0.10,
    )

    assert np.allclose(
        slope,
        0.0,
    )


def test_slope_of_linear_plane_is_constant():
    resolution = 1.0

    x = np.arange(
        5,
        dtype=np.float64,
    )

    y = np.arange(
        5,
        dtype=np.float64,
    )

    xx, yy = np.meshgrid(
        x,
        y,
    )

    elevation = 0.5 * xx

    slope = compute_slope(
        elevation,
        resolution,
    )

    expected = np.degrees(
        np.arctan(0.5)
    )

    assert np.allclose(
        slope,
        expected,
    )


def test_roughness_of_flat_plane_is_zero():
    elevation = np.full(
        (5, 5),
        2.5,
        dtype=np.float64,
    )

    roughness = compute_roughness(
        elevation,
    )

    assert np.allclose(
        roughness,
        0.0,
    )


def test_roughness_increases_for_local_terrain_change():
    elevation = np.zeros(
        (5, 5),
        dtype=np.float64,
    )

    elevation[2, 2] = 2.0

    roughness = compute_roughness(
        elevation,
    )

    assert roughness[2, 2] > 0.0
    assert np.max(roughness) > 0.0


def test_perfect_terrain_has_zero_all_metrics():
    x = np.arange(
        6,
        dtype=np.float64,
    )

    y = np.arange(
        6,
        dtype=np.float64,
    )

    xx, yy = np.meshgrid(
        x,
        y,
    )

    elevation = (
        0.25 * xx
        + 0.10 * yy
    )

    result = evaluate_terrain_accuracy(
        elevation,
        elevation.copy(),
        resolution_m=1.0,
    )

    assert result.elevation.metrics.mae == pytest.approx(
        0.0
    )

    assert result.elevation.metrics.rmse == pytest.approx(
        0.0
    )

    assert result.slope.mae == pytest.approx(
        0.0
    )

    assert result.slope.rmse == pytest.approx(
        0.0
    )

    assert result.roughness.mae == pytest.approx(
        0.0
    )

    assert result.roughness.rmse == pytest.approx(
        0.0
    )


def test_terrain_elevation_error_is_detected():
    reference = np.zeros(
        (6, 6),
        dtype=np.float64,
    )

    predicted = np.zeros(
        (6, 6),
        dtype=np.float64,
    )

    predicted[2:4, 2:4] = 0.5

    result = evaluate_terrain_accuracy(
        reference,
        predicted,
        resolution_m=0.10,
    )

    assert result.elevation.metrics.mae > 0.0
    assert result.elevation.metrics.rmse > 0.0
    assert result.elevation.metrics.max_abs_error == pytest.approx(
        0.5
    )


def test_terrain_slope_error_is_detected():
    x = np.arange(
        6,
        dtype=np.float64,
    )

    y = np.arange(
        6,
        dtype=np.float64,
    )

    xx, yy = np.meshgrid(
        x,
        y,
    )

    reference = 0.10 * xx

    predicted = 0.30 * xx

    result = evaluate_terrain_accuracy(
        reference,
        predicted,
        resolution_m=1.0,
    )

    assert result.slope.mae > 0.0
    assert result.slope.rmse > 0.0


def test_terrain_roughness_error_is_detected():
    reference = np.zeros(
        (7, 7),
        dtype=np.float64,
    )

    predicted = reference.copy()

    predicted[3, 3] = 1.0

    result = evaluate_terrain_accuracy(
        reference,
        predicted,
        resolution_m=0.20,
    )

    assert result.roughness.mae > 0.0
    assert result.roughness.rmse > 0.0


def test_mismatched_shapes_are_rejected():
    reference = np.zeros(
        (4, 4),
        dtype=np.float64,
    )

    predicted = np.zeros(
        (5, 5),
        dtype=np.float64,
    )

    with pytest.raises(ValueError):
        elevation_error_metrics(
            reference,
            predicted,
        )


def test_invalid_resolution_is_rejected():
    elevation = np.zeros(
        (4, 4),
        dtype=np.float64,
    )

    with pytest.raises(ValueError):
        compute_slope(
            elevation,
            0.0,
        )

    with pytest.raises(ValueError):
        compute_slope(
            elevation,
            -0.1,
        )


def test_tiny_raster_is_rejected_for_slope():
    elevation = np.zeros(
        (1, 4),
        dtype=np.float64,
    )

    with pytest.raises(ValueError):
        compute_slope(
            elevation,
            0.10,
        )


def test_tiny_raster_is_rejected_for_roughness():
    elevation = np.zeros(
        (2, 2),
        dtype=np.float64,
    )

    with pytest.raises(ValueError):
        compute_roughness(
            elevation,
        )


def test_non_2d_input_is_rejected():
    elevation = np.zeros(
        (3, 3, 1),
        dtype=np.float64,
    )

    with pytest.raises(ValueError):
        elevation_error_metrics(
            elevation,
            elevation,
        )


def test_result_can_be_serialized_to_dictionary():
    elevation = np.zeros(
        (5, 5),
        dtype=np.float64,
    )

    result = evaluate_terrain_accuracy(
        elevation,
        elevation.copy(),
        resolution_m=0.10,
    )

    summary = result.as_dict()

    assert summary["resolution_m"] == pytest.approx(
        0.10
    )

    assert summary["elevation"]["mae"] == pytest.approx(
        0.0
    )

    assert summary["slope"]["rmse"] == pytest.approx(
        0.0
    )

    assert summary["roughness"]["rmse"] == pytest.approx(
        0.0
    )