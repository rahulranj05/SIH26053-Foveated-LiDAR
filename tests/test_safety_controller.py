import numpy as np
import pytest

from mapping.safety_controller import (
    SAFETY_MODES,
    SafetyAssessment,
    SafetyConfig,
    SafetyPolicy,
    apply_safety_policy,
    assess_safety,
    safety_policy,
    validate_safety_reasons,
)


def make_points(count: int) -> np.ndarray:
    return np.column_stack(
        (
            np.arange(count, dtype=float),
            np.zeros(count),
            np.zeros(count),
        )
    )


def test_safety_modes_are_canonical():
    assert SAFETY_MODES == (
        "NORMAL",
        "DEGRADED",
        "SAFETY",
    )


def test_normal_mode_with_healthy_input():
    points = make_points(10_000)

    assessment = assess_safety(
        points,
        latency_ms=20.0,
        frame_gap_s=0.05,
    )

    assert assessment.mode == "NORMAL"
    assert assessment.reasons == ()
    assert assessment.point_count == 10_000
    assert assessment.invalid_fraction == 0.0
    assert assessment.is_safe is True
    assert assessment.is_degraded is False
    assert assessment.is_safety_mode is False


def test_degraded_mode_for_low_point_count():
    points = make_points(1_000)

    config = SafetyConfig(
        min_points_degraded=2_000,
        min_points_safety=500,
    )

    assessment = assess_safety(
        points,
        config=config,
    )

    assert assessment.mode == "DEGRADED"
    assert "LOW_POINT_COUNT" in assessment.reasons


def test_safety_mode_for_very_low_point_count():
    points = make_points(400)

    config = SafetyConfig(
        min_points_degraded=2_000,
        min_points_safety=500,
    )

    assessment = assess_safety(
        points,
        config=config,
    )

    assert assessment.mode == "SAFETY"
    assert "LOW_POINT_COUNT" in assessment.reasons


def test_empty_cloud_enters_safety():
    points = np.empty((0, 3), dtype=float)

    assessment = assess_safety(points)

    assert assessment.mode == "SAFETY"
    assert assessment.point_count == 0
    assert assessment.invalid_fraction == 1.0
    assert "LOW_POINT_COUNT" in assessment.reasons
    assert "HIGH_INVALID_POINT_FRACTION" in assessment.reasons


def test_invalid_fraction_causes_degraded_mode():
    points = make_points(100)
    points[0] = np.array([np.nan, 0.0, 0.0])

    config = SafetyConfig(
        min_points_degraded=50,
        min_points_safety=10,
        max_invalid_fraction_degraded=0.005,
        max_invalid_fraction_safety=0.25,
    )

    assessment = assess_safety(
        points,
        config=config,
    )

    assert assessment.mode == "DEGRADED"
    assert "HIGH_INVALID_POINT_FRACTION" in assessment.reasons


def test_invalid_fraction_causes_safety_mode():
    points = make_points(100)
    points[:30] = np.nan

    config = SafetyConfig(
        min_points_degraded=50,
        min_points_safety=10,
        max_invalid_fraction_degraded=0.05,
        max_invalid_fraction_safety=0.20,
    )

    assessment = assess_safety(
        points,
        config=config,
    )

    assert assessment.mode == "SAFETY"
    assert "HIGH_INVALID_POINT_FRACTION" in assessment.reasons


def test_high_latency_causes_degraded_mode():
    points = make_points(10_000)

    assessment = assess_safety(
        points,
        latency_ms=150.0,
    )

    assert assessment.mode == "DEGRADED"
    assert "HIGH_LATENCY" in assessment.reasons


def test_extreme_latency_causes_safety_mode():
    points = make_points(10_000)

    assessment = assess_safety(
        points,
        latency_ms=300.0,
    )

    assert assessment.mode == "SAFETY"
    assert "HIGH_LATENCY" in assessment.reasons


def test_stale_frame_causes_degraded_mode():
    points = make_points(10_000)

    assessment = assess_safety(
        points,
        frame_gap_s=0.30,
    )

    assert assessment.mode == "DEGRADED"
    assert "STALE_FRAME" in assessment.reasons


def test_very_stale_frame_causes_safety_mode():
    points = make_points(10_000)

    assessment = assess_safety(
        points,
        frame_gap_s=0.75,
    )

    assert assessment.mode == "SAFETY"
    assert "STALE_FRAME" in assessment.reasons


def test_multiple_failures_are_reported():
    points = make_points(400)

    assessment = assess_safety(
        points,
        latency_ms=300.0,
        frame_gap_s=0.75,
    )

    assert assessment.mode == "SAFETY"
    assert "LOW_POINT_COUNT" in assessment.reasons
    assert "HIGH_LATENCY" in assessment.reasons
    assert "STALE_FRAME" in assessment.reasons


def test_semantic_and_dynamic_flags_are_preserved():
    points = make_points(10_000)

    assessment = assess_safety(
        points,
        semantic_available=True,
        dynamic_available=True,
    )

    assert assessment.semantic_available is True
    assert assessment.dynamic_available is True


def test_default_signal_flags_are_false():
    points = make_points(10_000)

    assessment = assess_safety(points)

    assert assessment.semantic_available is False
    assert assessment.dynamic_available is False


def test_normal_policy():
    points = make_points(10_000)

    assessment = assess_safety(points)
    policy = safety_policy(assessment)

    assert isinstance(policy, SafetyPolicy)
    assert policy.mode == "NORMAL"
    assert policy.max_range_m is None
    assert policy.minimum_resolution_m is None
    assert policy.allow_full_foveation is True
    assert policy.near_field_priority is False


def test_degraded_policy():
    points = make_points(1_000)

    config = SafetyConfig(
        min_points_degraded=2_000,
        min_points_safety=500,
    )

    assessment = assess_safety(
        points,
        config=config,
    )

    policy = safety_policy(
        assessment,
        config=config,
    )

    assert policy.mode == "DEGRADED"
    assert policy.max_range_m == config.degraded_max_range_m
    assert (
        policy.minimum_resolution_m
        == config.degraded_min_resolution_m
    )
    assert policy.allow_full_foveation is True
    assert policy.near_field_priority is True


def test_safety_policy():
    points = make_points(400)

    config = SafetyConfig(
        min_points_degraded=2_000,
        min_points_safety=500,
    )

    assessment = assess_safety(
        points,
        config=config,
    )

    policy = safety_policy(
        assessment,
        config=config,
    )

    assert policy.mode == "SAFETY"
    assert policy.max_range_m == config.safety_max_range_m
    assert (
        policy.minimum_resolution_m
        == config.safety_min_resolution_m
    )
    assert policy.allow_full_foveation is False
    assert policy.near_field_priority is True


def test_normal_policy_does_not_filter_points():
    points = np.array(
        [
            [1.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
        ]
    )

    config = SafetyConfig(
        min_points_degraded=1,
        min_points_safety=0,
    )

    assessment = assess_safety(
        points,
        config=config,
    )

    assert assessment.mode == "NORMAL"

    result = apply_safety_policy(
        points,
        assessment,
        config=config,
    )

    assert np.array_equal(result, points)
    assert result is not points


def test_degraded_policy_limits_range():
    points = np.array(
        [
            [10.0, 0.0, 0.0],
            [25.0, 0.0, 0.0],
            [50.0, 0.0, 0.0],
            [75.0, 0.0, 0.0],
        ]
    )

    config = SafetyConfig(
        min_points_degraded=6,
        min_points_safety=0,
    )

    assessment = assess_safety(
        points,
        config=config,
    )

    assert assessment.mode == "DEGRADED"

    result = apply_safety_policy(
        points,
        assessment,
        config=config,
    )

    assert result.shape == (3, 3)
    assert np.allclose(
        result[:, 0],
        [10.0, 25.0, 50.0],
    )


def test_safety_policy_limits_range():
    points = np.array(
        [
            [2.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ]
    )

    config = SafetyConfig(
        min_points_degraded=10,
        min_points_safety=5,
    )

    assessment = assess_safety(
        points,
        config=config,
    )

    assert assessment.mode == "SAFETY"

    result = apply_safety_policy(
        points,
        assessment,
        config=config,
    )

    assert result.shape == (3, 3)
    assert np.allclose(
        result[:, 0],
        [2.0, 5.0, 10.0],
    )


def test_nonfinite_points_are_removed_when_policy_has_range():
    points = np.array(
        [
            [10.0, 0.0, 0.0],
            [np.nan, 0.0, 0.0],
            [np.inf, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ]
    )

    assessment = SafetyAssessment(
        mode="DEGRADED",
        reasons=("TEST",),
        point_count=4,
        invalid_fraction=0.5,
        latency_ms=0.0,
        frame_gap_s=None,
        semantic_available=False,
        dynamic_available=False,
    )

    config = SafetyConfig(
        degraded_max_range_m=25.0,
    )

    result = apply_safety_policy(
        points,
        assessment,
        config=config,
    )

    assert result.shape == (1, 3)
    assert np.allclose(
        result[0],
        [10.0, 0.0, 0.0],
    )


def test_invalid_xyz_shape_is_rejected():
    with pytest.raises(ValueError):
        assess_safety(
            np.zeros((10, 2)),
        )


def test_invalid_latency_is_rejected():
    points = make_points(10)

    with pytest.raises(ValueError):
        assess_safety(
            points,
            latency_ms=-1.0,
        )

    with pytest.raises(ValueError):
        assess_safety(
            points,
            latency_ms=np.inf,
        )


def test_invalid_frame_gap_is_rejected():
    points = make_points(10)

    with pytest.raises(ValueError):
        assess_safety(
            points,
            frame_gap_s=-1.0,
        )

    with pytest.raises(ValueError):
        assess_safety(
            points,
            frame_gap_s=np.inf,
        )


def test_invalid_mode_is_rejected():
    assessment = SafetyAssessment(
        mode="INVALID",
        reasons=(),
        point_count=10,
        invalid_fraction=0.0,
        latency_ms=0.0,
        frame_gap_s=None,
        semantic_available=False,
        dynamic_available=False,
    )

    with pytest.raises(ValueError):
        safety_policy(assessment)


def test_reason_validation():
    reasons = validate_safety_reasons(
        ["LOW_POINT_COUNT", "HIGH_LATENCY"]
    )

    assert reasons == (
        "LOW_POINT_COUNT",
        "HIGH_LATENCY",
    )


def test_empty_reason_is_rejected():
    with pytest.raises(ValueError):
        validate_safety_reasons(
            ["LOW_POINT_COUNT", ""]
        )


def test_custom_thresholds_are_respected():
    points = make_points(100)

    config = SafetyConfig(
        min_points_degraded=200,
        min_points_safety=50,
        max_latency_ms_degraded=10.0,
        max_latency_ms_safety=20.0,
    )

    assessment = assess_safety(
        points,
        latency_ms=15.0,
        config=config,
    )

    assert assessment.mode == "DEGRADED"
    assert "LOW_POINT_COUNT" in assessment.reasons
    assert "HIGH_LATENCY" in assessment.reasons


def test_safety_thresholds_are_stricter_than_degraded():
    config = SafetyConfig()

    assert (
        config.min_points_safety
        <= config.min_points_degraded
    )

    assert (
        config.max_invalid_fraction_safety
        >= config.max_invalid_fraction_degraded
    )

    assert (
        config.max_latency_ms_safety
        >= config.max_latency_ms_degraded
    )

    assert (
        config.max_frame_gap_s_safety
        >= config.max_frame_gap_s_degraded
    )

    assert (
        config.safety_max_range_m
        <= config.degraded_max_range_m
    )