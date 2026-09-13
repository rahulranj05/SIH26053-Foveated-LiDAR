"""
Tests for A39 FoveaMap dashboard data layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from mapping.foveamap_dashboard import (
    DashboardConfig,
    DashboardFrame,
    build_dashboard_frame,
)


@dataclass
class FakeSafetyAssessment:
    mode: str
    reasons: tuple[str, ...]
    semantic_available: bool = False
    dynamic_available: bool = False


@dataclass
class FakeLeafMap:
    levels: dict[float, list[object]]


def make_leaf_map() -> FakeLeafMap:
    return FakeLeafMap(
        levels={
            0.05: [1, 2, 3],
            0.10: [1, 2],
            0.20: [1],
            0.40: [1, 2, 3, 4],
        }
    )


def make_foveation() -> dict[str, np.ndarray]:
    return {
        "importance": np.array(
            [0.1, 0.5, 1.0, 0.7],
            dtype=np.float64,
        ),
        "resolution": np.array(
            [0.40, 0.20, 0.05, 0.10],
            dtype=np.float64,
        ),
        "dominant_reason": np.array(
            [
                "DISTANCE",
                "SEMANTIC",
                "DYNAMIC",
                "SEMANTIC",
            ],
            dtype=object,
        ),
    }


def test_dashboard_config_defaults() -> None:
    config = DashboardConfig()

    assert config.max_points_for_preview == 50_000
    assert config.max_preview_range_m == 100.0


def test_dashboard_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        DashboardConfig(
            max_points_for_preview=0
        )

    with pytest.raises(ValueError):
        DashboardConfig(
            max_preview_range_m=0.0
        )


def test_dashboard_frame_properties() -> None:
    frame = DashboardFrame(
        frame_id=1,
        timestamp=1.0,
        point_count=100,
        valid_point_count=90,
        invalid_point_count=10,
        safety_mode="DEGRADED",
        safety_reasons=("LOW_POINT_COUNT",),
        semantic_available=True,
        dynamic_available=True,
        leaf_count=10,
        resolution_counts={0.05: 5},
        dominant_reason_counts={"SEMANTIC": 5},
        processing_latency_ms=50.0,
        fps=20.0,
        preview_xyz=np.zeros((10, 3)),
    )

    assert frame.valid_fraction == 0.9
    assert frame.invalid_fraction == 0.1
    assert frame.is_safe
    assert frame.is_degraded
    assert not frame.is_safety_mode


def test_safety_mode_properties() -> None:
    frame = DashboardFrame(
        frame_id=1,
        timestamp=1.0,
        point_count=100,
        valid_point_count=100,
        invalid_point_count=0,
        safety_mode="SAFETY",
        safety_reasons=("HIGH_LATENCY",),
        semantic_available=False,
        dynamic_available=False,
        leaf_count=2,
        resolution_counts={0.20: 2},
        dominant_reason_counts={},
        processing_latency_ms=300.0,
        fps=3.333,
        preview_xyz=np.zeros((2, 3)),
    )

    assert not frame.is_safe
    assert not frame.is_degraded
    assert frame.is_safety_mode


def test_build_dashboard_frame_normal() -> None:
    xyz = np.array(
        [
            [2.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    assessment = FakeSafetyAssessment(
        mode="NORMAL",
        reasons=(),
        semantic_available=True,
        dynamic_available=True,
    )

    frame = build_dashboard_frame(
        frame_id="000001",
        timestamp=1.5,
        xyz=xyz,
        foveation=make_foveation(),
        leaf_map=make_leaf_map(),
        safety_assessment=assessment,
        latency_ms=50.0,
    )

    assert frame.frame_id == "000001"
    assert frame.timestamp == 1.5
    assert frame.point_count == 4
    assert frame.valid_point_count == 4
    assert frame.invalid_point_count == 0

    assert frame.safety_mode == "NORMAL"
    assert frame.semantic_available
    assert frame.dynamic_available

    assert frame.leaf_count == 10

    assert frame.resolution_counts == {
        0.05: 3,
        0.10: 2,
        0.20: 1,
        0.40: 4,
    }

    assert frame.dominant_reason_counts == {
        "DISTANCE": 1,
        "DYNAMIC": 1,
        "SEMANTIC": 2,
    }

    assert frame.processing_latency_ms == 50.0
    assert frame.fps == 20.0

    assert frame.preview_xyz.shape == (4, 3)


def test_invalid_points_are_counted() -> None:
    xyz = np.array(
        [
            [1.0, 0.0, 0.0],
            [np.nan, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [np.inf, 0.0, 0.0],
        ]
    )

    assessment = FakeSafetyAssessment(
        mode="DEGRADED",
        reasons=("HIGH_INVALID_POINT_FRACTION",),
    )

    frame = build_dashboard_frame(
        frame_id=2,
        timestamp=2.0,
        xyz=xyz,
        foveation=None,
        leaf_map=None,
        safety_assessment=assessment,
        latency_ms=100.0,
    )

    assert frame.point_count == 4
    assert frame.valid_point_count == 2
    assert frame.invalid_point_count == 2
    assert frame.invalid_fraction == 0.5
    assert frame.valid_fraction == 0.5


def test_preview_filters_nonfinite_points() -> None:
    xyz = np.array(
        [
            [1.0, 0.0, 0.0],
            [np.nan, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [np.inf, 0.0, 0.0],
        ]
    )

    assessment = FakeSafetyAssessment(
        mode="NORMAL",
        reasons=(),
    )

    frame = build_dashboard_frame(
        frame_id=3,
        timestamp=3.0,
        xyz=xyz,
        foveation=None,
        leaf_map=None,
        safety_assessment=assessment,
        latency_ms=20.0,
    )

    assert frame.preview_xyz.shape == (2, 3)


def test_preview_respects_range() -> None:
    xyz = np.array(
        [
            [5.0, 0.0, 0.0],
            [50.0, 0.0, 0.0],
            [101.0, 0.0, 0.0],
        ]
    )

    assessment = FakeSafetyAssessment(
        mode="NORMAL",
        reasons=(),
    )

    frame = build_dashboard_frame(
        frame_id=4,
        timestamp=4.0,
        xyz=xyz,
        foveation=None,
        leaf_map=None,
        safety_assessment=assessment,
        latency_ms=20.0,
    )

    assert frame.preview_xyz.shape == (2, 3)
    assert np.all(
        np.linalg.norm(
            frame.preview_xyz,
            axis=1,
        )
        <= 100.0
    )


def test_preview_is_capped() -> None:
    xyz = np.column_stack(
        (
            np.linspace(1.0, 50.0, 1000),
            np.zeros(1000),
            np.zeros(1000),
        )
    )

    assessment = FakeSafetyAssessment(
        mode="NORMAL",
        reasons=(),
    )

    frame = build_dashboard_frame(
        frame_id=5,
        timestamp=5.0,
        xyz=xyz,
        foveation=None,
        leaf_map=None,
        safety_assessment=assessment,
        latency_ms=20.0,
        config=DashboardConfig(
            max_points_for_preview=100
        ),
    )

    assert frame.preview_xyz.shape == (100, 3)


def test_preview_is_read_only() -> None:
    xyz = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )

    assessment = FakeSafetyAssessment(
        mode="NORMAL",
        reasons=(),
    )

    frame = build_dashboard_frame(
        frame_id=6,
        timestamp=6.0,
        xyz=xyz,
        foveation=None,
        leaf_map=None,
        safety_assessment=assessment,
        latency_ms=20.0,
    )

    assert not frame.preview_xyz.flags.writeable


def test_zero_latency_produces_infinite_fps() -> None:
    xyz = np.array(
        [[1.0, 0.0, 0.0]]
    )

    assessment = FakeSafetyAssessment(
        mode="NORMAL",
        reasons=(),
    )

    frame = build_dashboard_frame(
        frame_id=7,
        timestamp=7.0,
        xyz=xyz,
        foveation=None,
        leaf_map=None,
        safety_assessment=assessment,
        latency_ms=0.0,
    )

    assert np.isinf(frame.fps)


def test_invalid_safety_mode_rejected() -> None:
    xyz = np.array(
        [[1.0, 0.0, 0.0]]
    )

    assessment = FakeSafetyAssessment(
        mode="INVALID",
        reasons=(),
    )

    with pytest.raises(ValueError):
        build_dashboard_frame(
            frame_id=8,
            timestamp=8.0,
            xyz=xyz,
            foveation=None,
            leaf_map=None,
            safety_assessment=assessment,
            latency_ms=10.0,
        )


def test_missing_safety_assessment_rejected() -> None:
    xyz = np.array(
        [[1.0, 0.0, 0.0]]
    )

    with pytest.raises(ValueError):
        build_dashboard_frame(
            frame_id=9,
            timestamp=9.0,
            xyz=xyz,
            foveation=None,
            leaf_map=None,
            safety_assessment=None,
            latency_ms=10.0,
        )


def test_invalid_latency_rejected() -> None:
    xyz = np.array(
        [[1.0, 0.0, 0.0]]
    )

    assessment = FakeSafetyAssessment(
        mode="NORMAL",
        reasons=(),
    )

    with pytest.raises(ValueError):
        build_dashboard_frame(
            frame_id=10,
            timestamp=10.0,
            xyz=xyz,
            foveation=None,
            leaf_map=None,
            safety_assessment=assessment,
            latency_ms=-1.0,
        )


def test_invalid_timestamp_rejected() -> None:
    xyz = np.array(
        [[1.0, 0.0, 0.0]]
    )

    assessment = FakeSafetyAssessment(
        mode="NORMAL",
        reasons=(),
    )

    with pytest.raises(ValueError):
        build_dashboard_frame(
            frame_id=11,
            timestamp=np.nan,
            xyz=xyz,
            foveation=None,
            leaf_map=None,
            safety_assessment=assessment,
            latency_ms=10.0,
        )


def test_empty_cloud() -> None:
    xyz = np.empty(
        (0, 3),
        dtype=np.float64,
    )

    assessment = FakeSafetyAssessment(
        mode="SAFETY",
        reasons=("LOW_POINT_COUNT",),
    )

    frame = build_dashboard_frame(
        frame_id=12,
        timestamp=12.0,
        xyz=xyz,
        foveation=None,
        leaf_map=None,
        safety_assessment=assessment,
        latency_ms=100.0,
    )

    assert frame.point_count == 0
    assert frame.valid_point_count == 0
    assert frame.invalid_point_count == 0
    assert frame.valid_fraction == 0.0
    assert frame.invalid_fraction == 1.0
    assert frame.preview_xyz.shape == (0, 3)


def test_dashboard_preserves_frame_identity() -> None:
    frame_id = ("sequence", 12)

    xyz = np.array(
        [[1.0, 2.0, 3.0]]
    )

    assessment = FakeSafetyAssessment(
        mode="NORMAL",
        reasons=(),
    )

    frame = build_dashboard_frame(
        frame_id=frame_id,
        timestamp=12.5,
        xyz=xyz,
        foveation=None,
        leaf_map=None,
        safety_assessment=assessment,
        latency_ms=25.0,
    )

    assert frame.frame_id == frame_id
    assert frame.timestamp == 12.5