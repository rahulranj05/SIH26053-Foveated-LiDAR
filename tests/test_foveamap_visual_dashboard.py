"""
Tests for A39.2 FoveaMap Visual Dashboard.
"""

from __future__ import annotations

import numpy as np
import pytest

from mapping.foveamap_dashboard import DashboardFrame
from mapping.foveamap_visual_dashboard import (
    FoveaMapVisualDashboard,
    VisualDashboardConfig,
    VisualDashboardSnapshot,
    create_visual_dashboard,
)


def make_frame(
    *,
    frame_id: int = 1,
    safety_mode: str = "NORMAL",
    latency_ms: float = 50.0,
) -> DashboardFrame:
    return DashboardFrame(
        frame_id=frame_id,
        timestamp=float(frame_id),
        point_count=100,
        valid_point_count=90,
        invalid_point_count=10,
        safety_mode=safety_mode,
        safety_reasons=(
            ("LOW_POINT_COUNT",)
            if safety_mode != "NORMAL"
            else ()
        ),
        semantic_available=True,
        dynamic_available=True,
        leaf_count=20,
        resolution_counts={
            0.05: 10,
            0.10: 5,
            0.20: 5,
        },
        dominant_reason_counts={
            "DISTANCE": 50,
            "SEMANTIC": 30,
            "DYNAMIC": 20,
        },
        processing_latency_ms=latency_ms,
        fps=(
            1000.0 / latency_ms
            if latency_ms > 0.0
            else float("inf")
        ),
        preview_xyz=np.array(
            [
                [1.0, 0.0, 0.1],
                [2.0, 1.0, 0.2],
                [3.0, -1.0, 0.3],
            ],
            dtype=np.float64,
        ),
    )


def test_default_config():
    config = VisualDashboardConfig()

    assert config.point_size > 0.0
    assert config.history_size >= 1


@pytest.mark.parametrize(
    "point_size",
    [
        0.0,
        -1.0,
        float("nan"),
        float("inf"),
    ],
)
def test_invalid_point_size(point_size):
    with pytest.raises(ValueError):
        VisualDashboardConfig(
            point_size=point_size
        )


@pytest.mark.parametrize(
    "history_size",
    [
        0,
        -1,
    ],
)
def test_invalid_history_size(history_size):
    with pytest.raises(ValueError):
        VisualDashboardConfig(
            history_size=history_size
        )


def test_invalid_history_size_type():
    with pytest.raises(ValueError):
        VisualDashboardConfig(
            history_size=1.5
        )


def test_invalid_elevation_bounds():
    with pytest.raises(ValueError):
        VisualDashboardConfig(
            elevation_min=1.0,
            elevation_max=1.0,
        )


def test_dashboard_initial_state():
    dashboard = FoveaMapVisualDashboard()

    assert dashboard.frame_count == 0
    assert dashboard.latest_frame is None
    assert dashboard.snapshot() is None


def test_create_visual_dashboard():
    dashboard = create_visual_dashboard()

    assert isinstance(
        dashboard,
        FoveaMapVisualDashboard,
    )


def test_update_returns_snapshot():
    dashboard = FoveaMapVisualDashboard()

    frame = make_frame()

    snapshot = dashboard.update(
        frame
    )

    assert isinstance(
        snapshot,
        VisualDashboardSnapshot,
    )

    assert snapshot.frame_id == 1
    assert snapshot.safety_mode == "NORMAL"
    assert snapshot.point_count == 100
    assert snapshot.leaf_count == 20
    assert snapshot.latency_ms == 50.0
    assert snapshot.fps == 20.0


def test_latest_frame_after_update():
    dashboard = FoveaMapVisualDashboard()

    frame = make_frame(
        frame_id=7
    )

    dashboard.update(frame)

    assert dashboard.frame_count == 1
    assert dashboard.latest_frame is frame


def test_snapshot_after_update():
    dashboard = FoveaMapVisualDashboard()

    dashboard.update(
        make_frame(
            frame_id=8,
            safety_mode="DEGRADED",
        )
    )

    snapshot = dashboard.snapshot()

    assert snapshot is not None
    assert snapshot.frame_id == 8
    assert snapshot.safety_mode == "DEGRADED"


def test_history_limit():
    dashboard = FoveaMapVisualDashboard(
        config=VisualDashboardConfig(
            history_size=2
        )
    )

    dashboard.update(
        make_frame(frame_id=1)
    )

    dashboard.update(
        make_frame(frame_id=2)
    )

    dashboard.update(
        make_frame(frame_id=3)
    )

    assert dashboard.frame_count == 2
    assert dashboard.latest_frame.frame_id == 3


def test_clear():
    dashboard = FoveaMapVisualDashboard()

    dashboard.update(
        make_frame()
    )

    dashboard.clear()

    assert dashboard.frame_count == 0
    assert dashboard.latest_frame is None


def test_update_rejects_wrong_type():
    dashboard = FoveaMapVisualDashboard()

    with pytest.raises(TypeError):
        dashboard.update(
            "not a frame"
        )


def test_render_requires_frame():
    dashboard = FoveaMapVisualDashboard()

    with pytest.raises(RuntimeError):
        dashboard.render()


def test_render_returns_figure():
    pytest.importorskip(
        "matplotlib"
    )

    dashboard = FoveaMapVisualDashboard()

    figure = dashboard.render(
        make_frame()
    )

    assert figure is not None
    assert dashboard.frame_count == 1


def test_render_existing_frame():
    pytest.importorskip(
        "matplotlib"
    )

    dashboard = FoveaMapVisualDashboard()

    dashboard.update(
        make_frame()
    )

    figure = dashboard.render()

    assert figure is not None


def test_save_dashboard(tmp_path):
    pytest.importorskip(
        "matplotlib"
    )

    dashboard = FoveaMapVisualDashboard()

    path = tmp_path / "dashboard.png"

    result = dashboard.save(
        str(path),
        make_frame(),
    )

    assert result == str(path)
    assert path.exists()


def test_invalid_save_dpi():
    pytest.importorskip(
        "matplotlib"
    )

    dashboard = FoveaMapVisualDashboard()

    dashboard.update(
        make_frame()
    )

    with pytest.raises(ValueError):
        dashboard.save(
            "dashboard.png",
            dpi=0,
        )