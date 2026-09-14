"""
Focused tests for production foveation signal construction.
"""

import numpy as np
import pytest

from mapping.kinematic_foveation import VehicleState
from mapping.production_foveation_signals import (
    ProductionFoveationSignalConfig,
    build_production_foveation_signals,
    distance_importance,
    make_production_foveation_provider,
)
from mapping.sensor_frame import SensorFrame


def make_frame(
    xyz: np.ndarray,
    *,
    speed: float = 10.0,
    yaw_rate: float = 0.0,
    heading: float = 0.0,
) -> SensorFrame:
    """Build a minimal canonical SensorFrame for testing."""

    return SensorFrame(
        xyz=xyz,
        vehicle_state=VehicleState(
            speed=speed,
            yaw_rate=yaw_rate,
            heading=heading,
        ),
        timestamp=1.0,
        frame_id="test-frame",
    )


def test_distance_importance_is_normalized_by_distance() -> None:
    xyz = np.array(
        [
            [0.0, 0.0, 0.0],
            [50.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [150.0, 0.0, 0.0],
        ]
    )

    importance = distance_importance(xyz)

    np.testing.assert_allclose(
        importance,
        np.array([1.0, 0.5, 0.0, 0.0]),
    )


def test_distance_importance_uses_horizontal_distance() -> None:
    xyz = np.array(
        [
            [50.0, 0.0, 0.0],
            [0.0, 50.0, 20.0],
        ]
    )

    importance = distance_importance(xyz)

    np.testing.assert_allclose(
        importance,
        np.array([0.5, 0.5]),
    )


def test_distance_importance_rejects_invalid_maximum_distance() -> None:
    xyz = np.array([[1.0, 0.0, 0.0]])

    with pytest.raises(ValueError):
        distance_importance(xyz, maximum_distance=0.0)

    with pytest.raises(ValueError):
        distance_importance(xyz, maximum_distance=-1.0)

    with pytest.raises(ValueError):
        distance_importance(xyz, maximum_distance=np.inf)


def test_production_builder_returns_exact_canonical_signals() -> None:
    xyz = np.array(
        [
            [5.0, 0.0, 0.0],
            [10.0, 1.0, 0.0],
            [20.0, -2.0, 0.5],
            [40.0, 3.0, 1.0],
        ]
    )

    frame = make_frame(xyz)

    signals = build_production_foveation_signals(frame)

    assert set(signals) == {
        "DISTANCE",
        "KINEMATIC",
        "PREDICTED_PATH",
    }

    for name, signal in signals.items():
        assert signal.shape == (len(xyz),), name
        assert np.all(np.isfinite(signal)), name
        assert np.all(signal >= 0.0), name
        assert np.all(signal <= 1.0), name


def test_production_builder_preserves_point_count() -> None:
    rng = np.random.default_rng(42)
    xyz = rng.normal(size=(100, 3))

    frame = make_frame(xyz)

    signals = build_production_foveation_signals(frame)

    for signal in signals.values():
        assert len(signal) == len(xyz)


def test_production_builder_uses_config() -> None:
    xyz = np.array(
        [
            [25.0, 0.0, 0.0],
            [50.0, 0.0, 0.0],
        ]
    )

    frame = make_frame(xyz)

    config = ProductionFoveationSignalConfig(
        maximum_distance=50.0,
    )

    signals = build_production_foveation_signals(
        frame,
        config=config,
    )

    np.testing.assert_allclose(
        signals["DISTANCE"],
        np.array([0.5, 0.0]),
    )


def test_production_builder_rejects_non_sensor_frame() -> None:
    with pytest.raises(TypeError):
        build_production_foveation_signals(
            np.zeros((3, 3)),  # type: ignore[arg-type]
        )


def test_production_provider_matches_direct_builder() -> None:
    xyz = np.array(
        [
            [5.0, 0.0, 0.0],
            [15.0, 1.0, 0.0],
            [30.0, -1.0, 0.5],
        ]
    )

    frame = make_frame(
        xyz,
        speed=12.0,
        yaw_rate=0.1,
    )

    config = ProductionFoveationSignalConfig()

    direct = build_production_foveation_signals(
        frame,
        config=config,
    )

    provider = make_production_foveation_provider(
        config=config,
    )

    provided = provider(frame)

    assert set(provided) == set(direct)

    for name in direct:
        np.testing.assert_allclose(
            provided[name],
            direct[name],
        )


def test_production_signals_do_not_fuse_or_choose_resolution() -> None:
    xyz = np.array(
        [
            [1.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [50.0, 0.0, 0.0],
        ]
    )

    frame = make_frame(xyz)

    signals = build_production_foveation_signals(frame)

    assert set(signals) == {
        "DISTANCE",
        "KINEMATIC",
        "PREDICTED_PATH",
    }

    assert not isinstance(signals, np.ndarray)