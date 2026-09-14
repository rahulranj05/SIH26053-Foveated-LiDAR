import numpy as np

from mapping.kinematic_foveation import VehicleState
from mapping.sensor_frame import SensorFrame


def test_rahul_perception_frame_contract():
    """Verify the canonical Rahul -> FoveaMap frame contract."""

    xyz = np.array(
        [
            [2.0, 0.0, 0.2],
            [8.0, 1.0, 0.4],
            [18.0, -1.0, 0.3],
            [35.0, 2.0, 0.5],
        ],
        dtype=np.float32,
    )

    semantic_label = np.array(
        [1, 4, 5, 3],
        dtype=np.int64,
    )

    semantic_conf = np.array(
        [0.99, 0.94, 0.88, 0.91],
        dtype=np.float32,
    )

    vehicle_state = VehicleState(
        speed=8.0,
        yaw_rate=0.05,
        heading=0.0,
    )

    frame = SensorFrame(
        xyz=xyz,
        semantic_labels=semantic_label,
        semantic_confidence=semantic_conf,
        vehicle_state=vehicle_state,
        timestamp=12.5,
        frame_id="rahul-frame-000001",
    )

    assert frame.num_points == 4
    assert frame.has_semantics
    assert frame.has_semantic_confidence
    assert not frame.has_dynamic_probability

    signals = frame.to_foveamap_signals()

    assert set(signals) == {
        "semantic",
        "semantic_confidence",
    }

    np.testing.assert_array_equal(
        signals["semantic"],
        semantic_label.astype(np.float64),
    )

    np.testing.assert_allclose(
        signals["semantic_confidence"],
        semantic_conf,
    )

    assert frame.frame_id == "rahul-frame-000001"
    assert frame.timestamp == 12.5
    assert frame.vehicle_state.speed == 8.0


def test_semantic_confidence_must_match_point_count():
    """Semantic confidence must contain exactly one value per point."""

    xyz = np.zeros((4, 3))

    vehicle_state = VehicleState(
        speed=0.0,
        yaw_rate=0.0,
        heading=0.0,
    )

    try:
        SensorFrame(
            xyz=xyz,
            semantic_labels=np.array([1, 2, 3, 4]),
            semantic_confidence=np.array([0.9, 0.8, 0.7]),
            vehicle_state=vehicle_state,
            timestamp=0.0,
            frame_id=0,
        )
    except ValueError as exc:
        assert "semantic_confidence length" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError for mismatched semantic confidence"
        )