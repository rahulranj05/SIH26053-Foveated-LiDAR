import numpy as np

from mapping.kinematic_foveation import VehicleState
from mapping.sensor_frame import SensorFrame
from mapping.temporal_foveamap_pipeline import TemporalFoveaMapProcessor


def mock_base_signal_provider(frame: SensorFrame) -> dict[str, np.ndarray]:
    """
    Deterministic stand-in for the base signal layer.

    A31 expects caller-supplied base importance signals to be
    normalized to [0, 1].

    Semantic information is intentionally NOT supplied here because
    A30 derives the semantic signal directly from SensorFrame.
    """

    xyz = frame.xyz

    distance = np.linalg.norm(
        xyz[:, :2],
        axis=1,
    )

    # Convert physical distance in metres into normalized
    # distance importance.
    #
    # 0 m   -> 1.0 importance
    # 100 m -> 0.0 importance
    distance_importance = np.clip(
        1.0 - (distance / 100.0),
        0.0,
        1.0,
    )

    return {
        "distance": distance_importance,
    }


def make_mock_frame(
    frame_id: str,
    timestamp: float,
) -> SensorFrame:
    """
    Fake Rahul perception frame.

    The points deliberately span multiple distance bands and
    semantic categories so the real FoveaMap backend has
    meaningful input.
    """

    xyz = np.array(
        [
            [2.0, 0.0, 0.20],
            [5.0, 0.5, 0.30],
            [8.0, 1.0, 0.40],
            [12.0, -1.0, 0.35],
            [18.0, 0.5, 0.50],
            [30.0, 2.0, 0.45],
            [45.0, -2.0, 0.60],
            [70.0, 1.0, 0.55],
        ],
        dtype=np.float64,
    )

    # Canonical FoveaMap semantic classes:
    #
    # UNKNOWN           = 0
    # DRIVABLE          = 1
    # NON_DRIVABLE      = 2
    # STATIC_OBSTACLE   = 3
    # VEHICLE           = 4
    # VULNERABLE_USER   = 5

    semantic_labels = np.array(
        [3, 3, 4, 4, 4, 5, 5, 5],
        dtype=np.int64,
    )

    semantic_confidence = np.array(
        [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92],
        dtype=np.float64,
    )

    vehicle_state = VehicleState(
        speed=8.0,
        yaw_rate=0.05,
        heading=0.0,
    )

    return SensorFrame(
        xyz=xyz,
        semantic_labels=semantic_labels,
        semantic_confidence=semantic_confidence,
        vehicle_state=vehicle_state,
        timestamp=timestamp,
        frame_id=frame_id,
    )


def test_p2_mock_frame_reaches_adaptive_leaf_map():
    """
    Prove that a mock Rahul frame travels through the real
    temporal FoveaMap backend and produces an actual
    hierarchical adaptive leaf map.
    """

    processor = TemporalFoveaMapProcessor(
        base_signal_provider=mock_base_signal_provider,
    )

    frame = make_mock_frame(
        frame_id="p2-frame-000001",
        timestamp=1.0,
    )

    result = processor.process_frame(frame)

    # ------------------------------------------------------------------
    # A31 frame metadata
    # ------------------------------------------------------------------

    assert result.frame_id == "p2-frame-000001"
    assert result.timestamp == 1.0
    assert result.delta_time is None

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    assert "distance" in result.signals

    # A30 should have generated semantic information from SensorFrame.
    assert "semantic" in result.signals

    # ------------------------------------------------------------------
    # A20 result
    # ------------------------------------------------------------------

    pipeline_result = result.pipeline_result

    assert pipeline_result is not None
    assert pipeline_result.foveation is not None
    assert pipeline_result.leaf_map is not None

    # ------------------------------------------------------------------
    # A18.3 hierarchical leaf map
    # ------------------------------------------------------------------

    leaf_map = pipeline_result.leaf_map

    # Every input point must appear exactly once in the final leaf map.
    assert leaf_map.point_count.sum() == frame.num_points

    assert len(leaf_map.levels) > 0
    assert len(leaf_map.resolutions) == len(leaf_map.levels)

    # The mapper is allowed to refine the complete scene to the
    # finest level. What matters here is that it actually produced
    # valid adaptive leaves.
    assert np.all(
        np.isin(
            leaf_map.levels,
            [0, 1, 2, 3],
        )
    )

    assert np.all(
        np.isfinite(
            leaf_map.resolutions
        )
    )

    assert np.all(
        leaf_map.resolutions > 0.0
    )

    # The finest configured resolution must be represented for this
    # high-importance mock scene.
    assert np.any(
        np.isclose(
            leaf_map.resolutions,
            0.05,
        )
    )

    # Every leaf must contain at least one point.
    assert np.all(
        leaf_map.point_count > 0
    )


def test_p2_replay_is_deterministic():
    """
    The same input frame must produce the same adaptive map.
    """

    frame = make_mock_frame(
        frame_id="p2-replay",
        timestamp=10.0,
    )

    processor_a = TemporalFoveaMapProcessor(
        base_signal_provider=mock_base_signal_provider,
    )

    processor_b = TemporalFoveaMapProcessor(
        base_signal_provider=mock_base_signal_provider,
    )

    result_a = processor_a.process_frame(frame)
    result_b = processor_b.process_frame(frame)

    map_a = result_a.pipeline_result.leaf_map
    map_b = result_b.pipeline_result.leaf_map

    # ------------------------------------------------------------------
    # Hierarchy
    # ------------------------------------------------------------------

    np.testing.assert_array_equal(
        map_a.levels,
        map_b.levels,
    )

    np.testing.assert_allclose(
        map_a.resolutions,
        map_b.resolutions,
    )

    # ------------------------------------------------------------------
    # Spatial cells
    # ------------------------------------------------------------------

    np.testing.assert_array_equal(
        map_a.ix,
        map_b.ix,
    )

    np.testing.assert_array_equal(
        map_a.iy,
        map_b.iy,
    )

    # ------------------------------------------------------------------
    # Elevation statistics
    # ------------------------------------------------------------------

    np.testing.assert_allclose(
        map_a.z_min,
        map_b.z_min,
    )

    np.testing.assert_allclose(
        map_a.z_max,
        map_b.z_max,
    )

    np.testing.assert_allclose(
        map_a.z_mean,
        map_b.z_mean,
    )

    # ------------------------------------------------------------------
    # Point conservation
    # ------------------------------------------------------------------

    np.testing.assert_array_equal(
        map_a.point_count,
        map_b.point_count,
    )

    # ------------------------------------------------------------------
    # Dominant foveation reason
    # ------------------------------------------------------------------

    np.testing.assert_array_equal(
        map_a.dominant_reason,
        map_b.dominant_reason,
    )


def test_p2_two_frame_temporal_replay():
    """
    Prove that A31 correctly maintains temporal state across
    consecutive SensorFrames.
    """

    processor = TemporalFoveaMapProcessor(
        base_signal_provider=mock_base_signal_provider,
    )

    frame_1 = make_mock_frame(
        frame_id="p2-temporal-000001",
        timestamp=1.0,
    )

    frame_2 = make_mock_frame(
        frame_id="p2-temporal-000002",
        timestamp=1.1,
    )

    sequence_result = processor.process_sequence(
        [frame_1, frame_2]
    )

    # ------------------------------------------------------------------
    # Sequence metadata
    # ------------------------------------------------------------------

    assert sequence_result.total_frames == 2

    assert sequence_result.frame_ids == (
        "p2-temporal-000001",
        "p2-temporal-000002",
    )

    np.testing.assert_allclose(
        sequence_result.timestamps,
        [1.0, 1.1],
    )

    # ------------------------------------------------------------------
    # Temporal deltas
    # ------------------------------------------------------------------

    assert sequence_result.delta_times[0] is None

    # Floating-point arithmetic can represent 1.1 - 1.0 as
    # 0.10000000000000009, so compare using numerical tolerance.
    assert np.isclose(
        sequence_result.delta_times[1],
        0.1,
    )

    assert np.isclose(
        sequence_result.mean_delta_time,
        0.1,
    )

    assert np.isclose(
        sequence_result.total_duration_seconds,
        0.1,
    )

    # ------------------------------------------------------------------
    # Individual frame results
    # ------------------------------------------------------------------

    first_result = sequence_result.frames[0]
    second_result = sequence_result.frames[1]

    assert first_result.frame_id == "p2-temporal-000001"
    assert second_result.frame_id == "p2-temporal-000002"

    assert first_result.delta_time is None

    assert np.isclose(
        second_result.delta_time,
        0.1,
    )

    # ------------------------------------------------------------------
    # Both frames must produce real adaptive maps.
    # ------------------------------------------------------------------

    first_map = first_result.pipeline_result.leaf_map
    second_map = second_result.pipeline_result.leaf_map

    assert first_map is not None
    assert second_map is not None

    assert first_map.point_count.sum() == frame_1.num_points
    assert second_map.point_count.sum() == frame_2.num_points

    assert len(first_map.levels) > 0
    assert len(second_map.levels) > 0

    # Both frames must retain the canonical signals.
    assert "distance" in first_result.signals
    assert "semantic" in first_result.signals

    assert "distance" in second_result.signals
    assert "semantic" in second_result.signals