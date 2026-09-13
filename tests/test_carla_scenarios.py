import numpy as np
import pytest

from mapping.carla_scenarios import (
    ScenarioFrame,
    ScenarioObject,
    ScenarioObjectClass,
    ScenarioSequence,
    build_linear_motion_sequence,
    create_moving_object,
    create_static_object,
    scenario_frame_to_ground_truth,
)


def make_static():
    return create_static_object(
        object_id="building",
        position=[10.0, 5.0, 0.0],
    )


def make_vehicle():
    return create_moving_object(
        object_id="vehicle_1",
        position=[0.0, 0.0, 0.0],
        velocity=[5.0, 0.0, 0.0],
        object_class=ScenarioObjectClass.VEHICLE,
    )


def make_pedestrian():
    return create_moving_object(
        object_id="pedestrian_1",
        position=[2.0, 10.0, 0.0],
        velocity=[0.0, -1.5, 0.0],
        object_class=ScenarioObjectClass.PEDESTRIAN,
    )


def test_static_object_creation():
    obj = make_static()

    assert obj.object_id == "building"
    assert obj.object_class == ScenarioObjectClass.STATIC
    assert obj.dynamic is False
    assert np.array_equal(obj.position, [10.0, 5.0, 0.0])
    assert np.array_equal(obj.velocity, [0.0, 0.0, 0.0])


def test_moving_vehicle_creation():
    obj = make_vehicle()

    assert obj.object_id == "vehicle_1"
    assert obj.object_class == ScenarioObjectClass.VEHICLE
    assert obj.dynamic is True
    assert np.array_equal(obj.position, [0.0, 0.0, 0.0])
    assert np.array_equal(obj.velocity, [5.0, 0.0, 0.0])


def test_moving_pedestrian_creation():
    obj = make_pedestrian()

    assert obj.object_class == ScenarioObjectClass.PEDESTRIAN
    assert obj.dynamic is True
    assert np.allclose(obj.velocity, [0.0, -1.5, 0.0])


def test_object_arrays_are_read_only():
    obj = make_vehicle()

    with pytest.raises(ValueError):
        obj.position[0] = 99.0

    with pytest.raises(ValueError):
        obj.velocity[0] = 99.0


@pytest.mark.parametrize(
    "position",
    [
        [1.0, 2.0],
        [1.0, 2.0, 3.0, 4.0],
        [1.0],
    ],
)
def test_invalid_position_shape(position):
    with pytest.raises(ValueError):
        ScenarioObject(
            object_id="x",
            object_class=ScenarioObjectClass.STATIC,
            position=position,
            velocity=[0.0, 0.0, 0.0],
            dynamic=False,
        )


@pytest.mark.parametrize(
    "velocity",
    [
        [1.0, 2.0],
        [1.0, 2.0, 3.0, 4.0],
        [1.0],
    ],
)
def test_invalid_velocity_shape(velocity):
    with pytest.raises(ValueError):
        ScenarioObject(
            object_id="x",
            object_class=ScenarioObjectClass.STATIC,
            position=[0.0, 0.0, 0.0],
            velocity=velocity,
            dynamic=False,
        )


def test_dynamic_object_requires_nonzero_velocity():
    with pytest.raises(ValueError):
        ScenarioObject(
            object_id="x",
            object_class=ScenarioObjectClass.VEHICLE,
            position=[0.0, 0.0, 0.0],
            velocity=[0.0, 0.0, 0.0],
            dynamic=True,
        )


def test_static_object_requires_zero_velocity():
    with pytest.raises(ValueError):
        ScenarioObject(
            object_id="x",
            object_class=ScenarioObjectClass.STATIC,
            position=[0.0, 0.0, 0.0],
            velocity=[1.0, 0.0, 0.0],
            dynamic=False,
        )


def test_nonfinite_position_rejected():
    with pytest.raises(ValueError):
        ScenarioObject(
            object_id="x",
            object_class=ScenarioObjectClass.STATIC,
            position=[np.nan, 0.0, 0.0],
            velocity=[0.0, 0.0, 0.0],
            dynamic=False,
        )


def test_nonfinite_velocity_rejected():
    with pytest.raises(ValueError):
        ScenarioObject(
            object_id="x",
            object_class=ScenarioObjectClass.VEHICLE,
            position=[0.0, 0.0, 0.0],
            velocity=[np.inf, 0.0, 0.0],
            dynamic=True,
        )


def test_frame_creation_and_filters():
    static = make_static()
    vehicle = make_vehicle()
    pedestrian = make_pedestrian()

    frame = ScenarioFrame(
        frame_id=7,
        timestamp=1.5,
        objects=(static, vehicle, pedestrian),
    )

    assert frame.frame_id == 7
    assert frame.timestamp == 1.5
    assert frame.object_count == 3
    assert len(frame.static_objects) == 1
    assert len(frame.dynamic_objects) == 2


def test_frame_object_ids_must_be_unique():
    obj1 = make_static()
    obj2 = create_static_object(
        object_id="building",
        position=[20.0, 0.0, 0.0],
    )

    with pytest.raises(ValueError):
        ScenarioFrame(
            frame_id=0,
            timestamp=0.0,
            objects=(obj1, obj2),
        )


def test_frame_rejects_invalid_timestamp():
    with pytest.raises(ValueError):
        ScenarioFrame(
            frame_id=0,
            timestamp=np.nan,
            objects=(),
        )


def test_frame_get_object():
    vehicle = make_vehicle()

    frame = ScenarioFrame(
        frame_id=0,
        timestamp=0.0,
        objects=(make_static(), vehicle),
    )

    assert frame.get_object("vehicle_1") == vehicle


def test_frame_get_missing_object():
    frame = ScenarioFrame(
        frame_id=0,
        timestamp=0.0,
        objects=(),
    )

    with pytest.raises(KeyError):
        frame.get_object("missing")


def test_sequence_requires_chronological_order():
    frames = (
        ScenarioFrame(frame_id=0, timestamp=1.0, objects=()),
        ScenarioFrame(frame_id=1, timestamp=0.5, objects=()),
    )

    with pytest.raises(ValueError):
        ScenarioSequence(frames=frames)


def test_sequence_properties():
    frames = (
        ScenarioFrame(frame_id="a", timestamp=0.0, objects=()),
        ScenarioFrame(frame_id="b", timestamp=0.1, objects=()),
        ScenarioFrame(frame_id="c", timestamp=0.3, objects=()),
    )

    sequence = ScenarioSequence(frames=frames)

    assert sequence.frame_count == 3
    assert sequence.frame_ids == ("a", "b", "c")
    assert sequence.timestamps == (0.0, 0.1, 0.3)
    assert np.allclose(sequence.delta_times, [0.1, 0.2])


def test_empty_sequence_is_valid():
    sequence = ScenarioSequence(frames=())

    assert sequence.frame_count == 0
    assert sequence.frame_ids == ()
    assert sequence.timestamps == ()
    assert sequence.delta_times == ()


def test_linear_motion_sequence_moves_vehicle():
    vehicle = make_vehicle()

    sequence = build_linear_motion_sequence(
        frame_ids=[0, 1, 2],
        timestamps=[0.0, 1.0, 2.0],
        static_objects=[make_static()],
        moving_objects=[vehicle],
    )

    positions = [
        frame.get_object("vehicle_1").position
        for frame in sequence.frames
    ]

    assert np.allclose(positions[0], [0.0, 0.0, 0.0])
    assert np.allclose(positions[1], [5.0, 0.0, 0.0])
    assert np.allclose(positions[2], [10.0, 0.0, 0.0])


def test_linear_motion_preserves_static_object():
    sequence = build_linear_motion_sequence(
        frame_ids=[0, 1, 2],
        timestamps=[0.0, 1.0, 2.0],
        static_objects=[make_static()],
        moving_objects=[make_vehicle()],
    )

    positions = [
        frame.get_object("building").position
        for frame in sequence.frames
    ]

    assert np.allclose(positions[0], [10.0, 5.0, 0.0])
    assert np.allclose(positions[1], [10.0, 5.0, 0.0])
    assert np.allclose(positions[2], [10.0, 5.0, 0.0])


def test_linear_motion_preserves_velocity():
    sequence = build_linear_motion_sequence(
        frame_ids=[0, 1],
        timestamps=[2.0, 4.0],
        static_objects=[],
        moving_objects=[make_vehicle()],
    )

    for frame in sequence.frames:
        vehicle = frame.get_object("vehicle_1")
        assert np.allclose(vehicle.velocity, [5.0, 0.0, 0.0])


def test_linear_motion_requires_matching_lengths():
    with pytest.raises(ValueError):
        build_linear_motion_sequence(
            frame_ids=[0, 1],
            timestamps=[0.0],
            static_objects=[],
            moving_objects=[],
        )


def test_linear_motion_requires_increasing_timestamps():
    with pytest.raises(ValueError):
        build_linear_motion_sequence(
            frame_ids=[0, 1],
            timestamps=[0.0, 0.0],
            static_objects=[],
            moving_objects=[],
        )


def test_linear_motion_empty_sequence():
    sequence = build_linear_motion_sequence(
        frame_ids=[],
        timestamps=[],
        static_objects=[],
        moving_objects=[],
    )

    assert sequence.frame_count == 0


def test_object_trajectory():
    vehicle = make_vehicle()

    sequence = build_linear_motion_sequence(
        frame_ids=[0, 1, 2],
        timestamps=[0.0, 0.5, 1.0],
        static_objects=[],
        moving_objects=[vehicle],
    )

    trajectory = sequence.object_trajectory("vehicle_1")

    assert len(trajectory) == 3
    assert trajectory[0][0] == 0.0
    assert np.allclose(trajectory[0][1], [0.0, 0.0, 0.0])
    assert np.allclose(trajectory[1][1], [2.5, 0.0, 0.0])
    assert np.allclose(trajectory[2][1], [5.0, 0.0, 0.0])


def test_object_trajectory_skips_missing_frames():
    frame0 = ScenarioFrame(
        frame_id=0,
        timestamp=0.0,
        objects=(make_vehicle(),),
    )

    frame1 = ScenarioFrame(
        frame_id=1,
        timestamp=1.0,
        objects=(),
    )

    sequence = ScenarioSequence(frames=(frame0, frame1))

    trajectory = sequence.object_trajectory("vehicle_1")

    assert len(trajectory) == 1


def test_ground_truth_conversion():
    frame = ScenarioFrame(
        frame_id=5,
        timestamp=2.0,
        objects=(make_static(), make_vehicle()),
    )

    ground_truth = scenario_frame_to_ground_truth(frame)

    assert ground_truth["frame_id"] == 5
    assert ground_truth["timestamp"] == 2.0
    assert ground_truth["object_ids"] == ("building", "vehicle_1")
    assert ground_truth["classes"] == ("STATIC", "VEHICLE")
    assert ground_truth["positions"].shape == (2, 3)
    assert ground_truth["velocities"].shape == (2, 3)
    assert ground_truth["dynamic"].tolist() == [False, True]


def test_ground_truth_dynamic_mask():
    frame = ScenarioFrame(
        frame_id=0,
        timestamp=0.0,
        objects=(make_static(), make_vehicle(), make_pedestrian()),
    )

    ground_truth = scenario_frame_to_ground_truth(frame)

    assert np.array_equal(
        ground_truth["dynamic"],
        np.array([False, True, True]),
    )