from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

import numpy as np


class ScenarioObjectClass(str, Enum):
    """Canonical object classes for controlled dynamic-world scenarios."""

    UNKNOWN = "UNKNOWN"
    STATIC = "STATIC"
    VEHICLE = "VEHICLE"
    PEDESTRIAN = "PEDESTRIAN"


@dataclass(frozen=True)
class ScenarioObject:
    """
    Ground-truth state for one object in a controlled scenario.

    Position and velocity are expressed in the same world coordinate frame.
    """

    object_id: Any
    object_class: ScenarioObjectClass
    position: np.ndarray
    velocity: np.ndarray
    dynamic: bool

    def __post_init__(self) -> None:
        if not isinstance(self.object_class, ScenarioObjectClass):
            raise TypeError("object_class must be a ScenarioObjectClass")

        position = np.asarray(self.position, dtype=np.float64)
        velocity = np.asarray(self.velocity, dtype=np.float64)

        if position.shape != (3,):
            raise ValueError("position must have shape (3,)")

        if velocity.shape != (3,):
            raise ValueError("velocity must have shape (3,)")

        if not np.all(np.isfinite(position)):
            raise ValueError("position must contain only finite values")

        if not np.all(np.isfinite(velocity)):
            raise ValueError("velocity must contain only finite values")

        if not isinstance(self.dynamic, (bool, np.bool_)):
            raise TypeError("dynamic must be a boolean")

        if self.dynamic and np.linalg.norm(velocity) == 0.0:
            raise ValueError(
                "dynamic objects must have non-zero velocity"
            )

        if not self.dynamic and np.linalg.norm(velocity) != 0.0:
            raise ValueError(
                "static objects must have zero velocity"
            )

        position = position.copy()
        velocity = velocity.copy()

        position.setflags(write=False)
        velocity.setflags(write=False)

        object.__setattr__(self, "position", position)
        object.__setattr__(self, "velocity", velocity)


@dataclass(frozen=True)
class ScenarioFrame:
    """
    Ground-truth state of a controlled scenario at one timestamp.

    Objects are stored in deterministic tuple form so a scenario frame is
    immutable and safe to retain as part of an evaluation sequence.
    """

    frame_id: Any
    timestamp: float
    objects: tuple[ScenarioObject, ...]

    def __post_init__(self) -> None:
        timestamp = float(self.timestamp)

        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")

        objects = tuple(self.objects)

        if len({obj.object_id for obj in objects}) != len(objects):
            raise ValueError("object IDs must be unique within a frame")

        for obj in objects:
            if not isinstance(obj, ScenarioObject):
                raise TypeError(
                    "objects must contain only ScenarioObject instances"
                )

        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "objects", objects)

    @property
    def object_count(self) -> int:
        """Return the number of objects in this frame."""

        return len(self.objects)

    @property
    def dynamic_objects(self) -> tuple[ScenarioObject, ...]:
        """Return all dynamically moving objects."""

        return tuple(obj for obj in self.objects if obj.dynamic)

    @property
    def static_objects(self) -> tuple[ScenarioObject, ...]:
        """Return all static objects."""

        return tuple(obj for obj in self.objects if not obj.dynamic)

    def get_object(self, object_id: Any) -> ScenarioObject:
        """Return an object by ID."""

        for obj in self.objects:
            if obj.object_id == object_id:
                return obj

        raise KeyError(f"object ID not found: {object_id!r}")


@dataclass(frozen=True)
class ScenarioSequence:
    """
    Ordered ground-truth sequence for dynamic-world evaluation.
    """

    frames: tuple[ScenarioFrame, ...]

    def __post_init__(self) -> None:
        frames = tuple(self.frames)

        previous_timestamp: float | None = None

        for frame in frames:
            if not isinstance(frame, ScenarioFrame):
                raise TypeError(
                    "frames must contain only ScenarioFrame instances"
                )

            if previous_timestamp is not None:
                if frame.timestamp <= previous_timestamp:
                    raise ValueError(
                        "scenario frame timestamps must be strictly increasing"
                    )

            previous_timestamp = frame.timestamp

        object.__setattr__(self, "frames", frames)

    @property
    def frame_count(self) -> int:
        """Return the number of frames."""

        return len(self.frames)

    @property
    def frame_ids(self) -> tuple[Any, ...]:
        """Return frame IDs in chronological order."""

        return tuple(frame.frame_id for frame in self.frames)

    @property
    def timestamps(self) -> tuple[float, ...]:
        """Return timestamps in chronological order."""

        return tuple(frame.timestamp for frame in self.frames)

    @property
    def delta_times(self) -> tuple[float, ...]:
        """Return inter-frame durations."""

        if len(self.frames) < 2:
            return ()

        return tuple(
            current.timestamp - previous.timestamp
            for previous, current in zip(self.frames[:-1], self.frames[1:])
        )

    def object_trajectory(
        self,
        object_id: Any,
    ) -> tuple[tuple[float, np.ndarray, np.ndarray], ...]:
        """
        Return the known trajectory for an object.

        Each entry is:
            (timestamp, position, velocity)

        Frames where the object is absent are skipped.
        """

        trajectory: list[tuple[float, np.ndarray, np.ndarray]] = []

        for frame in self.frames:
            try:
                obj = frame.get_object(object_id)
            except KeyError:
                continue

            trajectory.append(
                (
                    frame.timestamp,
                    obj.position.copy(),
                    obj.velocity.copy(),
                )
            )

        return tuple(trajectory)


def create_static_object(
    object_id: Any,
    position: Iterable[float],
    object_class: ScenarioObjectClass = ScenarioObjectClass.STATIC,
) -> ScenarioObject:
    """
    Convenience constructor for a static scenario object.
    """

    if object_class not in (
        ScenarioObjectClass.STATIC,
        ScenarioObjectClass.UNKNOWN,
    ):
        raise ValueError(
            "static objects must use STATIC or UNKNOWN class"
        )

    return ScenarioObject(
        object_id=object_id,
        object_class=object_class,
        position=np.asarray(position, dtype=np.float64),
        velocity=np.zeros(3, dtype=np.float64),
        dynamic=False,
    )


def create_moving_object(
    object_id: Any,
    position: Iterable[float],
    velocity: Iterable[float],
    object_class: ScenarioObjectClass,
) -> ScenarioObject:
    """
    Convenience constructor for a moving vehicle or pedestrian.
    """

    if object_class not in (
        ScenarioObjectClass.VEHICLE,
        ScenarioObjectClass.PEDESTRIAN,
    ):
        raise ValueError(
            "moving objects must use VEHICLE or PEDESTRIAN class"
        )

    return ScenarioObject(
        object_id=object_id,
        object_class=object_class,
        position=np.asarray(position, dtype=np.float64),
        velocity=np.asarray(velocity, dtype=np.float64),
        dynamic=True,
    )


def build_linear_motion_sequence(
    *,
    frame_ids: Iterable[Any],
    timestamps: Iterable[float],
    static_objects: Iterable[ScenarioObject],
    moving_objects: Iterable[ScenarioObject],
) -> ScenarioSequence:
    """
    Build a deterministic sequence from constant-velocity motion.

    The supplied moving-object positions are interpreted as their positions
    at the first timestamp. Their positions at later timestamps are:

        p(t) = p0 + v * (t - t0)

    Static objects retain their original positions.
    """

    frame_ids = tuple(frame_ids)
    timestamps = tuple(float(timestamp) for timestamp in timestamps)
    static_objects = tuple(static_objects)
    moving_objects = tuple(moving_objects)

    if len(frame_ids) != len(timestamps):
        raise ValueError(
            "frame_ids and timestamps must have the same length"
        )

    if not timestamps:
        return ScenarioSequence(frames=())

    if not np.all(np.isfinite(np.asarray(timestamps, dtype=np.float64))):
        raise ValueError("timestamps must be finite")

    for previous, current in zip(timestamps[:-1], timestamps[1:]):
        if current <= previous:
            raise ValueError(
                "timestamps must be strictly increasing"
            )

    first_timestamp = timestamps[0]
    frames: list[ScenarioFrame] = []

    for frame_id, timestamp in zip(frame_ids, timestamps):
        objects: list[ScenarioObject] = list(static_objects)

        elapsed = timestamp - first_timestamp

        for moving in moving_objects:
            position = moving.position + moving.velocity * elapsed

            objects.append(
                ScenarioObject(
                    object_id=moving.object_id,
                    object_class=moving.object_class,
                    position=position,
                    velocity=moving.velocity,
                    dynamic=True,
                )
            )

        frames.append(
            ScenarioFrame(
                frame_id=frame_id,
                timestamp=timestamp,
                objects=tuple(objects),
            )
        )

    return ScenarioSequence(frames=tuple(frames))


def scenario_frame_to_ground_truth(
    frame: ScenarioFrame,
) -> dict[str, Any]:
    """
    Convert a ScenarioFrame into a simple evaluation-oriented dictionary.

    This is intentionally independent of CARLA and can later be populated
    from a CARLA adapter.
    """

    if not isinstance(frame, ScenarioFrame):
        raise TypeError("frame must be a ScenarioFrame")

    return {
        "frame_id": frame.frame_id,
        "timestamp": frame.timestamp,
        "object_ids": tuple(obj.object_id for obj in frame.objects),
        "classes": tuple(obj.object_class.value for obj in frame.objects),
        "positions": np.asarray(
            [obj.position for obj in frame.objects],
            dtype=np.float64,
        ),
        "velocities": np.asarray(
            [obj.velocity for obj in frame.objects],
            dtype=np.float64,
        ),
        "dynamic": np.asarray(
            [obj.dynamic for obj in frame.objects],
            dtype=bool,
        ),
    }