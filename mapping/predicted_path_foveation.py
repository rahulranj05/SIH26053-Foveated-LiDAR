from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VehicleState:
    speed: float
    yaw_rate: float
    heading: float = 0.0


def validate_xyz(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3).")

    if not np.all(np.isfinite(xyz)):
        raise ValueError("xyz must contain only finite values.")

    return xyz


def predict_path(
    state: VehicleState,
    horizon: float = 5.0,
    step: float = 0.25,
) -> np.ndarray:
    if not np.isfinite(state.speed) or state.speed < 0.0:
        raise ValueError("speed must be finite and non-negative.")

    if not np.isfinite(state.yaw_rate):
        raise ValueError("yaw_rate must be finite.")

    if not np.isfinite(state.heading):
        raise ValueError("heading must be finite.")

    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be finite and positive.")

    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step must be finite and positive.")

    times = np.arange(0.0, horizon + step * 0.5, step)

    speed = float(state.speed)
    yaw_rate = float(state.yaw_rate)
    heading = float(state.heading)

    if abs(yaw_rate) < 1e-9:
        distance = speed * times

        x = distance * np.cos(heading)
        y = distance * np.sin(heading)

    else:
        radius = speed / yaw_rate
        angles = heading + yaw_rate * times

        x = radius * (
            np.sin(angles) - np.sin(heading)
        )

        y = -radius * (
            np.cos(angles) - np.cos(heading)
        )

    z = np.zeros_like(x)

    return np.column_stack((x, y, z))


def _point_to_segment_distances(
    points_xy: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray:
    """
    Compute Euclidean distance from every point to one line segment.
    """

    segment = end - start

    segment_length_sq = float(
        np.dot(segment, segment)
    )

    if segment_length_sq <= 1e-12:
        difference = points_xy - start

        return np.sqrt(
            np.sum(difference * difference, axis=1)
        )

    relative = points_xy - start

    projection = (
        np.sum(relative * segment, axis=1)
        / segment_length_sq
    )

    projection = np.clip(
        projection,
        0.0,
        1.0,
    )

    closest = (
        start
        + projection[:, None] * segment
    )

    difference = points_xy - closest

    return np.sqrt(
        np.sum(difference * difference, axis=1)
    )


def _point_to_segment_squared_distances(
    points_xy: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray:
    """
    Compute squared Euclidean distance from every point
    to one line segment.

    This avoids sqrt() for every path segment.
    """

    segment = end - start

    segment_length_sq = float(
        np.dot(segment, segment)
    )

    if segment_length_sq <= 1e-12:
        difference = points_xy - start

        return np.sum(
            difference * difference,
            axis=1,
        )

    relative = points_xy - start

    projection = (
        np.sum(relative * segment, axis=1)
        / segment_length_sq
    )

    projection = np.clip(
        projection,
        0.0,
        1.0,
    )

    closest = (
        start
        + projection[:, None] * segment
    )

    difference = points_xy - closest

    return np.sum(
        difference * difference,
        axis=1,
    )


def distance_to_path(
    xyz: np.ndarray,
    path: np.ndarray,
) -> np.ndarray:
    """
    Compute the minimum Euclidean distance from every
    LiDAR point to the continuous predicted path.

    The path is treated as line segments between consecutive
    predicted trajectory samples.
    """

    xyz = validate_xyz(xyz)

    path = np.asarray(
        path,
        dtype=np.float64,
    )

    if path.ndim != 2 or path.shape[1] != 3:
        raise ValueError(
            "path must have shape (M, 3)."
        )

    if len(path) == 0:
        raise ValueError(
            "path must contain at least one point."
        )

    if not np.all(np.isfinite(path)):
        raise ValueError(
            "path must contain only finite values."
        )

    if len(xyz) == 0:
        return np.empty(
            0,
            dtype=np.float64,
        )

    points_xy = xyz[:, :2]
    path_xy = path[:, :2]

    # A single path point is just point-to-point distance.
    if len(path_xy) == 1:
        difference = (
            points_xy
            - path_xy[0]
        )

        return np.sqrt(
            np.sum(
                difference * difference,
                axis=1,
            )
        )

    minimum_squared_distance = np.full(
        len(points_xy),
        np.inf,
        dtype=np.float64,
    )

    # Keep this loop intentionally small and memory efficient.
    # The original implementation was ~53 ms on the real
    # 124k-point SemanticKITTI frame, so avoid creating
    # enormous (N x segments) temporary arrays.
    for index in range(len(path_xy) - 1):
        squared_distances = (
            _point_to_segment_squared_distances(
                points_xy,
                path_xy[index],
                path_xy[index + 1],
            )
        )

        minimum_squared_distance = np.minimum(
            minimum_squared_distance,
            squared_distances,
        )

    return np.sqrt(
        np.maximum(
            minimum_squared_distance,
            0.0,
        )
    )


def path_importance(
    xyz: np.ndarray,
    state: VehicleState,
    horizon: float = 5.0,
    step: float = 0.25,
    corridor_width: float = 3.0,
) -> np.ndarray:
    if (
        not np.isfinite(corridor_width)
        or corridor_width <= 0.0
    ):
        raise ValueError(
            "corridor_width must be finite and positive."
        )

    xyz = validate_xyz(xyz)

    if len(xyz) == 0:
        return np.empty(
            0,
            dtype=np.float64,
        )

    path = predict_path(
        state,
        horizon=horizon,
        step=step,
    )

    distances = distance_to_path(
        xyz,
        path,
    )

    importance = (
        1.0
        - distances / corridor_width
    )

    return np.clip(
        importance,
        0.0,
        1.0,
    )


def resolution_from_path_importance(
    importance: np.ndarray,
) -> np.ndarray:
    importance = np.asarray(
        importance,
        dtype=np.float64,
    )

    if importance.ndim != 1:
        raise ValueError(
            "importance must be a 1D array."
        )

    if not np.all(np.isfinite(importance)):
        raise ValueError(
            "importance must contain only finite values."
        )

    if (
        np.any(importance < 0.0)
        or np.any(importance > 1.0)
    ):
        raise ValueError(
            "importance values must be in [0, 1]."
        )

    resolutions = np.full_like(
        importance,
        0.40,
        dtype=np.float64,
    )

    resolutions[importance >= 0.25] = 0.20
    resolutions[importance >= 0.50] = 0.10
    resolutions[importance >= 0.75] = 0.05

    return resolutions
