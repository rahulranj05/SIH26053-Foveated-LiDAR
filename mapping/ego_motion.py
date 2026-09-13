"""
A26 — Ego-motion compensation for FoveaMap.

This module provides deterministic rigid-body ego-motion transforms for
LiDAR point clouds.

The transform convention used throughout this module is:

    p_new = R @ p_old + t

where:

    R = yaw rotation matrix
    t = translation vector

A26 is intentionally independent of CARLA and any specific dataset.
Dataset/sensor integration is handled by later milestones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EgoMotion:
    """
    Rigid-body ego motion represented by planar yaw and 3D translation.

    Parameters
    ----------
    translation_x:
        Translation along the x axis.
    translation_y:
        Translation along the y axis.
    translation_z:
        Translation along the z axis.
    yaw:
        Rotation around the z axis in radians.
    """

    translation_x: float = 0.0
    translation_y: float = 0.0
    translation_z: float = 0.0
    yaw: float = 0.0

    def __post_init__(self) -> None:
        """Validate that every motion parameter is finite."""
        values = np.array(
            [
                self.translation_x,
                self.translation_y,
                self.translation_z,
                self.yaw,
            ],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "EgoMotion parameters must all be finite"
            )

    @property
    def translation(self) -> np.ndarray:
        """Return the translation vector as a float64 array."""
        return np.array(
            [
                self.translation_x,
                self.translation_y,
                self.translation_z,
            ],
            dtype=np.float64,
        )


def yaw_rotation_matrix(yaw: float) -> np.ndarray:
    """
    Construct a 3x3 rotation matrix for a yaw rotation.

    Parameters
    ----------
    yaw:
        Rotation around the z axis in radians.

    Returns
    -------
    numpy.ndarray
        A 3x3 float64 rotation matrix.
    """
    yaw = float(yaw)

    if not np.isfinite(yaw):
        raise ValueError("yaw must be finite")

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    return np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def ego_motion_matrix(motion: EgoMotion) -> np.ndarray:
    """
    Construct the homogeneous 4x4 transformation matrix.

    The matrix represents:

        p_new = R @ p_old + t
    """
    if not isinstance(motion, EgoMotion):
        raise TypeError("motion must be an EgoMotion")

    matrix = np.eye(4, dtype=np.float64)

    matrix[:3, :3] = yaw_rotation_matrix(motion.yaw)
    matrix[:3, 3] = motion.translation

    return matrix


def inverse_ego_motion(motion: EgoMotion) -> EgoMotion:
    """
    Return the inverse rigid-body motion.

    For:

        p_new = R @ p_old + t

    the inverse is:

        p_old = R.T @ p_new - R.T @ t
    """
    if not isinstance(motion, EgoMotion):
        raise TypeError("motion must be an EgoMotion")

    rotation = yaw_rotation_matrix(motion.yaw)

    inverse_translation = -rotation.T @ motion.translation

    return EgoMotion(
        translation_x=float(inverse_translation[0]),
        translation_y=float(inverse_translation[1]),
        translation_z=float(inverse_translation[2]),
        yaw=float(-motion.yaw),
    )


def _validate_points(points: np.ndarray) -> np.ndarray:
    """
    Validate and normalize a point-cloud array.

    Returns a float64 array suitable for transformation.
    """
    points = np.asarray(points)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            "points must have shape (N, 3)"
        )

    if not np.issubdtype(points.dtype, np.number):
        raise TypeError(
            "points must contain numeric values"
        )

    points = np.asarray(
        points,
        dtype=np.float64,
    )

    if not np.all(np.isfinite(points)):
        raise ValueError(
            "points must contain only finite values"
        )

    return points


def transform_points(
    points: np.ndarray,
    motion: EgoMotion,
) -> np.ndarray:
    """
    Apply an ego-motion transform to a point cloud.

    Parameters
    ----------
    points:
        Point cloud with shape (N, 3).
    motion:
        EgoMotion transformation.

    Returns
    -------
    numpy.ndarray
        Transformed point cloud with shape (N, 3).
    """
    if not isinstance(motion, EgoMotion):
        raise TypeError("motion must be an EgoMotion")

    points = _validate_points(points)

    rotation = yaw_rotation_matrix(motion.yaw)

    transformed = (
        points @ rotation.T
        + motion.translation
    )

    return np.asarray(
        transformed,
        dtype=np.float64,
    )


def compensate_previous_frame(
    previous_points: np.ndarray,
    motion: EgoMotion,
) -> np.ndarray:
    """
    Transform the previous-frame point cloud into the current-frame frame.

    This is useful when ``motion`` describes the ego vehicle movement from
    the previous frame to the current frame.

    The previous-frame points are transformed using the forward ego motion.
    """
    if not isinstance(motion, EgoMotion):
        raise TypeError("motion must be an EgoMotion")

    return transform_points(
        previous_points,
        motion,
    )


def compensate_current_frame(
    current_points: np.ndarray,
    motion: EgoMotion,
) -> np.ndarray:
    """
    Transform the current-frame point cloud into the previous-frame frame.

    This applies the inverse of the supplied ego motion.
    """
    if not isinstance(motion, EgoMotion):
        raise TypeError("motion must be an EgoMotion")

    return transform_points(
        current_points,
        inverse_ego_motion(motion),
    )


def compose_ego_motion(
    first: EgoMotion,
    second: EgoMotion,
) -> EgoMotion:
    """
    Compose two ego motions.

    The returned motion is equivalent to applying ``first`` and then
    ``second`` to a point cloud.

    For:

        p1 = R1 @ p + t1
        p2 = R2 @ p1 + t2

    the combined transform is:

        p2 = (R2 @ R1) @ p + (R2 @ t1 + t2)
    """
    if not isinstance(
        first,
        EgoMotion,
    ):
        raise TypeError(
            "first must be an EgoMotion"
        )

    if not isinstance(
        second,
        EgoMotion,
    ):
        raise TypeError(
            "second must be an EgoMotion"
        )

    second_rotation = yaw_rotation_matrix(
        second.yaw
    )

    combined_translation = (
        second_rotation
        @ first.translation
        + second.translation
    )

    return EgoMotion(
        translation_x=float(
            combined_translation[0]
        ),
        translation_y=float(
            combined_translation[1]
        ),
        translation_z=float(
            combined_translation[2]
        ),
        yaw=float(
            first.yaw + second.yaw
        ),
    )