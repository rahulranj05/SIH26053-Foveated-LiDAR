"""
A26 — Ego-Motion Compensation tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from mapping.ego_motion import (
    EgoMotion,
    compensate_current_frame,
    compensate_previous_frame,
    compose_ego_motion,
    ego_motion_matrix,
    inverse_ego_motion,
    transform_points,
    yaw_rotation_matrix,
)


# ============================================================================
# EGOMOTION DATA MODEL
# ============================================================================


def test_default_motion_is_identity() -> None:
    motion = EgoMotion()

    assert np.allclose(
        motion.translation,
        [0.0, 0.0, 0.0],
    )

    assert motion.yaw == 0.0


def test_translation_property() -> None:
    motion = EgoMotion(
        translation_x=1.0,
        translation_y=2.0,
        translation_z=3.0,
    )

    assert np.allclose(
        motion.translation,
        [1.0, 2.0, 3.0],
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"translation_x": np.nan},
        {"translation_y": np.inf},
        {"translation_z": -np.inf},
        {"yaw": np.nan},
    ],
)
def test_nonfinite_motion_is_rejected(
    kwargs: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        EgoMotion(**kwargs)


# ============================================================================
# ROTATION
# ============================================================================


def test_zero_yaw_rotation_is_identity() -> None:
    rotation = yaw_rotation_matrix(0.0)

    assert np.allclose(
        rotation,
        np.eye(3),
    )


def test_ninety_degree_yaw_rotation() -> None:
    rotation = yaw_rotation_matrix(
        np.pi / 2.0
    )

    point = np.array(
        [1.0, 0.0, 0.0]
    )

    rotated = rotation @ point

    assert np.allclose(
        rotated,
        [0.0, 1.0, 0.0],
        atol=1e-12,
    )


def test_yaw_rotation_preserves_z() -> None:
    rotation = yaw_rotation_matrix(
        np.pi / 3.0
    )

    point = np.array(
        [1.0, 2.0, 7.0]
    )

    rotated = rotation @ point

    assert np.isclose(
        rotated[2],
        7.0,
    )


def test_rotation_matrix_is_orthogonal() -> None:
    rotation = yaw_rotation_matrix(
        0.73
    )

    assert np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        atol=1e-12,
    )


def test_rotation_matrix_rejects_nan() -> None:
    with pytest.raises(ValueError):
        yaw_rotation_matrix(np.nan)


# ============================================================================
# HOMOGENEOUS MATRIX
# ============================================================================


def test_homogeneous_matrix_shape() -> None:
    matrix = ego_motion_matrix(
        EgoMotion()
    )

    assert matrix.shape == (4, 4)


def test_homogeneous_matrix_identity() -> None:
    matrix = ego_motion_matrix(
        EgoMotion()
    )

    assert np.allclose(
        matrix,
        np.eye(4),
    )


def test_homogeneous_matrix_translation() -> None:
    motion = EgoMotion(
        translation_x=1.0,
        translation_y=2.0,
        translation_z=3.0,
    )

    matrix = ego_motion_matrix(motion)

    assert np.allclose(
        matrix[:3, 3],
        [1.0, 2.0, 3.0],
    )


def test_homogeneous_matrix_rejects_invalid_motion() -> None:
    with pytest.raises(TypeError):
        ego_motion_matrix(
            object()  # type: ignore[arg-type]
        )


# ============================================================================
# POINT TRANSFORMATION
# ============================================================================


def test_identity_transform_preserves_points() -> None:
    points = np.array(
        [
            [1.0, 2.0, 3.0],
            [-1.0, 0.5, 2.0],
        ]
    )

    transformed = transform_points(
        points,
        EgoMotion(),
    )

    assert np.allclose(
        transformed,
        points,
    )

    assert transformed is not points


def test_translation_transform() -> None:
    points = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
        ]
    )

    motion = EgoMotion(
        translation_x=3.0,
        translation_y=-2.0,
        translation_z=1.0,
    )

    transformed = transform_points(
        points,
        motion,
    )

    expected = np.array(
        [
            [4.0, -2.0, 1.0],
            [5.0, -1.0, 1.0],
        ]
    )

    assert np.allclose(
        transformed,
        expected,
    )


def test_rotation_transform() -> None:
    points = np.array(
        [[1.0, 0.0, 0.0]]
    )

    transformed = transform_points(
        points,
        EgoMotion(
            yaw=np.pi / 2.0
        ),
    )

    assert np.allclose(
        transformed,
        [[0.0, 1.0, 0.0]],
        atol=1e-12,
    )


def test_rotation_and_translation_transform() -> None:
    points = np.array(
        [[1.0, 0.0, 0.0]]
    )

    motion = EgoMotion(
        translation_x=2.0,
        translation_y=3.0,
        yaw=np.pi / 2.0,
    )

    transformed = transform_points(
        points,
        motion,
    )

    assert np.allclose(
        transformed,
        [[2.0, 4.0, 0.0]],
        atol=1e-12,
    )


def test_transform_does_not_mutate_input() -> None:
    points = np.array(
        [[1.0, 2.0, 3.0]]
    )

    original = points.copy()

    transform_points(
        points,
        EgoMotion(
            translation_x=5.0
        ),
    )

    assert np.array_equal(
        points,
        original,
    )


def test_empty_point_cloud_is_supported() -> None:
    points = np.empty(
        (0, 3)
    )

    transformed = transform_points(
        points,
        EgoMotion(),
    )

    assert transformed.shape == (0, 3)


@pytest.mark.parametrize(
    "points",
    [
        np.zeros((3, 2)),
        np.zeros((3, 4)),
        np.zeros(3),
    ],
)
def test_invalid_point_shapes_are_rejected(
    points: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        transform_points(
            points,
            EgoMotion(),
        )


def test_nonfinite_points_are_rejected() -> None:
    points = np.array(
        [
            [1.0, 0.0, 0.0],
            [np.nan, 1.0, 0.0],
        ]
    )

    with pytest.raises(ValueError):
        transform_points(
            points,
            EgoMotion(),
        )


def test_invalid_motion_is_rejected_for_transform() -> None:
    with pytest.raises(TypeError):
        transform_points(
            np.zeros((1, 3)),
            object(),  # type: ignore[arg-type]
        )


# ============================================================================
# INVERSE MOTION
# ============================================================================


def test_inverse_identity_is_identity() -> None:
    inverse = inverse_ego_motion(
        EgoMotion()
    )

    assert np.allclose(
        inverse.translation,
        [0.0, 0.0, 0.0],
    )

    assert inverse.yaw == 0.0


def test_inverse_motion_restores_points() -> None:
    points = np.array(
        [
            [1.0, 2.0, 0.0],
            [-3.0, 1.0, 2.0],
        ]
    )

    motion = EgoMotion(
        translation_x=4.0,
        translation_y=-2.0,
        translation_z=1.0,
        yaw=0.7,
    )

    transformed = transform_points(
        points,
        motion,
    )

    restored = transform_points(
        transformed,
        inverse_ego_motion(motion),
    )

    assert np.allclose(
        restored,
        points,
        atol=1e-12,
    )


def test_inverse_negates_yaw() -> None:
    motion = EgoMotion(yaw=0.5)

    inverse = inverse_ego_motion(motion)

    assert np.isclose(
        inverse.yaw,
        -0.5,
    )


def test_inverse_rejects_invalid_type() -> None:
    with pytest.raises(TypeError):
        inverse_ego_motion(
            object()  # type: ignore[arg-type]
        )


# ============================================================================
# FRAME COMPENSATION
# ============================================================================


def test_previous_frame_compensation() -> None:
    previous = np.array(
        [[1.0, 0.0, 0.0]]
    )

    motion = EgoMotion(
        translation_x=2.0
    )

    compensated = compensate_previous_frame(
        previous,
        motion,
    )

    assert np.allclose(
        compensated,
        [[3.0, 0.0, 0.0]],
    )


def test_current_frame_compensation() -> None:
    current = np.array(
        [[3.0, 0.0, 0.0]]
    )

    motion = EgoMotion(
        translation_x=2.0
    )

    compensated = compensate_current_frame(
        current,
        motion,
    )

    assert np.allclose(
        compensated,
        [[1.0, 0.0, 0.0]],
    )


def test_previous_and_current_compensation_round_trip() -> None:
    points = np.array(
        [
            [1.0, 2.0, 0.0],
            [-1.0, 4.0, 2.0],
        ]
    )

    motion = EgoMotion(
        translation_x=1.0,
        translation_y=-2.0,
        yaw=0.4,
    )

    current = compensate_previous_frame(
        points,
        motion,
    )

    restored = compensate_current_frame(
        current,
        motion,
    )

    assert np.allclose(
        restored,
        points,
        atol=1e-12,
    )


# ============================================================================
# COMPOSITION
# ============================================================================


def test_composed_identity_is_identity() -> None:
    combined = compose_ego_motion(
        EgoMotion(),
        EgoMotion(),
    )

    assert np.allclose(
        combined.translation,
        [0.0, 0.0, 0.0],
    )

    assert combined.yaw == 0.0


def test_composed_translation() -> None:
    first = EgoMotion(
        translation_x=1.0
    )

    second = EgoMotion(
        translation_x=2.0
    )

    combined = compose_ego_motion(
        first,
        second,
    )

    assert np.allclose(
        combined.translation,
        [3.0, 0.0, 0.0],
    )


def test_composition_matches_sequential_transform() -> None:
    points = np.array(
        [
            [1.0, 2.0, 0.0],
            [3.0, -1.0, 1.0],
        ]
    )

    first = EgoMotion(
        translation_x=1.0,
        translation_y=2.0,
        yaw=0.3,
    )

    second = EgoMotion(
        translation_x=-0.5,
        translation_y=1.5,
        yaw=-0.2,
    )

    sequential = transform_points(
        transform_points(
            points,
            first,
        ),
        second,
    )

    combined = compose_ego_motion(
        first,
        second,
    )

    direct = transform_points(
        points,
        combined,
    )

    assert np.allclose(
        sequential,
        direct,
        atol=1e-12,
    )


def test_composition_rejects_invalid_first() -> None:
    with pytest.raises(TypeError):
        compose_ego_motion(
            object(),  # type: ignore[arg-type]
            EgoMotion(),
        )


def test_composition_rejects_invalid_second() -> None:
    with pytest.raises(TypeError):
        compose_ego_motion(
            EgoMotion(),
            object(),  # type: ignore[arg-type]
        )