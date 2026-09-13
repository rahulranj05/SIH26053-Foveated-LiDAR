import numpy as np

from models.voxel_adapter import (
    VoxelAdapter,
    project_voxel_predictions_to_points,
)


def test_single_point_per_voxel():

    adapter = VoxelAdapter(
        voxel_size=0.05,
        ignore_index=0,
    )

    xyz = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.10, 0.00, 0.00],
            [0.20, 0.00, 0.00],
        ],
        dtype=np.float32,
    )

    features = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
        ],
        dtype=np.float32,
    )

    labels = np.array(
        [1, 2, 3],
        dtype=np.int64,
    )

    out = adapter(
        xyz,
        features,
        labels,
    )

    assert out.voxel_coords.shape == (3, 3)
    assert out.voxel_features.shape == (3, 1)
    assert out.voxel_labels.shape == (3,)
    assert out.inverse_map.shape == (3,)

    assert np.array_equal(
        out.inverse_map,
        [0, 1, 2],
    )


def test_feature_mean_within_voxel():

    adapter = VoxelAdapter(
        voxel_size=0.10,
    )

    xyz = np.array(
        [
            [0.01, 0.01, 0.01],
            [0.02, 0.02, 0.02],
            [0.15, 0.00, 0.00],
        ],
        dtype=np.float32,
    )

    features = np.array(
        [
            [0.0],
            [1.0],
            [0.25],
        ],
        dtype=np.float32,
    )

    labels = np.array(
        [1, 1, 2],
        dtype=np.int64,
    )

    out = adapter(
        xyz,
        features,
        labels,
    )

    assert out.voxel_coords.shape[0] == 2

    assert np.isclose(
        out.voxel_features[0, 0],
        0.5,
    )

    assert np.isclose(
        out.voxel_features[1, 0],
        0.25,
    )


def test_majority_label():

    adapter = VoxelAdapter(
        voxel_size=1.0,
        ignore_index=0,
    )

    xyz = np.array(
        [
            [0.1, 0.1, 0.1],
            [0.2, 0.1, 0.1],
            [0.3, 0.1, 0.1],
        ],
        dtype=np.float32,
    )

    features = np.ones(
        (3, 1),
        dtype=np.float32,
    )

    labels = np.array(
        [4, 4, 7],
        dtype=np.int64,
    )

    out = adapter(
        xyz,
        features,
        labels,
    )

    assert out.voxel_coords.shape[0] == 1
    assert out.voxel_labels[0] == 4


def test_ignore_not_allowed_to_override_real_class():

    adapter = VoxelAdapter(
        voxel_size=1.0,
        ignore_index=0,
    )

    xyz = np.array(
        [
            [0.1, 0.1, 0.1],
            [0.2, 0.1, 0.1],
            [0.3, 0.1, 0.1],
        ],
        dtype=np.float32,
    )

    features = np.ones(
        (3, 1),
        dtype=np.float32,
    )

    labels = np.array(
        [0, 0, 16],
        dtype=np.int64,
    )

    out = adapter(
        xyz,
        features,
        labels,
    )

    assert out.voxel_labels[0] == 16


def test_all_ignore_stays_ignore():

    adapter = VoxelAdapter(
        voxel_size=1.0,
        ignore_index=0,
    )

    xyz = np.array(
        [
            [0.1, 0.1, 0.1],
            [0.2, 0.1, 0.1],
        ],
        dtype=np.float32,
    )

    features = np.ones(
        (2, 1),
        dtype=np.float32,
    )

    labels = np.array(
        [0, 0],
        dtype=np.int64,
    )

    out = adapter(
        xyz,
        features,
        labels,
    )

    assert out.voxel_labels[0] == 0


def test_negative_coordinates_floor_correctly():

    adapter = VoxelAdapter(
        voxel_size=0.1,
    )

    xyz = np.array(
        [
            [-0.01, 0.0, 0.0],
            [0.01, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    features = np.ones(
        (2, 1),
        dtype=np.float32,
    )

    labels = np.array(
        [1, 1],
        dtype=np.int64,
    )

    out = adapter(
        xyz,
        features,
        labels,
    )

    # -0.01 / 0.1 floors to -1,
    # +0.01 / 0.1 floors to 0.
    assert out.voxel_coords.shape[0] == 2


def test_inverse_projection():

    voxel_predictions = np.array(
        [4, 16],
        dtype=np.int64,
    )

    inverse = np.array(
        [0, 0, 1, 1, 1],
        dtype=np.int64,
    )

    point_predictions = (
        project_voxel_predictions_to_points(
            voxel_predictions,
            inverse,
        )
    )

    assert np.array_equal(
        point_predictions,
        [4, 4, 16, 16, 16],
    )