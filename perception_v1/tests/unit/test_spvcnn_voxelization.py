import numpy as np

from perception_v1.src.perception.spvcnn.voxelization import (
    DEFAULT_VOXEL_SIZE_M,
    voxelize_points,
)


def check_default_voxel_size():
    assert DEFAULT_VOXEL_SIZE_M == 0.10


def check_signed_floor_and_negative_coordinates():
    xyz = np.array([
        [0.00, 0.00, 0.00],
        [0.09, 0.19, 0.29],
        [0.10, 0.20, 0.30],
        [-0.01, -0.11, -0.21],
        [-0.10, -0.20, -0.30],
    ], dtype=np.float32)

    result = voxelize_points(xyz)

    expected_per_point = np.floor(
        xyz.astype(np.float64) / 0.10
    ).astype(np.int64)

    recovered = result.voxel_coordinates[
        result.point_to_voxel_inverse
    ]

    assert np.array_equal(
        recovered,
        expected_per_point,
    )

    assert np.any(
        result.voxel_coordinates < 0
    )


def check_no_origin_shift():
    xyz = np.array([
        [-10.05, 2.05, -0.05],
        [3.15, -4.25, 5.35],
    ], dtype=np.float32)

    result = voxelize_points(xyz)

    expected = np.floor(
        xyz.astype(np.float64) / 0.10
    ).astype(np.int64)

    recovered = result.voxel_coordinates[
        result.point_to_voxel_inverse
    ]

    assert np.array_equal(
        recovered,
        expected,
    )


def check_inverse_mapping():
    xyz = np.array([
        [1.01, 1.01, 1.01],
        [1.02, 1.02, 1.02],
        [2.01, 2.01, 2.01],
        [1.03, 1.03, 1.03],
    ], dtype=np.float32)

    result = voxelize_points(xyz)

    reconstructed = result.voxel_coordinates[
        result.point_to_voxel_inverse
    ]

    expected = np.floor(
        xyz.astype(np.float64) / 0.10
    ).astype(np.int64)

    assert np.array_equal(
        reconstructed,
        expected,
    )

    assert (
        len(result.point_to_voxel_inverse)
        == len(xyz)
    )


def check_nearest_range_representative():
    # First three points deliberately occupy the same voxel.
    xyz = np.array([
        [1.09, 0.09, 0.09],
        [1.01, 0.01, 0.01],
        [1.05, 0.05, 0.05],
        [2.01, 0.01, 0.01],
    ], dtype=np.float32)

    result = voxelize_points(xyz)

    voxel_id = result.point_to_voxel_inverse[0]

    assert result.point_to_voxel_inverse[1] == voxel_id
    assert result.point_to_voxel_inverse[2] == voxel_id

    # Point 1 has minimum Euclidean range.
    assert (
        result.voxel_to_representative_point[voxel_id]
        == 1
    )


def check_exact_tie_uses_lowest_point_index():
    # Duplicate canonical points create a genuine exact range tie.
    xyz = np.array([
        [1.05, 0.05, 0.05],
        [1.05, 0.05, 0.05],
    ], dtype=np.float32)

    expected_voxels = np.floor(
        xyz.astype(np.float64) / 0.10
    ).astype(np.int64)

    assert np.array_equal(
        expected_voxels[0],
        expected_voxels[1],
    )

    ranges_squared = np.sum(
        xyz.astype(np.float64) ** 2,
        axis=1,
    )

    assert ranges_squared[0] == ranges_squared[1]

    result = voxelize_points(xyz)

    assert (
        result.point_to_voxel_inverse[0]
        == result.point_to_voxel_inverse[1]
    )

    voxel_id = result.point_to_voxel_inverse[0]

    # Exact tie -> lowest canonical point index.
    assert (
        result.voxel_to_representative_point[voxel_id]
        == 0
    )


def check_deterministic_replay():
    rng = np.random.default_rng(26053)

    xyz = rng.normal(
        size=(1000, 3)
    ).astype(np.float32)

    a = voxelize_points(xyz)
    b = voxelize_points(xyz)

    assert np.array_equal(
        a.voxel_coordinates,
        b.voxel_coordinates,
    )
    assert np.array_equal(
        a.point_to_voxel_inverse,
        b.point_to_voxel_inverse,
    )
    assert np.array_equal(
        a.voxel_to_representative_point,
        b.voxel_to_representative_point,
    )


def check_every_point_maps_exactly_once():
    rng = np.random.default_rng(26053)

    xyz = rng.uniform(
        -20.0,
        20.0,
        size=(5000, 3),
    ).astype(np.float32)

    result = voxelize_points(xyz)

    inverse = result.point_to_voxel_inverse

    assert inverse.shape == (len(xyz),)
    assert np.all(inverse >= 0)
    assert np.all(
        inverse < len(result.voxel_coordinates)
    )


def check_representatives_are_valid_point_indices():
    rng = np.random.default_rng(26053)

    xyz = rng.uniform(
        -2.0,
        2.0,
        size=(2000, 3),
    ).astype(np.float32)

    result = voxelize_points(xyz)

    reps = result.voxel_to_representative_point

    assert reps.shape == (
        len(result.voxel_coordinates),
    )

    assert np.all(reps >= 0)
    assert np.all(reps < len(xyz))

    # Representative must actually belong to its voxel.
    for voxel_id, point_id in enumerate(reps):
        assert (
            result.point_to_voxel_inverse[point_id]
            == voxel_id
        )


def check_empty_frame():
    xyz = np.empty(
        (0, 3),
        dtype=np.float32,
    )

    result = voxelize_points(xyz)

    assert result.voxel_coordinates.shape == (0, 3)
    assert result.point_to_voxel_inverse.shape == (0,)
    assert result.voxel_to_representative_point.shape == (0,)


def check_invalid_inputs_rejected():
    bad_shape = np.zeros(
        (4, 4),
        dtype=np.float32,
    )

    try:
        voxelize_points(bad_shape)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Invalid XYZ shape was accepted"
        )

    nonfinite = np.array(
        [[1.0, np.nan, 2.0]],
        dtype=np.float32,
    )

    try:
        voxelize_points(nonfinite)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Non-finite XYZ was accepted"
        )

    for bad_size in (
        0.0,
        -0.1,
        np.nan,
        np.inf,
    ):
        try:
            voxelize_points(
                np.zeros((1, 3), dtype=np.float32),
                voxel_size_m=bad_size,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"Invalid voxel size accepted: {bad_size}"
            )


def main():
    print("M4 SPVCNN VOXELIZATION UNIT GATE")
    print("=" * 50)

    check_default_voxel_size()
    print("[PASS] Frozen 0.10 m voxel size")

    check_signed_floor_and_negative_coordinates()
    print("[PASS] Signed floor voxel coordinates")

    check_no_origin_shift()
    print("[PASS] No origin shift")

    check_inverse_mapping()
    print("[PASS] Point-to-voxel inverse")

    check_nearest_range_representative()
    print("[PASS] Nearest-range representative")

    check_exact_tie_uses_lowest_point_index()
    print("[PASS] Lowest-index exact tie-break")

    check_deterministic_replay()
    print("[PASS] Deterministic voxelization")

    check_every_point_maps_exactly_once()
    print("[PASS] Every point maps exactly once")

    check_representatives_are_valid_point_indices()
    print("[PASS] Representative ownership")

    check_empty_frame()
    print("[PASS] Empty frame handling")

    check_invalid_inputs_rejected()
    print("[PASS] Invalid inputs rejected")

    print("=" * 50)
    print("M4 SPVCNN VOXELIZATION UNIT GATE: PASS")


if __name__ == "__main__":
    main()