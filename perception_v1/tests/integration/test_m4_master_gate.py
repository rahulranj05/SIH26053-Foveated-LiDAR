import numpy as np

from perception_v1.src.perception.spvcnn.input_builder import (
    backproject_voxel_values_to_points,
    build_spvcnn_input,
)


def make_case():
    # Points 0,1,2 share voxel [10,0,0].
    # Point 1 is nearest range and must be representative.
    # Point 3 occupies a separate voxel.
    xyz = np.array([
        [1.09, 0.09, 0.09],
        [1.01, 0.01, 0.01],
        [1.05, 0.05, 0.05],
        [2.01, 0.01, 0.01],
    ], dtype=np.float32)

    features = np.array([
        [100.0, 101.0],
        [200.0, 201.0],
        [300.0, 301.0],
        [400.0, 401.0],
    ], dtype=np.float32)

    # Majority in first voxel is class 7,
    # but representative point 1 is class 19.
    # This adversarially proves NO majority vote.
    targets = np.array(
        [7, 19, 7, 24],
        dtype=np.int64,
    )

    return xyz, features, targets


def check_representative_features():
    xyz, features, targets = make_case()

    result = build_spvcnn_input(
        xyz,
        features,
        point_targets=targets,
    )

    voxel_id = result.point_to_voxel_inverse[0]

    assert result.point_to_voxel_inverse[1] == voxel_id
    assert result.point_to_voxel_inverse[2] == voxel_id

    representative = (
        result.voxel_to_representative_point[voxel_id]
    )

    assert representative == 1

    assert np.array_equal(
        result.voxel_features[voxel_id],
        features[1],
    )


def check_no_semantic_majority_vote():
    xyz, features, targets = make_case()

    result = build_spvcnn_input(
        xyz,
        features,
        point_targets=targets,
    )

    voxel_id = result.point_to_voxel_inverse[0]

    # Majority label = 7.
    assert targets[0] == 7
    assert targets[2] == 7

    # Frozen rule must instead use representative point 1.
    assert (
        result.voxel_targets[voxel_id]
        == 19
    )


def check_backprojection():
    xyz, features, targets = make_case()

    result = build_spvcnn_input(
        xyz,
        features,
        point_targets=targets,
    )

    num_voxels = len(result.voxel_coordinates)

    voxel_logits = np.arange(
        num_voxels * 24,
        dtype=np.float32,
    ).reshape(num_voxels, 24)

    point_logits = backproject_voxel_values_to_points(
        voxel_logits,
        result.point_to_voxel_inverse,
    )

    assert point_logits.shape == (
        len(xyz),
        24,
    )

    for point_id, voxel_id in enumerate(
        result.point_to_voxel_inverse
    ):
        assert np.array_equal(
            point_logits[point_id],
            voxel_logits[voxel_id],
        )


def check_all_points_retained_by_inverse():
    xyz, features, targets = make_case()

    result = build_spvcnn_input(
        xyz,
        features,
        point_targets=targets,
    )

    assert (
        len(result.point_to_voxel_inverse)
        == len(xyz)
    )

    assert np.array_equal(
        result.voxel_coordinates[
            result.point_to_voxel_inverse
        ],
        np.floor(
            xyz.astype(np.float64) / 0.10
        ).astype(np.int64),
    )


def check_deterministic_build():
    xyz, features, targets = make_case()

    a = build_spvcnn_input(
        xyz,
        features,
        point_targets=targets,
    )

    b = build_spvcnn_input(
        xyz,
        features,
        point_targets=targets,
    )

    assert np.array_equal(
        a.voxel_coordinates,
        b.voxel_coordinates,
    )
    assert np.array_equal(
        a.voxel_features,
        b.voxel_features,
    )
    assert np.array_equal(
        a.voxel_targets,
        b.voxel_targets,
    )
    assert np.array_equal(
        a.point_to_voxel_inverse,
        b.point_to_voxel_inverse,
    )
    assert np.array_equal(
        a.voxel_to_representative_point,
        b.voxel_to_representative_point,
    )


def check_inference_without_targets():
    xyz, features, _ = make_case()

    result = build_spvcnn_input(
        xyz,
        features,
    )

    assert result.voxel_targets is None


def check_input_immutability():
    xyz, features, targets = make_case()

    xyz_before = xyz.copy()
    features_before = features.copy()
    targets_before = targets.copy()

    build_spvcnn_input(
        xyz,
        features,
        point_targets=targets,
    )

    assert np.array_equal(xyz, xyz_before)
    assert np.array_equal(features, features_before)
    assert np.array_equal(targets, targets_before)


def check_invalid_alignment_rejected():
    xyz, features, targets = make_case()

    try:
        build_spvcnn_input(
            xyz,
            features[:-1],
            point_targets=targets,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Feature length mismatch accepted"
        )

    try:
        build_spvcnn_input(
            xyz,
            features,
            point_targets=targets[:-1],
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Target length mismatch accepted"
        )


def check_invalid_backprojection_rejected():
    voxel_values = np.zeros(
        (2, 24),
        dtype=np.float32,
    )

    try:
        backproject_voxel_values_to_points(
            voxel_values,
            np.array([0, 2], dtype=np.int64),
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Out-of-range voxel reference accepted"
        )

    try:
        backproject_voxel_values_to_points(
            voxel_values,
            np.array([0, -1], dtype=np.int64),
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Negative voxel reference accepted"
        )


def main():
    print("M4 MASTER GATE")
    print("=" * 50)

    check_representative_features()
    print("[PASS] Representative-point voxel features")

    check_no_semantic_majority_vote()
    print("[PASS] No semantic majority voting")

    check_backprojection()
    print("[PASS] Voxel-to-point backprojection")

    check_all_points_retained_by_inverse()
    print("[PASS] All canonical points retain voxel mapping")

    check_deterministic_build()
    print("[PASS] Deterministic sparse-input build")

    check_inference_without_targets()
    print("[PASS] Target-free inference input")

    check_input_immutability()
    print("[PASS] Canonical inputs remain immutable")

    check_invalid_alignment_rejected()
    print("[PASS] Alignment violations rejected")

    check_invalid_backprojection_rejected()
    print("[PASS] Invalid backprojection rejected")

    print("=" * 50)
    print("M4 MASTER GATE: PASS")


if __name__ == "__main__":
    main()