import numpy as np

from perception_v1.src.perception.circular.input_builder import (
    build_circular_input,
    lift_pixel_features_to_points,
)


def xyz_from_angles(
    azimuth_deg,
    elevation_deg,
    radius,
):
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)

    horizontal = radius * np.cos(el)

    return np.array([
        horizontal * np.cos(az),
        horizontal * np.sin(az),
        radius * np.sin(el),
    ], dtype=np.float32)


def make_case():
    # 0 and 1 collide in one pixel; point 1 is nearer.
    # 2 is an independent in-FOV winner.
    # 3 is outside SemanticKITTI vertical FOV.
    xyz = np.stack([
        xyz_from_angles(10.0, 0.0, 20.0),
        xyz_from_angles(10.0, 0.0, 5.0),
        xyz_from_angles(90.0, 0.0, 10.0),
        xyz_from_angles(0.0, 20.0, 10.0),
    ])

    features = np.array([
        [10.0, 11.0],
        [20.0, 21.0],
        [30.0, 31.0],
        [40.0, 41.0],
    ], dtype=np.float32)

    targets = np.array(
        [7, 19, 24, 6],
        dtype=np.int64,
    )

    source_ids = np.array(
        [100, 200, 300, 400],
        dtype=np.int64,
    )

    return xyz, features, targets, source_ids


def check_winner_owns_pixel_feature():
    xyz, features, targets, source_ids = make_case()

    result = build_circular_input(
        xyz,
        features,
        dataset_id="SemanticKITTI",
        source_point_id=source_ids,
        point_targets=targets,
    )

    assert not result.winner_mask[0]
    assert result.winner_mask[1]

    row, col = result.point_to_pixel[1]

    assert result.pixel_valid_mask[row, col]

    assert np.array_equal(
        result.image_features[row, col],
        features[1],
    )

    assert result.image_targets[row, col] == 19

    assert (
        result.pixel_to_source_point[row, col]
        == 200
    )


def check_loser_does_not_overwrite_winner():
    xyz, features, targets, source_ids = make_case()

    result = build_circular_input(
        xyz,
        features,
        dataset_id="SemanticKITTI",
        source_point_id=source_ids,
        point_targets=targets,
    )

    row0, col0 = result.point_to_pixel[0]
    row1, col1 = result.point_to_pixel[1]

    assert row0 == row1
    assert col0 == col1

    assert not np.array_equal(
        result.image_features[row0, col0],
        features[0],
    )

    assert np.array_equal(
        result.image_features[row1, col1],
        features[1],
    )


def check_out_of_fov_has_no_pixel():
    xyz, features, targets, source_ids = make_case()

    result = build_circular_input(
        xyz,
        features,
        dataset_id="SemanticKITTI",
        source_point_id=source_ids,
        point_targets=targets,
    )

    assert not result.in_fov_mask[3]
    assert not result.winner_mask[3]

    assert np.array_equal(
        result.point_to_pixel[3],
        np.array([-1, -1]),
    )


def check_point_domain_lift():
    xyz, features, targets, source_ids = make_case()

    result = build_circular_input(
        xyz,
        features,
        dataset_id="SemanticKITTI",
        source_point_id=source_ids,
        point_targets=targets,
    )

    h, w = result.pixel_valid_mask.shape

    pixel_embeddings = np.zeros(
        (h, w, 3),
        dtype=np.float32,
    )

    winner_indices = np.flatnonzero(
        result.winner_mask
    )

    for point_index in winner_indices:
        row, col = result.point_to_pixel[
            point_index
        ]

        pixel_embeddings[
            row, col
        ] = np.array(
            [
                point_index + 1,
                point_index + 10,
                point_index + 100,
            ],
            dtype=np.float32,
        )

    lifted, valid = lift_pixel_features_to_points(
        pixel_embeddings,
        result,
    )

    assert lifted.shape == (4, 3)

    # Point 0 = z-buffer loser.
    assert np.array_equal(
        lifted[0],
        np.zeros(3, dtype=np.float32),
    )
    assert not valid[0]

    # Point 1 = winner.
    assert np.array_equal(
        lifted[1],
        np.array(
            [2.0, 11.0, 101.0],
            dtype=np.float32,
        ),
    )
    assert valid[1]

    # Point 2 = independent winner.
    assert valid[2]

    # Point 3 = out of FOV.
    assert np.array_equal(
        lifted[3],
        np.zeros(3, dtype=np.float32),
    )
    assert not valid[3]


def check_no_winner_feature_copy_to_loser():
    xyz, features, targets, source_ids = make_case()

    result = build_circular_input(
        xyz,
        features,
        dataset_id="SemanticKITTI",
        source_point_id=source_ids,
        point_targets=targets,
    )

    h, w = result.pixel_valid_mask.shape

    pixel_embeddings = np.ones(
        (h, w, 5),
        dtype=np.float32,
    )

    lifted, valid = lift_pixel_features_to_points(
        pixel_embeddings,
        result,
    )

    # Point 0 shares pixel with winner point 1,
    # but MUST NOT receive point 1's circular feature.
    assert not valid[0]
    assert np.all(lifted[0] == 0.0)

    assert valid[1]
    assert np.all(lifted[1] == 1.0)


def check_target_free_inference():
    xyz, features, _, source_ids = make_case()

    result = build_circular_input(
        xyz,
        features,
        dataset_id="SemanticKITTI",
        source_point_id=source_ids,
    )

    assert result.image_targets is None


def check_input_immutability():
    xyz, features, targets, source_ids = make_case()

    xyz_before = xyz.copy()
    features_before = features.copy()
    targets_before = targets.copy()
    ids_before = source_ids.copy()

    build_circular_input(
        xyz,
        features,
        dataset_id="SemanticKITTI",
        source_point_id=source_ids,
        point_targets=targets,
    )

    assert np.array_equal(xyz, xyz_before)
    assert np.array_equal(features, features_before)
    assert np.array_equal(targets, targets_before)
    assert np.array_equal(source_ids, ids_before)


def check_deterministic_build():
    xyz, features, targets, source_ids = make_case()

    a = build_circular_input(
        xyz,
        features,
        dataset_id="SemanticKITTI",
        source_point_id=source_ids,
        point_targets=targets,
    )

    b = build_circular_input(
        xyz,
        features,
        dataset_id="SemanticKITTI",
        source_point_id=source_ids,
        point_targets=targets,
    )

    assert np.array_equal(
        a.image_features,
        b.image_features,
    )
    assert np.array_equal(
        a.image_targets,
        b.image_targets,
    )
    assert np.array_equal(
        a.pixel_valid_mask,
        b.pixel_valid_mask,
    )
    assert np.array_equal(
        a.point_to_pixel,
        b.point_to_pixel,
    )
    assert np.array_equal(
        a.pixel_to_source_point,
        b.pixel_to_source_point,
    )


def check_all_dataset_shapes():
    xyz = np.array(
        [[5.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    features = np.array(
        [[1.0, 2.0]],
        dtype=np.float32,
    )

    expected = {
        "SemanticKITTI": 64,
        "RELLIS-3D": 64,
        "SemanticSTF": 64,
        "nuScenes": 32,
    }

    for dataset, height in expected.items():
        result = build_circular_input(
            xyz,
            features,
            dataset_id=dataset,
        )

        assert result.image_features.shape == (
            height,
            2048,
            2,
        )

        assert result.pixel_valid_mask.shape == (
            height,
            2048,
        )


def main():
    print("M5 MASTER GATE")
    print("=" * 50)

    check_winner_owns_pixel_feature()
    print("[PASS] Z-buffer winner owns pixel")

    check_loser_does_not_overwrite_winner()
    print("[PASS] Z-buffer loser cannot overwrite winner")

    check_out_of_fov_has_no_pixel()
    print("[PASS] Out-of-FOV point excluded from range view")

    check_point_domain_lift()
    print("[PASS] Circular features lift to point domain")

    check_no_winner_feature_copy_to_loser()
    print("[PASS] No winner-feature copying to loser")

    check_target_free_inference()
    print("[PASS] Target-free circular inference")

    check_input_immutability()
    print("[PASS] Canonical inputs remain immutable")

    check_deterministic_build()
    print("[PASS] Deterministic circular-input build")

    check_all_dataset_shapes()
    print("[PASS] Frozen per-dataset range-view dimensions")

    print("=" * 50)
    print("M5 MASTER GATE: PASS")


if __name__ == "__main__":
    main()