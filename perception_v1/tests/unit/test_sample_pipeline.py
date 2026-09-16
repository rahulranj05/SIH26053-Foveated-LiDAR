import numpy as np

from perception_v1.src.data.sample_pipeline import (
    build_unified_sample,
)
from perception_v1.src.geometry.features import (
    compute_spherical_features,
)


def make_case():
    xyz = np.array(
        [
            [5.00, 0.00, 0.10],
            [5.10, 0.10, 0.20],
            [6.00, 1.00, -0.20],
            [7.00, -1.00, 0.30],
        ],
        dtype=np.float32,
    )

    intensity = np.array(
        [0.1, 0.2, 0.3, 0.4],
        dtype=np.float32,
    )

    source_ids = np.array(
        [0, 1, 2, 3],
        dtype=np.int64,
    )

    targets = np.array(
        [1, 18, 24, 10],
        dtype=np.int64,
    )

    return (
        xyz,
        intensity,
        source_ids,
        targets,
    )


def build_train(seed):
    xyz, intensity, ids, targets = make_case()

    return build_unified_sample(
        dataset_id="SemanticKITTI",
        xyz=xyz,
        intensity=intensity,
        source_point_id=ids,
        targets=targets,
        split_role="train",
        rng=np.random.default_rng(seed),
    )


def check_training_requires_rng():
    xyz, intensity, ids, targets = make_case()

    try:
        build_unified_sample(
            dataset_id="SemanticKITTI",
            xyz=xyz,
            intensity=intensity,
            source_point_id=ids,
            targets=targets,
            split_role="train",
        )
    except ValueError:
        return

    raise AssertionError(
        "Training accepted without explicit RNG"
    )


def check_seeded_replay():
    a = build_train(26053)
    b = build_train(26053)

    assert np.array_equal(
        a.xyz,
        b.xyz,
    )

    assert np.array_equal(
        a.point_features,
        b.point_features,
    )

    assert np.array_equal(
        a.spvcnn.voxel_coordinates,
        b.spvcnn.voxel_coordinates,
    )

    assert np.array_equal(
        a.spvcnn.point_to_voxel_inverse,
        b.spvcnn.point_to_voxel_inverse,
    )

    assert np.array_equal(
        a.circular.point_to_pixel,
        b.circular.point_to_pixel,
    )

    assert np.array_equal(
        a.circular.pixel_to_source_point,
        b.circular.pixel_to_source_point,
    )


def check_different_seed():
    a = build_train(26053)
    b = build_train(26054)

    assert not np.array_equal(
        a.xyz,
        b.xyz,
    )


def check_train_transform_recorded():
    sample = build_train(26053)

    assert (
        sample.augmentation_transform
        is not None
    )


def check_eval_no_augmentation():
    xyz, intensity, ids, targets = make_case()

    for split in (
        "val",
        "test",
        "inference",
    ):
        sample = build_unified_sample(
            dataset_id="SemanticKITTI",
            xyz=xyz,
            intensity=intensity,
            source_point_id=ids,
            targets=(
                None
                if split == "inference"
                else targets
            ),
            split_role=split,
        )

        assert np.array_equal(
            sample.xyz,
            xyz,
        )

        assert (
            sample.augmentation_transform
            is None
        )


def check_geometry_recomputed():
    sample = build_train(26053)

    expected = compute_spherical_features(
        sample.xyz
    )

    assert np.allclose(
        sample.range,
        expected[0],
    )

    assert np.allclose(
        sample.azimuth,
        expected[1],
    )

    assert np.allclose(
        sample.elevation,
        expected[2],
    )

    assert np.allclose(
        sample.point_features[:, 4],
        sample.range,
    )

    assert np.allclose(
        sample.point_features[:, 5],
        sample.azimuth,
    )

    assert np.allclose(
        sample.point_features[:, 6],
        sample.elevation,
    )


def check_shared_branch_source():
    sample = build_train(26053)

    reps = (
        sample.spvcnn
        .voxel_to_representative_point
    )

    assert np.array_equal(
        sample.spvcnn.voxel_features,
        sample.point_features[reps],
    )

    winners = np.flatnonzero(
        sample.circular.winner_mask
    )

    for point_index in winners:
        row, col = (
            sample.circular
            .point_to_pixel[point_index]
        )

        assert np.array_equal(
            sample.circular.image_features[
                row,
                col,
            ],
            sample.point_features[
                point_index
            ],
        )


def check_identity_preserved():
    xyz, intensity, ids, targets = make_case()

    sample = build_train(26053)

    assert len(sample.xyz) == len(xyz)

    assert np.array_equal(
        sample.source_point_id,
        ids,
    )

    assert np.array_equal(
        sample.targets,
        targets,
    )

    assert np.array_equal(
        sample.intensity,
        intensity,
    )


def check_inputs_immutable():
    xyz, intensity, ids, targets = make_case()

    xyz_before = xyz.copy()
    intensity_before = intensity.copy()
    ids_before = ids.copy()
    targets_before = targets.copy()

    build_unified_sample(
        dataset_id="SemanticKITTI",
        xyz=xyz,
        intensity=intensity,
        source_point_id=ids,
        targets=targets,
        split_role="train",
        rng=np.random.default_rng(26053),
    )

    assert np.array_equal(
        xyz,
        xyz_before,
    )

    assert np.array_equal(
        intensity,
        intensity_before,
    )

    assert np.array_equal(
        ids,
        ids_before,
    )

    assert np.array_equal(
        targets,
        targets_before,
    )


def check_all_datasets():
    xyz, intensity, ids, targets = make_case()

    expected_height = {
        "SemanticKITTI": 64,
        "RELLIS-3D": 64,
        "SemanticSTF": 64,
        "nuScenes": 32,
    }

    for dataset_id, height in (
        expected_height.items()
    ):
        sample = build_unified_sample(
            dataset_id=dataset_id,
            xyz=xyz,
            intensity=intensity,
            source_point_id=ids,
            targets=targets,
            split_role="val",
        )

        assert (
            sample.circular
            .image_features
            .shape[0]
            == height
        )

        assert (
            sample.circular
            .image_features
            .shape[1]
            == 2048
        )


def check_invalid_inputs():
    xyz, intensity, ids, targets = make_case()

    bad_xyz = xyz.copy()
    bad_xyz[0] = 0.0

    try:
        build_unified_sample(
            dataset_id="SemanticKITTI",
            xyz=bad_xyz,
            intensity=intensity,
            source_point_id=ids,
            targets=targets,
            split_role="val",
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Zero canonical XYZ accepted"
        )

    try:
        build_unified_sample(
            dataset_id="SemanticKITTI",
            xyz=xyz,
            intensity=intensity[:-1],
            source_point_id=ids,
            targets=targets,
            split_role="val",
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Misaligned intensity accepted"
        )

    try:
        build_unified_sample(
            dataset_id="SemanticKITTI",
            xyz=xyz,
            intensity=intensity,
            source_point_id=ids,
            targets=targets,
            split_role="banana",
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Invalid split accepted"
        )


def main():
    print("M6 UNIFIED SAMPLE UNIT GATE")
    print("=" * 50)

    check_training_requires_rng()
    print("[PASS] Training requires explicit RNG")

    check_seeded_replay()
    print("[PASS] Seeded training replay deterministic")

    check_different_seed()
    print("[PASS] Different seed changes training geometry")

    check_train_transform_recorded()
    print("[PASS] Training transform recorded")

    check_eval_no_augmentation()
    print("[PASS] Val/test/inference unaugmented")

    check_geometry_recomputed()
    print("[PASS] Geometry recomputed after augmentation")

    check_shared_branch_source()
    print("[PASS] Both branches share one canonical source")

    check_identity_preserved()
    print("[PASS] Point identity/targets/intensity preserved")

    check_inputs_immutable()
    print("[PASS] Input arrays remain immutable")

    check_all_datasets()
    print("[PASS] All four dataset projection routes")

    check_invalid_inputs()
    print("[PASS] Invalid unified inputs rejected")

    print("=" * 50)
    print("M6 UNIFIED SAMPLE UNIT GATE: PASS")


if __name__ == "__main__":
    main()