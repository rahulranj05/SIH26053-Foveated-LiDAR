import copy

import numpy as np

from perception_v1.src.data.batching import (
    collate_whole_frames,
)
from perception_v1.src.data.sample_pipeline import (
    build_unified_sample,
)
from perception_v1.src.data.sampling import (
    DatasetBalancedSampler,
)
from perception_v1.src.geometry.features import (
    compute_spherical_features,
)


def make_raw_case(dataset_id, n):
    xyz = np.column_stack(
        (
            np.linspace(
                5.0,
                8.0,
                n,
                dtype=np.float32,
            ),
            np.linspace(
                -1.0,
                1.0,
                n,
                dtype=np.float32,
            ),
            np.linspace(
                0.1,
                0.8,
                n,
                dtype=np.float32,
            ),
        )
    ).astype(np.float32)

    intensity = np.linspace(
        0.1,
        0.9,
        n,
        dtype=np.float32,
    )

    source_ids = np.arange(
        n,
        dtype=np.int64,
    )

    targets = (
        np.arange(n, dtype=np.int64)
        % 24
    ) + 1

    return (
        dataset_id,
        xyz,
        intensity,
        source_ids,
        targets,
    )


def build_train(dataset_id, n, seed):
    (
        dataset_id,
        xyz,
        intensity,
        source_ids,
        targets,
    ) = make_raw_case(
        dataset_id,
        n,
    )

    return build_unified_sample(
        dataset_id=dataset_id,
        xyz=xyz,
        intensity=intensity,
        source_point_id=source_ids,
        targets=targets,
        split_role="train",
        rng=np.random.default_rng(seed),
    )


def check_shared_post_augmentation_source():
    sample = build_train(
        "SemanticKITTI",
        9,
        26053,
    )

    expected = compute_spherical_features(
        sample.xyz
    )

    assert np.allclose(
        sample.point_features[:, 0:3],
        sample.xyz,
    )

    assert np.allclose(
        sample.point_features[:, 4],
        expected[0],
    )

    assert np.allclose(
        sample.point_features[:, 5],
        expected[1],
    )

    assert np.allclose(
        sample.point_features[:, 6],
        expected[2],
    )

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


def check_train_replay():
    a = build_train(
        "SemanticKITTI",
        9,
        26053,
    )

    b = build_train(
        "SemanticKITTI",
        9,
        26053,
    )

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
        a.circular.point_to_pixel,
        b.circular.point_to_pixel,
    )


def check_eval_is_geometry_identity():
    (
        dataset_id,
        xyz,
        intensity,
        ids,
        targets,
    ) = make_raw_case(
        "SemanticKITTI",
        8,
    )

    for split in (
        "val",
        "test",
    ):
        sample = build_unified_sample(
            dataset_id=dataset_id,
            xyz=xyz,
            intensity=intensity,
            source_point_id=ids,
            targets=targets,
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


def check_dataset_balanced_sampling_and_resume():
    frames = {
        "SemanticKITTI": tuple(
            f"sk-{i}"
            for i in range(100)
        ),
        "RELLIS-3D": tuple(
            f"re-{i}"
            for i in range(7)
        ),
        "SemanticSTF": tuple(
            f"stf-{i}"
            for i in range(20)
        ),
        "nuScenes": tuple(
            f"nu-{i}"
            for i in range(2)
        ),
    }

    sampler = DatasetBalancedSampler(
        frames,
        seed=26053,
    )

    initial = sampler.sample_many(
        100
    )

    assert len(initial) == 100

    state = copy.deepcopy(
        sampler.state_dict()
    )

    future = sampler.sample_many(
        200
    )

    resumed = DatasetBalancedSampler(
        frames,
        seed=26053,
    )

    resumed.load_state_dict(
        state
    )

    assert (
        resumed.sample_many(200)
        == future
    )

    balance_sampler = (
        DatasetBalancedSampler(
            frames,
            seed=26053,
        )
    )

    draws = balance_sampler.sample_many(
        40000
    )

    counts = {
        name: 0
        for name in frames
    }

    for draw in draws:
        counts[draw.dataset_id] += 1

    for count in counts.values():
        assert abs(
            count - 10000
        ) < 700


def check_whole_frame_batch():
    samples = (
        build_train(
            "SemanticKITTI",
            5,
            26053,
        ),
        build_train(
            "RELLIS-3D",
            8,
            26054,
        ),
        build_train(
            "nuScenes",
            3,
            26055,
        ),
    )

    batch = collate_whole_frames(
        samples
    )

    assert batch.total_points == 16

    assert np.array_equal(
        batch.point_offsets,
        np.array(
            [0, 5, 13, 16],
            dtype=np.int64,
        ),
    )

    for index, sample in enumerate(
        samples
    ):
        assert (
            np.sum(
                batch.point_batch_index
                == index
            )
            == len(sample.xyz)
        )

        assert (
            batch.samples[index]
            is sample
        )

    # Dataset-specific range-view geometry survives.
    assert (
        batch.samples[0]
        .circular.image_features.shape[0]
        == 64
    )

    assert (
        batch.samples[2]
        .circular.image_features.shape[0]
        == 32
    )


def check_point_identity_survives_pipeline():
    sample = build_train(
        "SemanticKITTI",
        11,
        26053,
    )

    expected = np.arange(
        11,
        dtype=np.int64,
    )

    assert np.array_equal(
        sample.source_point_id,
        expected,
    )

    assert np.array_equal(
        sample.circular.source_point_id,
        expected,
    )

    assert len(
        sample.spvcnn.point_to_voxel_inverse
    ) == 11


def check_target_free_inference():
    (
        dataset_id,
        xyz,
        intensity,
        ids,
        _,
    ) = make_raw_case(
        "nuScenes",
        6,
    )

    sample = build_unified_sample(
        dataset_id=dataset_id,
        xyz=xyz,
        intensity=intensity,
        source_point_id=ids,
        targets=None,
        split_role="inference",
    )

    assert sample.targets is None
    assert sample.spvcnn.voxel_targets is None
    assert sample.circular.image_targets is None


def main():
    print("M6 MASTER GATE")
    print("=" * 50)

    check_shared_post_augmentation_source()
    print(
        "[PASS] Shared post-augmentation branch source"
    )

    check_train_replay()
    print(
        "[PASS] Deterministic seeded preprocessing replay"
    )

    check_eval_is_geometry_identity()
    print(
        "[PASS] Validation/test augmentation disabled"
    )

    check_dataset_balanced_sampling_and_resume()
    print(
        "[PASS] Dataset-balanced sampling + exact resume"
    )

    check_whole_frame_batch()
    print(
        "[PASS] Variable-length whole-frame batching"
    )

    check_point_identity_survives_pipeline()
    print(
        "[PASS] Canonical point identity preserved"
    )

    check_target_free_inference()
    print(
        "[PASS] Target-free inference path"
    )

    print("=" * 50)
    print("M6 MASTER GATE: PASS")


if __name__ == "__main__":
    main()