import numpy as np

from perception_v1.src.data.batching import (
    collate_whole_frames,
)
from perception_v1.src.data.sample_pipeline import (
    build_unified_sample,
)


def make_sample(
    dataset_id,
    n,
    *,
    split_role="val",
    seed=None,
):
    # Distinct valid points.
    xyz = np.column_stack(
        (
            np.linspace(
                5.0,
                5.0 + n,
                n,
                dtype=np.float32,
            ),
            np.linspace(
                0.1,
                1.0,
                n,
                dtype=np.float32,
            ),
            np.linspace(
                0.2,
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

    source_point_id = np.arange(
        n,
        dtype=np.int64,
    )

    targets = (
        None
        if split_role == "inference"
        else (
            (
                np.arange(n, dtype=np.int64)
                % 24
            ) + 1
        )
    )

    kwargs = {}

    if split_role == "train":
        kwargs["rng"] = (
            np.random.default_rng(seed)
        )

    return build_unified_sample(
        dataset_id=dataset_id,
        xyz=xyz,
        intensity=intensity,
        source_point_id=source_point_id,
        targets=targets,
        split_role=split_role,
        **kwargs,
    )


def make_batch():
    samples = (
        make_sample(
            "SemanticKITTI",
            4,
        ),
        make_sample(
            "RELLIS-3D",
            7,
        ),
        make_sample(
            "nuScenes",
            3,
        ),
    )

    return samples, collate_whole_frames(
        samples
    )


def check_whole_frames_preserved():
    samples, batch = make_batch()

    assert len(batch.samples) == 3

    for original, batched in zip(
        samples,
        batch.samples,
    ):
        assert batched is original

        assert np.array_equal(
            batched.xyz,
            original.xyz,
        )

        assert np.array_equal(
            batched.source_point_id,
            original.source_point_id,
        )


def check_point_accounting():
    samples, batch = make_batch()

    expected = sum(
        len(sample.xyz)
        for sample in samples
    )

    assert batch.total_points == expected

    assert len(
        batch.point_batch_index
    ) == expected

    assert np.array_equal(
        batch.point_offsets,
        np.array(
            [0, 4, 11, 14],
            dtype=np.int64,
        ),
    )


def check_voxel_accounting():
    samples, batch = make_batch()

    expected = sum(
        len(
            sample.spvcnn.voxel_coordinates
        )
        for sample in samples
    )

    assert batch.total_voxels == expected

    assert len(
        batch.voxel_batch_index
    ) == expected

    assert batch.voxel_offsets[0] == 0

    assert (
        batch.voxel_offsets[-1]
        == expected
    )


def check_batch_ownership():
    _, batch = make_batch()

    assert np.array_equal(
        batch.point_batch_index[:4],
        np.zeros(
            4,
            dtype=np.int64,
        ),
    )

    assert np.array_equal(
        batch.point_batch_index[4:11],
        np.ones(
            7,
            dtype=np.int64,
        ),
    )

    assert np.array_equal(
        batch.point_batch_index[11:],
        np.full(
            3,
            2,
            dtype=np.int64,
        ),
    )


def check_mixed_circular_geometry_preserved():
    samples, batch = make_batch()

    assert (
        batch.samples[0]
        .circular
        .image_features
        .shape[0]
        == 64
    )

    assert (
        batch.samples[1]
        .circular
        .image_features
        .shape[0]
        == 64
    )

    assert (
        batch.samples[2]
        .circular
        .image_features
        .shape[0]
        == 32
    )

    # No forced common-height padding.
    assert (
        batch.samples[0]
        .circular
        .image_features
        .shape
        !=
        batch.samples[2]
        .circular
        .image_features
        .shape
    )


def check_no_point_deletion():
    samples, batch = make_batch()

    recovered_counts = [
        int(
            np.sum(
                batch.point_batch_index
                == index
            )
        )
        for index
        in range(len(samples))
    ]

    expected_counts = [
        len(sample.xyz)
        for sample in samples
    ]

    assert (
        recovered_counts
        == expected_counts
    )


def check_deterministic_collation():
    samples, a = make_batch()

    b = collate_whole_frames(
        samples
    )

    assert np.array_equal(
        a.point_batch_index,
        b.point_batch_index,
    )

    assert np.array_equal(
        a.voxel_batch_index,
        b.voxel_batch_index,
    )

    assert np.array_equal(
        a.point_offsets,
        b.point_offsets,
    )

    assert np.array_equal(
        a.voxel_offsets,
        b.voxel_offsets,
    )


def check_inference_batch():
    samples = (
        make_sample(
            "SemanticKITTI",
            4,
            split_role="inference",
        ),
        make_sample(
            "nuScenes",
            3,
            split_role="inference",
        ),
    )

    batch = collate_whole_frames(
        samples
    )

    assert batch.total_points == 7

    for sample in batch.samples:
        assert sample.targets is None

        assert (
            sample.spvcnn.voxel_targets
            is None
        )

        assert (
            sample.circular.image_targets
            is None
        )


def check_empty_batch_rejected():
    try:
        collate_whole_frames(())
    except ValueError:
        return

    raise AssertionError(
        "Empty batch was accepted"
    )


def main():
    print("M6 WHOLE-FRAME BATCHING UNIT GATE")
    print("=" * 50)

    check_whole_frames_preserved()
    print("[PASS] Whole frames preserved")

    check_point_accounting()
    print("[PASS] Point accounting exact")

    check_voxel_accounting()
    print("[PASS] Voxel accounting exact")

    check_batch_ownership()
    print("[PASS] Batch ownership explicit")

    check_mixed_circular_geometry_preserved()
    print("[PASS] Mixed circular geometry preserved")

    check_no_point_deletion()
    print("[PASS] No point deletion or padding")

    check_deterministic_collation()
    print("[PASS] Collation deterministic")

    check_inference_batch()
    print("[PASS] Target-free inference batching")

    check_empty_batch_rejected()
    print("[PASS] Empty batch rejected")

    print("=" * 50)
    print(
        "M6 WHOLE-FRAME BATCHING UNIT GATE: PASS"
    )


if __name__ == "__main__":
    main()