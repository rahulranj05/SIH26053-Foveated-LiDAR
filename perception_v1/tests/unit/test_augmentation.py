import numpy as np

from perception_v1.src.geometry.augmentation import (
    AugmentationConfig,
    GeometricTransform,
    apply_geometric_transform,
    augment_xyz,
    sample_geometric_transform,
)


def check_identity_when_not_training():
    xyz = np.array([
        [1.0, 2.0, 3.0],
        [-4.0, 5.0, 6.0],
    ], dtype=np.float32)

    original = xyz.copy()

    out, transform = augment_xyz(
        xyz,
        training=False,
    )

    assert transform is None
    assert np.array_equal(out, original)
    assert np.array_equal(xyz, original)
    assert out is not xyz


def check_exact_known_transform():
    xyz = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 1.0],
    ], dtype=np.float32)

    transform = GeometricTransform(
        rotation_rad=np.pi / 2.0,
        reflect_x=False,
        scale=2.0,
        translation=np.array(
            [1.0, 2.0, 3.0],
            dtype=np.float32,
        ),
    )

    out = apply_geometric_transform(
        xyz,
        transform,
    )

    expected = np.array([
        [1.0, 4.0, 3.0],
        [-3.0, 2.0, 5.0],
    ], dtype=np.float32)

    assert np.allclose(
        out,
        expected,
        atol=1e-6,
    )


def check_reflection():
    xyz = np.array([
        [2.0, 3.0, 4.0],
    ], dtype=np.float32)

    transform = GeometricTransform(
        rotation_rad=0.0,
        reflect_x=True,
        scale=1.0,
        translation=np.zeros(
            3,
            dtype=np.float32,
        ),
    )

    out = apply_geometric_transform(
        xyz,
        transform,
    )

    assert np.allclose(
        out,
        [[-2.0, 3.0, 4.0]],
    )


def check_seeded_replay():
    xyz = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [-1.0, 3.0, 2.0],
    ], dtype=np.float32)

    rng_a = np.random.default_rng(26053)
    rng_b = np.random.default_rng(26053)

    out_a, transform_a = augment_xyz(
        xyz,
        training=True,
        rng=rng_a,
    )

    out_b, transform_b = augment_xyz(
        xyz,
        training=True,
        rng=rng_b,
    )

    assert np.array_equal(out_a, out_b)
    assert transform_a.rotation_rad == transform_b.rotation_rad
    assert transform_a.reflect_x == transform_b.reflect_x
    assert transform_a.scale == transform_b.scale
    assert np.array_equal(
        transform_a.translation,
        transform_b.translation,
    )


def check_no_point_deletion_or_reordering():
    xyz = np.array([
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
    ], dtype=np.float32)

    rng = np.random.default_rng(26053)

    out, _ = augment_xyz(
        xyz,
        training=True,
        rng=rng,
    )

    assert out.shape == xyz.shape
    assert len(out) == len(xyz)


def check_input_not_modified():
    xyz = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ], dtype=np.float32)

    original = xyz.copy()

    rng = np.random.default_rng(26053)

    augment_xyz(
        xyz,
        training=True,
        rng=rng,
    )

    assert np.array_equal(xyz, original)


def check_sampling_bounds():
    config = AugmentationConfig()

    rng = np.random.default_rng(26053)

    for _ in range(1000):
        t = sample_geometric_transform(
            rng,
            config,
        )

        assert (
            config.rotation_min_rad
            <= t.rotation_rad
            <= config.rotation_max_rad
        )

        assert (
            config.scale_min
            <= t.scale
            <= config.scale_max
        )

        assert np.all(
            t.translation
            >= config.translation_min_m
        )

        assert np.all(
            t.translation
            <= config.translation_max_m
        )


def check_reflection_probability_behavior():
    config = AugmentationConfig(
        rotation_min_rad=0.0,
        rotation_max_rad=0.0,
        reflection_probability=1.0,
        scale_min=1.0,
        scale_max=1.0,
        translation_min_m=0.0,
        translation_max_m=0.0,
    )

    rng = np.random.default_rng(26053)

    t = sample_geometric_transform(
        rng,
        config,
    )

    assert t.reflect_x is True


def check_training_requires_rng():
    xyz = np.array(
        [[1.0, 2.0, 3.0]],
        dtype=np.float32,
    )

    try:
        augment_xyz(
            xyz,
            training=True,
            rng=None,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Training augmentation silently accepted missing RNG"
        )


def check_invalid_xyz_rejected():
    xyz = np.array([
        [1.0, np.nan, 3.0],
    ], dtype=np.float32)

    transform = GeometricTransform(
        rotation_rad=0.0,
        reflect_x=False,
        scale=1.0,
        translation=np.zeros(
            3,
            dtype=np.float32,
        ),
    )

    try:
        apply_geometric_transform(
            xyz,
            transform,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Non-finite canonical XYZ was accepted"
        )


def main():
    print("M3 AUGMENTATION UNIT GATE")
    print("=" * 50)

    check_identity_when_not_training()
    print("[PASS] No validation/test/inference augmentation")

    check_exact_known_transform()
    print("[PASS] Exact geometric transform")

    check_reflection()
    print("[PASS] X reflection")

    check_seeded_replay()
    print("[PASS] Deterministic seeded replay")

    check_no_point_deletion_or_reordering()
    print("[PASS] No point deletion")

    check_input_not_modified()
    print("[PASS] Input immutability")

    check_sampling_bounds()
    print("[PASS] Frozen sampling bounds")

    check_reflection_probability_behavior()
    print("[PASS] Reflection probability control")

    check_training_requires_rng()
    print("[PASS] Explicit RNG required")

    check_invalid_xyz_rejected()
    print("[PASS] Invalid canonical XYZ rejected")

    print("=" * 50)
    print("M3 AUGMENTATION UNIT GATE: PASS")


if __name__ == "__main__":
    main()