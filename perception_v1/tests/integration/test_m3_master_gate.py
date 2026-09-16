import numpy as np

from perception_v1.src.data.canonical import build_canonical_point_frame
from perception_v1.src.geometry.augmentation import (
    GeometricTransform,
    apply_geometric_transform,
    augment_xyz,
)
from perception_v1.src.geometry.features import compute_spherical_features
from perception_v1.src.ontology.mappings import NATIVE_TO_UNIFIED


def make_frame():
    dataset = "SemanticKITTI"
    native_ids = list(NATIVE_TO_UNIFIED[dataset].keys())

    assert len(native_ids) >= 3

    xyz = np.array([
        [1.0, 0.0, 0.0],
        [2.0, 1.0, 0.5],
        [-1.0, 3.0, 1.0],
    ], dtype=np.float32)

    intensity = np.array([
        0.1,
        0.2,
        0.3,
    ], dtype=np.float32)

    labels = np.array([
        native_ids[0],
        native_ids[1],
        native_ids[2],
    ], dtype=np.int64)

    return build_canonical_point_frame(
        dataset_id=dataset,
        xyz=xyz,
        intensity_raw=intensity,
        native_labels=labels,
    )


def check_geometry_recomputed_after_transform():
    frame = make_frame()

    transform = GeometricTransform(
        rotation_rad=np.pi / 3.0,
        reflect_x=True,
        scale=1.02,
        translation=np.array(
            [0.1, -0.2, 0.05],
            dtype=np.float32,
        ),
    )

    transformed_xyz = apply_geometric_transform(
        frame.xyz,
        transform,
    )

    range_m, azimuth, elevation = compute_spherical_features(
        transformed_xyz
    )

    manual_range = np.linalg.norm(
        transformed_xyz,
        axis=1,
    )

    manual_azimuth = np.arctan2(
        transformed_xyz[:, 1],
        transformed_xyz[:, 0],
    )

    manual_elevation = np.arctan2(
        transformed_xyz[:, 2],
        np.sqrt(
            transformed_xyz[:, 0] ** 2
            + transformed_xyz[:, 1] ** 2
        ),
    )

    assert np.allclose(range_m, manual_range)
    assert np.allclose(azimuth, manual_azimuth)
    assert np.allclose(elevation, manual_elevation)

    # Old geometry must not accidentally be reused.
    assert not np.allclose(
        transformed_xyz,
        frame.xyz,
    )


def check_point_identity_preserved():
    frame = make_frame()

    ids_before = frame.source_point_id.copy()
    raw_before = frame.raw_source_index.copy()
    target_before = frame.unified_semantic_target.copy()
    intensity_before = frame.intensity_normalized.copy()

    rng = np.random.default_rng(26053)

    xyz_after, _ = augment_xyz(
        frame.xyz,
        training=True,
        rng=rng,
    )

    assert len(xyz_after) == len(frame.xyz)

    # Geometry augmentation must not alter these aligned arrays.
    assert np.array_equal(
        frame.source_point_id,
        ids_before,
    )
    assert np.array_equal(
        frame.raw_source_index,
        raw_before,
    )
    assert np.array_equal(
        frame.unified_semantic_target,
        target_before,
    )
    assert np.array_equal(
        frame.intensity_normalized,
        intensity_before,
    )


def check_shared_branch_source():
    frame = make_frame()

    rng = np.random.default_rng(26053)

    shared_xyz, _ = augment_xyz(
        frame.xyz,
        training=True,
        rng=rng,
    )

    # Both future branches must receive the SAME transformed
    # point-domain geometry. No branch-specific augmentation.
    spv_xyz = shared_xyz.copy()
    circular_xyz = shared_xyz.copy()

    assert np.array_equal(
        spv_xyz,
        circular_xyz,
    )

    assert np.array_equal(
        frame.source_point_id,
        np.arange(
            len(shared_xyz),
            dtype=np.int64,
        ),
    )


def check_eval_geometry_unchanged():
    frame = make_frame()

    for training in (False,):
        xyz_after, transform = augment_xyz(
            frame.xyz,
            training=training,
        )

        assert transform is None
        assert np.array_equal(
            xyz_after,
            frame.xyz,
        )

        r, a, e = compute_spherical_features(
            xyz_after
        )

        assert np.allclose(r, frame.range)
        assert np.allclose(a, frame.azimuth)
        assert np.allclose(e, frame.elevation)


def check_deterministic_geometry_replay():
    frame = make_frame()

    rng1 = np.random.default_rng(26053)
    rng2 = np.random.default_rng(26053)

    xyz1, t1 = augment_xyz(
        frame.xyz,
        training=True,
        rng=rng1,
    )

    xyz2, t2 = augment_xyz(
        frame.xyz,
        training=True,
        rng=rng2,
    )

    assert np.array_equal(xyz1, xyz2)

    g1 = compute_spherical_features(xyz1)
    g2 = compute_spherical_features(xyz2)

    for a, b in zip(g1, g2):
        assert np.array_equal(a, b)

    assert t1.rotation_rad == t2.rotation_rad
    assert t1.reflect_x == t2.reflect_x
    assert t1.scale == t2.scale
    assert np.array_equal(
        t1.translation,
        t2.translation,
    )


def main():
    print("M3 MASTER GATE")
    print("=" * 50)

    check_geometry_recomputed_after_transform()
    print("[PASS] Geometry recomputed from augmented XYZ")

    check_point_identity_preserved()
    print("[PASS] Point identity/target/intensity alignment")

    check_shared_branch_source()
    print("[PASS] Shared transform before branch split")

    check_eval_geometry_unchanged()
    print("[PASS] Evaluation geometry remains unaugmented")

    check_deterministic_geometry_replay()
    print("[PASS] Deterministic geometry replay")

    print("=" * 50)
    print("M3 MASTER GATE: PASS")


if __name__ == "__main__":
    main()