import numpy as np

from perception_v1.src.data.canonical import (
    build_canonical_point_frame,
    map_semantic_targets,
)
from perception_v1.src.data.normalization import normalize_intensity
from perception_v1.src.geometry.features import compute_validity_mask
from perception_v1.src.ontology.mappings import NATIVE_TO_UNIFIED


DATASETS = (
    "SemanticKITTI",
    "RELLIS-3D",
    "SemanticSTF",
    "nuScenes",
)


def check_validity_contract():
    xyz = np.array([
        [1.0, 2.0, 3.0],          # valid
        [0.0, 0.0, 0.0],          # invalid zero XYZ
        [np.nan, 1.0, 2.0],       # invalid NaN
        [1.0, np.inf, 2.0],       # invalid +inf
        [1.0, 2.0, -np.inf],      # invalid -inf
        [0.0, 0.0, 1.0],          # valid
        [-1.0, 0.0, 0.0],         # valid
    ], dtype=np.float32)

    mask = compute_validity_mask(xyz)

    assert mask.dtype == np.bool_
    assert mask.tolist() == [
        True,
        False,
        False,
        False,
        False,
        True,
        True,
    ]


def check_post_validity_identity_contract():
    dataset = "SemanticKITTI"

    xyz = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [np.nan, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ], dtype=np.float32)

    intensity = np.array(
        [0.1, 0.2, 0.3, 0.4, 0.5],
        dtype=np.float32,
    )

    native = next(iter(NATIVE_TO_UNIFIED[dataset].keys()))
    labels = np.full(5, native, dtype=np.int64)

    frame = build_canonical_point_frame(
        dataset,
        xyz,
        intensity,
        labels,
    )

    assert frame.raw_source_index.tolist() == [0, 2, 4]
    assert frame.source_point_id.tolist() == [0, 1, 2]
    assert frame.validity_mask.tolist() == [True, True, True]

    assert np.allclose(
        frame.intensity_raw,
        [0.1, 0.3, 0.5],
    )


def check_target_contract():
    for dataset in DATASETS:
        mapping = NATIVE_TO_UNIFIED[dataset]

        native_ids = np.array(
            list(mapping.keys()),
            dtype=np.int64,
        )

        expected = np.array(
            [mapping[int(x)] for x in native_ids],
            dtype=np.int64,
        )

        actual = map_semantic_targets(
            dataset,
            native_ids,
        )

        assert np.array_equal(actual, expected)
        assert np.all((actual >= 0) & (actual <= 24))

    # Unknown labels MUST fail, never silently become ignore.
    dataset = "SemanticKITTI"
    mapping = NATIVE_TO_UNIFIED[dataset]

    unknown = max(mapping.keys()) + 100000

    try:
        map_semantic_targets(
            dataset,
            np.array([unknown], dtype=np.int64),
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Unknown native semantic ID was silently accepted"
        )

    # Unknown dataset MUST fail.
    try:
        map_semantic_targets(
            "NOT_A_DATASET",
            np.array([0], dtype=np.int64),
        )
    except KeyError:
        pass
    else:
        raise AssertionError(
            "Unknown dataset was silently accepted"
        )


def check_normalization_contract():
    # SemanticKITTI: identity.
    x = np.array(
        [0.0, 0.25, 0.99],
        dtype=np.float32,
    )

    y = normalize_intensity("SemanticKITTI", x)

    assert np.allclose(y, x)


    # SemanticSTF: raw / 255.
    x = np.array(
        [0.0, 127.5, 255.0],
        dtype=np.float32,
    )

    y = normalize_intensity("SemanticSTF", x)

    assert np.allclose(
        y,
        [0.0, 0.5, 1.0],
    )


    # nuScenes: raw / 255.
    y = normalize_intensity("nuScenes", x)

    assert np.allclose(
        y,
        [0.0, 0.5, 1.0],
    )


    # RELLIS frozen p1/p99.
    p1 = 0.0002594033721834421
    p99 = 0.00869764294475317
    mid = (p1 + p99) / 2.0

    x = np.array([
        p1 - 1.0,
        p1,
        mid,
        p99,
        p99 + 1.0,
    ], dtype=np.float32)

    y = normalize_intensity("RELLIS-3D", x)

    assert np.allclose(
        y,
        [0.0, 0.0, 0.5, 1.0, 1.0],
        atol=1e-5,
    )

    # Input must not be modified in-place.
    original = np.array(
        [0.1, 0.2, 0.3],
        dtype=np.float32,
    )
    preserved = original.copy()

    normalize_intensity(
        "SemanticKITTI",
        original,
    )

    assert np.array_equal(original, preserved)


def check_alignment_after_filtering():
    dataset = "SemanticKITTI"
    mapping = NATIVE_TO_UNIFIED[dataset]

    native_ids = list(mapping.keys())

    assert len(native_ids) >= 2

    a = native_ids[0]
    b = native_ids[1]

    xyz = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ], dtype=np.float32)

    intensity = np.array(
        [0.11, 0.22, 0.33],
        dtype=np.float32,
    )

    labels = np.array(
        [a, a, b],
        dtype=np.int64,
    )

    frame = build_canonical_point_frame(
        dataset,
        xyz,
        intensity,
        labels,
    )

    assert len(frame.xyz) == 2
    assert frame.raw_source_index.tolist() == [0, 2]

    assert np.allclose(
        frame.intensity_raw,
        [0.11, 0.33],
    )

    assert frame.unified_semantic_target.tolist() == [
        mapping[a],
        mapping[b],
    ]


def main():
    print("M2 MASTER GATE")
    print("=" * 50)

    check_validity_contract()
    print("[PASS] Validity policy")

    check_post_validity_identity_contract()
    print("[PASS] Post-validity canonical identity")

    check_target_contract()
    print("[PASS] Frozen semantic target mapping")

    check_normalization_contract()
    print("[PASS] Frozen intensity normalization")

    check_alignment_after_filtering()
    print("[PASS] Point/intensity/target alignment")

    print("=" * 50)
    print("M2 MASTER GATE: PASS")


if __name__ == "__main__":
    main()