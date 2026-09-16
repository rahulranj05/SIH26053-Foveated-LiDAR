import numpy as np

from perception_v1.src.data.canonical import (
    build_canonical_point_frame,
    map_semantic_targets,
)


def test_semantickitti_car_mapping():
    result = map_semantic_targets(
        "SemanticKITTI",
        np.array([10], dtype=np.int32),
    )

    assert result.tolist() == [18]


def test_ignore_mapping():
    result = map_semantic_targets(
        "SemanticKITTI",
        np.array([0], dtype=np.int32),
    )

    assert result.tolist() == [0]


def test_unknown_label_rejected():
    try:
        map_semantic_targets(
            "SemanticKITTI",
            np.array([999999], dtype=np.int32),
        )
    except ValueError:
        return

    raise AssertionError("Unknown native label was not rejected")


def test_label_length_rejected():
    xyz = np.array([
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ], dtype=np.float32)

    intensity = np.array(
        [0.5, 0.6],
        dtype=np.float32,
    )

    labels = np.array(
        [10],
        dtype=np.int32,
    )

    try:
        build_canonical_point_frame(
            "SemanticKITTI",
            xyz,
            intensity,
            labels,
        )
    except ValueError:
        return

    raise AssertionError("Mismatched label length was not rejected")


def test_supervised_frame():
    xyz = np.array([
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ], dtype=np.float32)

    intensity = np.array(
        [0.5, 0.6],
        dtype=np.float32,
    )

    labels = np.array(
        [10, 0],
        dtype=np.int32,
    )

    frame = build_canonical_point_frame(
        "SemanticKITTI",
        xyz,
        intensity,
        labels,
    )

    assert frame.unified_semantic_target.tolist() == [18, 0]
    assert frame.source_point_id.tolist() == [0, 1]


def test_unsupervised_frame():
    xyz = np.array(
        [[1.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    intensity = np.array(
        [0.5],
        dtype=np.float32,
    )

    frame = build_canonical_point_frame(
        "SemanticKITTI",
        xyz,
        intensity,
    )

    assert frame.unified_semantic_target is None


if __name__ == "__main__":
    test_semantickitti_car_mapping()
    test_ignore_mapping()
    test_unknown_label_rejected()
    test_label_length_rejected()
    test_supervised_frame()
    test_unsupervised_frame()

    print("SEMANTIC TARGET TEST: PASS")
