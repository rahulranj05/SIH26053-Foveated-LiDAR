from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from perception_v1.src.data.semantic_kitti import (
    decode_semantickitti_labels,
    frame_paths,
    load_semantickitti_frame,
    load_semantickitti_scan,
)


def test_label_decoding():
    semantic_id = 10
    instance_id = 7

    raw = np.array(
        [(instance_id << 16) | semantic_id],
        dtype=np.uint32,
    )

    semantic, instance = decode_semantickitti_labels(raw)

    assert semantic.tolist() == [10]
    assert instance.tolist() == [7]


def test_path_resolution():
    info = frame_paths(
        "/dataset",
        sequence="05",
        frame_id="001762",
        supervised=True,
    )

    assert info.scan_path == Path(
        "/dataset/sequences/05/velodyne/001762.bin"
    )

    assert info.label_path == Path(
        "/dataset/sequences/05/labels/001762.label"
    )


def test_scan_and_frame_loading():
    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        scan_path = tmp / "frame.bin"
        label_path = tmp / "frame.label"

        scan = np.array([
            [1.0, 0.0, 0.0, 0.23],
            [2.0, 1.0, 0.5, 0.50],
            [0.0, 0.0, 0.0, 0.10],
        ], dtype=np.float32)

        scan.tofile(scan_path)

        raw_labels = np.array([
            10,
            0,
            40,
        ], dtype=np.uint32)

        raw_labels.tofile(label_path)

        xyz, intensity = load_semantickitti_scan(scan_path)

        assert xyz.shape == (3, 3)
        assert intensity.shape == (3,)
        assert np.allclose(intensity, [0.23, 0.50, 0.10])

        frame = load_semantickitti_frame(
            scan_path,
            label_path,
        )

        assert frame.dataset_id == "SemanticKITTI"
        # Zero-XYZ raw point is excluded from canonical points.
        assert len(frame.xyz) == 2

        # native 10 -> unified car = 18
        assert frame.unified_semantic_target[0] == 18

        # native 0 -> ignore = 0
        assert frame.unified_semantic_target[1] == 0

        assert frame.validity_mask.tolist() == [
            True,
            True,
        ]

        # IDs assigned after validity filtering.
        assert frame.source_point_id.tolist() == [0, 1]

        # Raw records remain traceable.
        assert frame.raw_source_index.tolist() == [0, 1]


def test_unsupervised_loading():
    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        scan_path = tmp / "frame.bin"

        scan = np.array([
            [1.0, 0.0, 0.0, 0.25],
        ], dtype=np.float32)

        scan.tofile(scan_path)

        frame = load_semantickitti_frame(scan_path)

        assert frame.unified_semantic_target is None


if __name__ == "__main__":
    test_label_decoding()
    test_path_resolution()
    test_scan_and_frame_loading()
    test_unsupervised_loading()

    print("SEMANTICKITTI LOADER TEST: PASS")
