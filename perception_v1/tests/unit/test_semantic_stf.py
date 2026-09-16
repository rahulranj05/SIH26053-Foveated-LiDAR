from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from perception_v1.src.data.semantic_stf import (
    frame_paths,
    load_semanticstf_frame,
    load_semanticstf_scan,
)


def test_paths():
    info = frame_paths(
        "/semanticstf",
        "train",
        "2018-02-04_11-09-42_00400",
    )

    assert info.scan_path == Path(
        "/semanticstf/train/velodyne/"
        "2018-02-04_11-09-42_00400.bin"
    )

    assert info.label_path == Path(
        "/semanticstf/train/labels/"
        "2018-02-04_11-09-42_00400.label"
    )


def test_scan_layout_and_unknown_channel():
    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        scan_path = tmp / "frame.bin"

        scan = np.array([
            [1.0, 2.0, 3.0, 255.0, 63.0],
            [2.0, 3.0, 4.0, 128.0, 12.0],
        ], dtype=np.float32)

        scan.tofile(scan_path)

        loaded = load_semanticstf_scan(scan_path)

        assert loaded.xyz.shape == (2, 3)
        assert loaded.intensity.tolist() == [255.0, 128.0]
        assert loaded.unknown_sensor_channel.tolist() == [63.0, 12.0]

        # Critical S7 contract:
        # do not expose this channel as "ring".
        assert not hasattr(loaded, "ring")


def test_canonical_frame():
    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        scan_path = tmp / "frame.bin"
        label_path = tmp / "frame.label"

        scan = np.array([
            [1.0, 0.0, 0.0, 255.0, 63.0],
            [2.0, 0.0, 0.0, 127.5, 10.0],
        ], dtype=np.float32)

        # Native IDs confirmed present in real SemanticSTF.
        labels = np.array([
            10,
            0,
        ], dtype=np.uint32)

        scan.tofile(scan_path)
        labels.tofile(label_path)

        frame = load_semanticstf_frame(
            scan_path,
            label_path,
        )

        assert frame.dataset_id == "SemanticSTF"
        assert len(frame.xyz) == 2
        assert frame.validity_mask.tolist() == [True, True]

        # Frozen STF intensity normalization = raw / 255.
        assert np.allclose(
            frame.intensity_normalized,
            [1.0, 0.5],
        )

        assert len(frame.unified_semantic_target) == 2


def test_bad_scan_rejected():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad.bin"

        np.array(
            [1.0, 2.0, 3.0, 4.0],
            dtype=np.float32,
        ).tofile(path)

        try:
            load_semanticstf_scan(path)
        except ValueError:
            return

        raise AssertionError(
            "Malformed SemanticSTF scan was not rejected"
        )


if __name__ == "__main__":
    test_paths()
    test_scan_layout_and_unknown_channel()
    test_canonical_frame()
    test_bad_scan_rejected()

    print("SEMANTICSTF LOADER TEST: PASS")
