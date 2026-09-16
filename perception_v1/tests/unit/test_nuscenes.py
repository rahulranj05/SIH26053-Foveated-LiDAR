from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from perception_v1.src.data.nuscenes import (
    load_nuscenes_frame,
    load_nuscenes_scan,
)


def test_scan_and_native_ring():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "frame.pcd.bin"

        scan = np.array([
            [1.0, 2.0, 3.0, 255.0, 0.0],
            [2.0, 3.0, 4.0, 128.0, 15.0],
            [3.0, 4.0, 5.0, 64.0, 31.0],
        ], dtype=np.float32)

        scan.tofile(path)

        loaded = load_nuscenes_scan(path)

        assert loaded.xyz.shape == (3, 3)
        assert loaded.intensity.tolist() == [
            255.0,
            128.0,
            64.0,
        ]

        assert loaded.ring.dtype == np.int32
        assert loaded.ring.tolist() == [0, 15, 31]


def test_canonical_frame():
    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        scan_path = tmp / "frame.pcd.bin"
        label_path = tmp / "frame_lidarseg.bin"

        scan = np.array([
            [1.0, 0.0, 0.0, 255.0, 0.0],
            [2.0, 0.0, 0.0, 127.5, 31.0],
        ], dtype=np.float32)

        # Native IDs observed in the real nuScenes frame.
        labels = np.array([
            0,
            2,
        ], dtype=np.uint8)

        scan.tofile(scan_path)
        labels.tofile(label_path)

        frame = load_nuscenes_frame(
            scan_path,
            label_path,
        )

        assert frame.dataset_id == "nuScenes"
        assert len(frame.xyz) == 2
        assert frame.validity_mask.tolist() == [True, True]

        # Frozen normalization: raw / 255.
        assert np.allclose(
            frame.intensity_normalized,
            [1.0, 0.5],
        )

        assert len(frame.unified_semantic_target) == 2


def test_bad_ring_rejected():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad_ring.pcd.bin"

        scan = np.array([
            [1.0, 2.0, 3.0, 100.0, 32.0],
        ], dtype=np.float32)

        scan.tofile(path)

        try:
            load_nuscenes_scan(path)
        except ValueError:
            return

        raise AssertionError(
            "Out-of-range nuScenes ring was not rejected"
        )


def test_noninteger_ring_rejected():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad_ring.pcd.bin"

        scan = np.array([
            [1.0, 2.0, 3.0, 100.0, 4.5],
        ], dtype=np.float32)

        scan.tofile(path)

        try:
            load_nuscenes_scan(path)
        except ValueError:
            return

        raise AssertionError(
            "Non-integer nuScenes ring was not rejected"
        )


def test_bad_scan_rejected():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad.pcd.bin"

        np.array(
            [1.0, 2.0, 3.0, 4.0],
            dtype=np.float32,
        ).tofile(path)

        try:
            load_nuscenes_scan(path)
        except ValueError:
            return

        raise AssertionError(
            "Malformed nuScenes scan was not rejected"
        )


if __name__ == "__main__":
    test_scan_and_native_ring()
    test_canonical_frame()
    test_bad_ring_rejected()
    test_noninteger_ring_rejected()
    test_bad_scan_rejected()

    print("NUSCENES LOADER TEST: PASS")
