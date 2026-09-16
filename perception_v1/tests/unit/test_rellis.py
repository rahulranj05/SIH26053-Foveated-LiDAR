from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from perception_v1.src.data.rellis import (
    frame_paths,
    load_rellis_frame,
    load_rellis_scan,
)


def test_paths():
    info = frame_paths(
        "/rellis",
        "00000",
        "000000",
    )

    assert info.scan_path == Path(
        "/rellis/00000/"
        "os1_cloud_node_kitti_bin/000000.bin"
    )

    assert info.label_path == Path(
        "/rellis/00000/"
        "os1_cloud_node_semantickitti_label_id/"
        "000000.label"
    )


def test_loader_and_zero_xyz_policy():
    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        scan_path = tmp / "000000.bin"
        label_path = tmp / "000000.label"

        scan = np.array([
            [1.0, 2.0, 3.0, 0.001],
            [0.0, 0.0, 0.0, 0.002],
            [-1.0, 2.0, 0.5, 0.003],
        ], dtype=np.float32)

        # These native IDs are observed in the frozen RELLIS mapping.
        labels = np.array(
            [3, 4, 17],
            dtype=np.uint32,
        )

        scan.tofile(scan_path)
        labels.tofile(label_path)

        xyz, intensity = load_rellis_scan(scan_path)

        assert xyz.shape == (3, 3)
        assert intensity.shape == (3,)

        # Loader must NOT delete zero XYZ.
        assert len(xyz) == 3

        frame = load_rellis_frame(
            scan_path,
            label_path,
        )

        assert frame.dataset_id == "RELLIS-3D"

        # S7-B:
        # finite XYZ AND XYZ != (0,0,0)
        assert frame.validity_mask.tolist() == [
            True,
            False,
            True,
        ]

        # Raw source indexing must remain intact.
        assert frame.source_point_id.tolist() == [
            0,
            1,
            2,
        ]

        assert len(frame.unified_semantic_target) == 3

        # Frozen RELLIS normalization must remain [0,1].
        assert np.all(
            frame.intensity_normalized >= 0.0
        )
        assert np.all(
            frame.intensity_normalized <= 1.0
        )


def test_bad_scan_rejected():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad.bin"

        np.array(
            [1.0, 2.0, 3.0],
            dtype=np.float32,
        ).tofile(path)

        try:
            load_rellis_scan(path)
        except ValueError:
            return

        raise AssertionError(
            "Malformed RELLIS scan was not rejected"
        )


if __name__ == "__main__":
    test_paths()
    test_loader_and_zero_xyz_policy()
    test_bad_scan_rejected()

    print("RELLIS LOADER TEST: PASS")
