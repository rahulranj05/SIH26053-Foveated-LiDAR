from pathlib import Path

import numpy as np

from datasets.paths import (
    get_dataset_roots,
)

from datasets.semantic_kitti import (
    SemanticKITTIDataset,
)


MAPPING_FILE = Path(
    "datasets/mappings/"
    "semantic_kitti.yaml"
)


def test_semantic_kitti_loader():
    roots = get_dataset_roots()

    root = roots[
        "semantic_kitti"
    ]

    print("=" * 80)
    print(
        "SemanticKITTI Loader Test"
    )
    print("=" * 80)

    print(
        f"Dataset root : {root}"
    )

    print(
        f"Mapping file : {MAPPING_FILE}"
    )

    dataset = SemanticKITTIDataset(
        root=root,
        sequences=["00"],
        mapping_file=MAPPING_FILE,
        strict_labels=True,
    )

    print(
        f"Frames found : {len(dataset)}"
    )

    assert len(dataset) == 4541, (
        "Expected 4541 frames in "
        f"sequence 00, found {len(dataset)}"
    )

    sample = dataset[0]

    required_keys = {
        "xyz",
        "intensity",
        "native_label",
        "semantic_label",
        "dataset",
        "sequence",
        "frame_id",
        "scan_path",
        "label_path",
    }

    missing = (
        required_keys
        - set(sample)
    )

    assert not missing, (
        "Missing required keys: "
        f"{sorted(missing)}"
    )

    xyz = np.asarray(
        sample["xyz"]
    )

    intensity = np.asarray(
        sample["intensity"]
    )

    native = np.asarray(
        sample["native_label"]
    )

    semantic = np.asarray(
        sample["semantic_label"]
    )

    n = len(xyz)

    assert xyz.shape == (
        n,
        3,
    )

    assert intensity.shape == (
        n,
    )

    assert native.shape == (
        n,
    )

    assert semantic.shape == (
        n,
    )

    assert np.isfinite(
        xyz
    ).all()

    assert np.isfinite(
        intensity
    ).all()

    unified_ids = set(
        map(
            int,
            np.unique(semantic),
        )
    )

    assert unified_ids.issubset(
        set(range(23))
    )

    assert (
        sample["dataset"]
        == "semantic_kitti"
    )

    assert (
        sample["sequence"]
        == "00"
    )

    assert Path(
        sample["scan_path"]
    ).exists()

    assert Path(
        sample["label_path"]
    ).exists()

    print(
        f"Frame ID     : "
        f"{sample['frame_id']}"
    )

    print(
        f"Points       : {n:,}"
    )

    print(
        "Native IDs   :",
        sorted(
            map(
                int,
                np.unique(native),
            )
        ),
    )

    print(
        "Unified IDs  :",
        sorted(unified_ids),
    )

    print()
    print(
        "PASS: SemanticKITTI "
        "loader real-data test"
    )


if __name__ == "__main__":
    test_semantic_kitti_loader()