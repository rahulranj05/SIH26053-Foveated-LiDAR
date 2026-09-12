import numpy as np

from datasets.semantic_kitti import (
    SemanticKITTIDataset,
)

from datasets.rellis import (
    RELLISDataset,
)

from datasets.semantic_stf import (
    SemanticSTFDataset,
)

from datasets.nuscenes_mini import (
    NuScenesMiniDataset,
)

from datasets.unified_dataset import (
    UnifiedDataset,
)


class DummyDataset:
    def __init__(
        self,
        name: str,
        count: int,
    ):
        self.name = name
        self.count = count

    def __len__(self):
        return self.count

    def __getitem__(
        self,
        index: int,
    ):
        return {
            "xyz": np.zeros(
                (10, 3),
                dtype=np.float32,
            ),
            "intensity": np.zeros(
                10,
                dtype=np.float32,
            ),
            "native_label": np.zeros(
                10,
                dtype=np.uint16,
            ),
            "semantic_label": np.zeros(
                10,
                dtype=np.uint8,
            ),
            "dataset": self.name,
            "frame_id": str(index),
        }


def main():
    print("=" * 70)
    print("DATASET MODULE SMOKE TEST")
    print("=" * 70)

    loaders = [
        SemanticKITTIDataset,
        RELLISDataset,
        SemanticSTFDataset,
        NuScenesMiniDataset,
    ]

    print()
    print("Loader imports:")

    for loader in loaders:
        print(
            f"PASS: {loader.__name__}"
        )

    first = DummyDataset(
        "dummy_a",
        3,
    )

    second = DummyDataset(
        "dummy_b",
        2,
    )

    unified = UnifiedDataset(
        [
            first,
            second,
        ]
    )

    assert len(unified) == 5

    for index in range(
        len(unified)
    ):
        sample = unified[index]

        assert (
            sample["xyz"].shape
            == (10, 3)
        )

        assert (
            len(sample["semantic_label"])
            == 10
        )

    print()
    print(
        "PASS: UnifiedDataset indexing"
    )

    print(
        "PASS: common sample interface"
    )

    print()
    print("=" * 70)
    print(
        "PASS: ALL DATASET MODULE "
        "SMOKE TESTS PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()