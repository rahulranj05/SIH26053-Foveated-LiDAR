from typing import Dict, List, Sequence


class UnifiedDataset:
    """
    Lightweight wrapper combining multiple dataset loaders.

    Each child dataset must implement:

        __len__()
        __getitem__()

    and return the common fields:

        xyz
        intensity
        native_label
        semantic_label
        dataset
        frame_id
    """

    REQUIRED_KEYS = {
        "xyz",
        "intensity",
        "native_label",
        "semantic_label",
        "dataset",
        "frame_id",
    }

    def __init__(
        self,
        datasets: Sequence,
    ):
        if not datasets:
            raise ValueError(
                "At least one dataset is required."
            )

        self.datasets = list(datasets)

        self.index = []

        for dataset_index, dataset in enumerate(
            self.datasets
        ):
            for sample_index in range(
                len(dataset)
            ):
                self.index.append(
                    (
                        dataset_index,
                        sample_index,
                    )
                )

    def __len__(self):
        return len(self.index)

    def __getitem__(
        self,
        index: int,
    ) -> Dict:

        dataset_index, sample_index = (
            self.index[index]
        )

        sample = self.datasets[
            dataset_index
        ][sample_index]

        missing = (
            self.REQUIRED_KEYS
            - set(sample.keys())
        )

        if missing:
            raise KeyError(
                "Dataset sample is missing required keys: "
                f"{sorted(missing)}"
            )

        return sample

    def summary(self):
        print("=" * 60)
        print("Unified Dataset")
        print("=" * 60)

        print(
            f"Datasets:      {len(self.datasets)}"
        )

        print(
            f"Total samples: {len(self)}"
        )

        for index, dataset in enumerate(
            self.datasets
        ):
            print(
                f"[{index}] "
                f"{dataset.__class__.__name__}: "
                f"{len(dataset)}"
            )

        print("=" * 60)


if __name__ == "__main__":
    print("UnifiedDataset module OK")