from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml


class SemanticSTFDataset:
    """
    SemanticSTF LiDAR semantic segmentation loader.

    SemanticSTF point clouds contain five float32 values per point.
    """

    def __init__(
        self,
        root: str,
        mapping_file: str = "datasets/mappings/semantic_stf.yaml",
    ):
        self.root = Path(root)
        self.mapping_file = Path(mapping_file)

        self.mapping = self._load_mapping()
        self.lookup_table = self._build_lookup_table()
        self.samples = self._discover_samples()

        if not self.samples:
            raise RuntimeError(
                f"No SemanticSTF samples found under: {self.root}"
            )

    def _load_mapping(self) -> Dict[int, int]:
        if not self.mapping_file.exists():
            raise FileNotFoundError(
                f"Mapping file not found: {self.mapping_file}"
            )

        with open(
            self.mapping_file,
            "r",
            encoding="utf-8",
        ) as file:
            config = yaml.safe_load(file)

        if "mapping" not in config:
            raise KeyError(
                f"'mapping' missing from {self.mapping_file}"
            )

        return {
            int(native_id): int(unified_id)
            for native_id, unified_id
            in config["mapping"].items()
        }

    def _build_lookup_table(self) -> np.ndarray:
        maximum = max(self.mapping)

        lookup = np.zeros(
            maximum + 1,
            dtype=np.uint8,
        )

        for native_id, unified_id in self.mapping.items():
            lookup[native_id] = unified_id

        return lookup

    def _discover_samples(self) -> List[Dict]:
        samples = []

        bin_files = sorted(
            self.root.rglob("*.bin")
        )

        label_files = sorted(
            self.root.rglob("*.label")
        )

        label_index = {
            path.stem: path
            for path in label_files
        }

        for scan_path in bin_files:
            if "label" in str(scan_path).lower():
                continue

            label_path = label_index.get(
                scan_path.stem
            )

            if label_path is None:
                continue

            split = "unknown"

            lower_path = str(scan_path).lower()

            for candidate in (
                "train",
                "val",
                "test",
            ):
                if candidate in lower_path:
                    split = candidate
                    break

            samples.append(
                {
                    "scan_path": scan_path,
                    "label_path": label_path,
                    "frame_id": scan_path.stem,
                    "split": split,
                }
            )

        return samples

    @staticmethod
    def _load_scan(
        scan_path: Path,
    ) -> np.ndarray:

        raw = np.fromfile(
            scan_path,
            dtype=np.float32,
        )

        if raw.size % 5 != 0:
            raise ValueError(
                f"SemanticSTF scan is not Nx5: {scan_path}"
            )

        return raw.reshape(-1, 5)

    @staticmethod
    def _load_native_labels(
        label_path: Path,
    ) -> np.ndarray:

        return np.fromfile(
            label_path,
            dtype=np.uint32,
        )

    def _remap_labels(
        self,
        native_labels: np.ndarray,
    ) -> np.ndarray:

        unified = np.zeros(
            native_labels.shape,
            dtype=np.uint8,
        )

        valid = (
            native_labels
            < len(self.lookup_table)
        )

        unified[valid] = self.lookup_table[
            native_labels[valid]
        ]

        return unified

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]

        scan = self._load_scan(
            sample["scan_path"]
        )

        native_labels = self._load_native_labels(
            sample["label_path"]
        )

        if len(scan) != len(native_labels):
            raise ValueError(
                "Point/label count mismatch\n"
                f"Points: {len(scan)}\n"
                f"Labels: {len(native_labels)}"
            )

        return {
            "xyz": scan[:, :3].astype(
                np.float32,
                copy=False,
            ),
            "intensity": scan[:, 3].astype(
                np.float32,
                copy=False,
            ),
            "native_label": native_labels,
            "semantic_label": self._remap_labels(
                native_labels
            ),
            "dataset": "semantic_stf",
            "split": sample["split"],
            "frame_id": sample["frame_id"],
            "scan_path": str(sample["scan_path"]),
            "label_path": str(sample["label_path"]),
        }


if __name__ == "__main__":
    print("SemanticSTFDataset module OK")