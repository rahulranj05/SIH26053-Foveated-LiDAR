from pathlib import Path
from typing import Dict, List
import json

import numpy as np
import yaml


class NuScenesMiniDataset:
    """
    nuScenes mini LiDAR segmentation loader.

    Uses nuScenes metadata to associate lidarseg labels with LiDAR samples.
    """

    def __init__(
        self,
        root: str,
        mapping_file: str = "datasets/mappings/nuscenes.yaml",
    ):
        self.root = Path(root)
        self.mapping_file = Path(mapping_file)

        self.mapping = self._load_mapping()
        self.lookup_table = self._build_lookup_table()

        self.samples = self._discover_samples()

        if not self.samples:
            raise RuntimeError(
                f"No nuScenes lidarseg samples found under: {self.root}"
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

    def _find_metadata_directory(self) -> Path:
        candidates = list(
            self.root.rglob("v1.0-mini")
        )

        for candidate in candidates:
            if (
                candidate.is_dir()
                and (candidate / "sample_data.json").exists()
            ):
                return candidate

        raise FileNotFoundError(
            "Could not find nuScenes v1.0-mini metadata."
        )

    def _discover_samples(self) -> List[Dict]:
        metadata_dir = self._find_metadata_directory()

        sample_data_file = (
            metadata_dir / "sample_data.json"
        )

        lidarseg_file = (
            metadata_dir / "lidarseg.json"
        )

        if not lidarseg_file.exists():
            raise FileNotFoundError(
                f"Missing lidarseg.json: {lidarseg_file}"
            )

        with open(
            sample_data_file,
            "r",
            encoding="utf-8",
        ) as file:
            sample_data = json.load(file)

        with open(
            lidarseg_file,
            "r",
            encoding="utf-8",
        ) as file:
            lidarseg = json.load(file)

        sample_index = {
            item["token"]: item
            for item in sample_data
        }

        samples = []

        for item in lidarseg:
            token = item[
                "sample_data_token"
            ]

            sample_info = sample_index.get(token)

            if sample_info is None:
                continue

            scan_path = (
                self.root
                / sample_info["filename"]
            )

            label_path = (
                self.root
                / item["filename"]
            )

            if not scan_path.exists():
                continue

            if not label_path.exists():
                continue

            samples.append(
                {
                    "token": token,
                    "scan_path": scan_path,
                    "label_path": label_path,
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
                f"nuScenes scan is not Nx5: {scan_path}"
            )

        return raw.reshape(-1, 5)

    @staticmethod
    def _load_native_labels(
        label_path: Path,
    ) -> np.ndarray:

        return np.fromfile(
            label_path,
            dtype=np.uint8,
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

    def __getitem__(
        self,
        index: int,
    ) -> Dict:

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
            "dataset": "nuscenes_mini",
            "frame_id": sample["token"],
            "scan_path": str(sample["scan_path"]),
            "label_path": str(sample["label_path"]),
        }


if __name__ == "__main__":
    print("NuScenesMiniDataset module OK")