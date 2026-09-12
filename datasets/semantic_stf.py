from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import yaml


class SemanticSTFDataset:
    """
    SemanticSTF LiDAR segmentation loader.

    Canonical SIH26053 top-level folder:

        .../datasets/SemanticSTF

    Frames are paired independently inside train/val/test
    so identical frame stems across splits can never collide.
    """

    DEFAULT_SPLITS = (
        "train",
        "val",
        "test",
    )

    def __init__(
        self,
        root,
        mapping_file="datasets/mappings/semantic_stf.yaml",
        splits: Optional[Sequence[str]] = None,
        strict_labels: bool = True,
    ):
        self.root = Path(root)
        self.mapping_file = Path(mapping_file)
        self.strict_labels = strict_labels

        if not self.root.exists():
            raise FileNotFoundError(
                f"SemanticSTF root not found: {self.root}"
            )

        if splits is None:
            self.splits = list(
                self.DEFAULT_SPLITS
            )
        else:
            self.splits = [
                str(split).lower()
                for split in splits
            ]

        invalid = (
            set(self.splits)
            - set(self.DEFAULT_SPLITS)
        )

        if invalid:
            raise ValueError(
                "Unknown SemanticSTF split(s): "
                f"{sorted(invalid)}"
            )

        self.mapping = self._load_mapping()
        self.lookup_table = self._build_lookup_table()
        self.samples = self._discover_samples()

        if not self.samples:
            raise RuntimeError(
                "No SemanticSTF samples found under:\n"
                f"{self.root}"
            )

    def _load_mapping(self):
        if not self.mapping_file.exists():
            raise FileNotFoundError(
                f"Mapping not found: {self.mapping_file}"
            )

        with open(
            self.mapping_file,
            "r",
            encoding="utf-8",
        ) as file:
            config = yaml.safe_load(file)

        mapping = config.get("mapping")

        if not mapping:
            raise RuntimeError(
                f"Empty SemanticSTF mapping: {self.mapping_file}"
            )

        return {
            int(native): int(unified)
            for native, unified in mapping.items()
        }

    def _build_lookup_table(self):
        maximum = max(self.mapping)

        lookup = np.zeros(
            maximum + 1,
            dtype=np.uint8,
        )

        for native, unified in self.mapping.items():
            lookup[native] = unified

        return lookup

    def _find_split_root(
        self,
        split: str,
    ) -> Path:

        direct = (
            self.root
            / split
        )

        if direct.is_dir():
            return direct

        candidates = [
            path
            for path in self.root.rglob(split)
            if path.is_dir()
        ]

        if len(candidates) == 1:
            return candidates[0]

        if not candidates:
            raise FileNotFoundError(
                f"SemanticSTF split '{split}' not found under "
                f"{self.root}"
            )

        raise RuntimeError(
            f"Multiple SemanticSTF '{split}' directories found:\n"
            + "\n".join(
                str(path)
                for path in candidates
            )
        )

    def _discover_split(
        self,
        split: str,
    ) -> List[Dict]:

        split_root = (
            self._find_split_root(
                split
            )
        )

        scan_files = sorted(
            split_root.rglob("*.bin")
        )

        label_files = sorted(
            split_root.rglob("*.label")
        )

        scan_files = [
            path
            for path in scan_files
            if "label" not in str(path).lower()
        ]

        if not scan_files:
            raise RuntimeError(
                f"No SemanticSTF scans found in {split_root}"
            )

        if not label_files:
            raise RuntimeError(
                f"No SemanticSTF labels found in {split_root}"
            )

        label_index = {}

        for label_path in label_files:
            stem = label_path.stem

            if stem in label_index:
                raise RuntimeError(
                    "Duplicate SemanticSTF label stem "
                    f"inside split '{split}': {stem}"
                )

            label_index[stem] = label_path

        samples = []

        for scan_path in scan_files:
            label_path = label_index.get(
                scan_path.stem
            )

            if label_path is None:
                raise FileNotFoundError(
                    "Missing SemanticSTF label for scan:\n"
                    f"{scan_path}"
                )

            samples.append(
                {
                    "split": split,
                    "frame_id": scan_path.stem,
                    "scan_path": scan_path,
                    "label_path": label_path,
                }
            )

        samples.sort(
            key=lambda item: item["frame_id"]
        )

        return samples

    def _discover_samples(self):
        samples = []

        for split in self.splits:
            samples.extend(
                self._discover_split(
                    split
                )
            )

        return samples

    @staticmethod
    def _load_scan(
        scan_path: Path,
    ):
        raw = np.fromfile(
            scan_path,
            dtype=np.float32,
        )

        if raw.size % 5 != 0:
            raise ValueError(
                "SemanticSTF scan is not Nx5:\n"
                f"{scan_path}"
            )

        scan = raw.reshape(-1, 5)

        if not np.isfinite(scan).all():
            raise ValueError(
                "SemanticSTF scan contains NaN/Inf:\n"
                f"{scan_path}"
            )

        return scan

    @staticmethod
    def _load_native_labels(
        label_path: Path,
    ):
        return np.fromfile(
            label_path,
            dtype=np.uint32,
        )

    def _check_native_ids(
        self,
        native_labels,
    ):
        if not self.strict_labels:
            return

        observed = set(
            map(
                int,
                np.unique(native_labels),
            )
        )

        unknown = (
            observed
            - set(self.mapping)
        )

        if unknown:
            raise ValueError(
                "SemanticSTF native IDs missing from mapping:\n"
                f"{sorted(unknown)}"
            )

    def _remap_labels(
        self,
        native_labels,
    ):
        self._check_native_ids(
            native_labels
        )

        unified = np.zeros(
            native_labels.shape,
            dtype=np.uint8,
        )

        valid = (
            native_labels
            < len(self.lookup_table)
        )

        unified[valid] = (
            self.lookup_table[
                native_labels[valid]
            ]
        )

        return unified

    def __len__(self):
        return len(self.samples)

    def __getitem__(
        self,
        index,
    ):
        sample = self.samples[index]

        scan = self._load_scan(
            sample["scan_path"]
        )

        native_labels = (
            self._load_native_labels(
                sample["label_path"]
            )
        )

        if len(scan) != len(native_labels):
            raise ValueError(
                "SemanticSTF point/label mismatch:\n"
                f"Split: {sample['split']}\n"
                f"Frame: {sample['frame_id']}\n"
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
            "scan_path": str(
                sample["scan_path"]
            ),
            "label_path": str(
                sample["label_path"]
            ),
        }


if __name__ == "__main__":
    print("SemanticSTFDataset module OK")