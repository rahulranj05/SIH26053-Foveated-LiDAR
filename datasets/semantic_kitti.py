from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import yaml


class SemanticKITTIDataset:
    """
    SemanticKITTI semantic LiDAR loader.

    Expected structure:

    root/
        sequences/
            00/
                velodyne/
                labels/
            ...
            10/

    Returned unified semantic labels use the SIH26053 ontology.
    """

    DEFAULT_SEQUENCES = tuple(
        f"{i:02d}"
        for i in range(11)
    )

    REQUIRED_SCAN_WIDTH = 4

    def __init__(
        self,
        root,
        mapping_file="datasets/mappings/semantic_kitti.yaml",
        sequences: Optional[Sequence[str]] = None,
        strict_labels: bool = True,
    ):
        self.root = Path(root)
        self.mapping_file = Path(mapping_file)
        self.strict_labels = strict_labels

        if not self.root.exists():
            raise FileNotFoundError(
                f"SemanticKITTI root not found: {self.root}"
            )

        if sequences is None:
            self.sequences = list(
                self.DEFAULT_SEQUENCES
            )
        else:
            self.sequences = [
                str(sequence).zfill(2)
                for sequence in sequences
            ]

        self.mapping = self._load_mapping()
        self.lookup_table = self._build_lookup_table()
        self.samples = self._discover_samples()

        if not self.samples:
            raise RuntimeError(
                "No SemanticKITTI scan-label pairs found under:\n"
                f"{self.root}"
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

        mapping = config.get("mapping")

        if not mapping:
            raise RuntimeError(
                f"Mapping is empty: {self.mapping_file}"
            )

        return {
            int(native): int(unified)
            for native, unified in mapping.items()
        }

    def _build_lookup_table(self) -> np.ndarray:
        maximum = max(self.mapping)

        lookup = np.zeros(
            maximum + 1,
            dtype=np.uint8,
        )

        for native, unified in self.mapping.items():
            lookup[native] = unified

        return lookup

    def _discover_samples(self) -> List[Dict]:
        sequences_root = (
            self.root
            / "sequences"
        )

        if not sequences_root.exists():
            raise FileNotFoundError(
                "SemanticKITTI sequences directory missing:\n"
                f"{sequences_root}"
            )

        samples = []

        for sequence in self.sequences:
            sequence_root = (
                sequences_root
                / sequence
            )

            scan_dir = (
                sequence_root
                / "velodyne"
            )

            label_dir = (
                sequence_root
                / "labels"
            )

            if not scan_dir.exists():
                continue

            if not label_dir.exists():
                continue

            scan_files = sorted(
                scan_dir.glob("*.bin")
            )

            label_files = {
                path.stem: path
                for path in label_dir.glob("*.label")
            }

            for scan_path in scan_files:
                label_path = label_files.get(
                    scan_path.stem
                )

                if label_path is None:
                    raise FileNotFoundError(
                        "Missing SemanticKITTI label for scan:\n"
                        f"{scan_path}"
                    )

                samples.append(
                    {
                        "sequence": sequence,
                        "frame_id": scan_path.stem,
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

        if raw.size % 4 != 0:
            raise ValueError(
                "SemanticKITTI scan is not Nx4:\n"
                f"{scan_path}"
            )

        scan = raw.reshape(-1, 4)

        if not np.isfinite(scan).all():
            raise ValueError(
                "SemanticKITTI scan contains NaN/Inf:\n"
                f"{scan_path}"
            )

        return scan

    @staticmethod
    def _load_native_labels(
        label_path: Path,
    ) -> np.ndarray:

        packed = np.fromfile(
            label_path,
            dtype=np.uint32,
        )

        semantic = (
            packed & 0xFFFF
        ).astype(
            np.uint16,
            copy=False,
        )

        return semantic

    def _check_native_ids(
        self,
        native_labels: np.ndarray,
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
            - set(self.mapping.keys())
        )

        if unknown:
            raise ValueError(
                "SemanticKITTI contains native IDs "
                "missing from mapping:\n"
                f"{sorted(unknown)}"
            )

    def _remap_labels(
        self,
        native_labels: np.ndarray,
    ) -> np.ndarray:

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
        index: int,
    ) -> Dict:

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
                "SemanticKITTI point/label mismatch:\n"
                f"Frame: {sample['frame_id']}\n"
                f"Points: {len(scan)}\n"
                f"Labels: {len(native_labels)}"
            )

        semantic_labels = (
            self._remap_labels(
                native_labels
            )
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
            "semantic_label": semantic_labels,
            "dataset": "semantic_kitti",
            "sequence": sample["sequence"],
            "frame_id": sample["frame_id"],
            "scan_path": str(
                sample["scan_path"]
            ),
            "label_path": str(
                sample["label_path"]
            ),
        }

    def summary(self):
        counts = {}

        for sample in self.samples:
            sequence = sample["sequence"]

            counts[sequence] = (
                counts.get(sequence, 0)
                + 1
            )

        print("=" * 70)
        print("SemanticKITTI")
        print("=" * 70)
        print(f"Root: {self.root}")
        print(f"Samples: {len(self)}")

        for sequence in sorted(counts):
            print(
                f"  {sequence}: "
                f"{counts[sequence]}"
            )

        print("=" * 70)


if __name__ == "__main__":
    print("SemanticKITTIDataset module OK")