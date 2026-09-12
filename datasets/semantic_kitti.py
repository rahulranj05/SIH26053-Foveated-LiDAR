from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import yaml


class SemanticKITTIDataset:
    """
    Lightweight SemanticKITTI loader.

    Responsibilities:
    - Discover LiDAR scans and semantic labels.
    - Read .bin point clouds.
    - Read .label files.
    - Remove SemanticKITTI instance information.
    - Convert native SemanticKITTI classes into our unified ontology.

    Expected SemanticKITTI structure:

    root/
    └── sequences/
        ├── 00/
        │   ├── velodyne/
        │   │   ├── 000000.bin
        │   │   └── ...
        │   └── labels/
        │       ├── 000000.label
        │       └── ...
        └── ...
    """

    DEFAULT_SEQUENCES = tuple(f"{i:02d}" for i in range(11))

    def __init__(
        self,
        root: str,
        mapping_file: str = "datasets/mappings/semantic_kitti.yaml",
        sequences: Optional[Sequence[str]] = None,
    ):
        self.root = Path(root)
        self.mapping_file = Path(mapping_file)

        if sequences is None:
            self.sequences = list(self.DEFAULT_SEQUENCES)
        else:
            self.sequences = [str(seq).zfill(2) for seq in sequences]

        self.mapping = self._load_mapping()
        self.lookup_table = self._build_lookup_table()

        self.samples = self._discover_samples()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No SemanticKITTI scan-label pairs found under: {self.root}"
            )

    def _load_mapping(self) -> Dict[int, int]:
        """
        Load native SemanticKITTI ID -> unified ontology ID mapping.
        """

        if not self.mapping_file.exists():
            raise FileNotFoundError(
                f"Mapping file not found: {self.mapping_file}"
            )

        with open(self.mapping_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if "mapping" not in config:
            raise KeyError(
                f"'mapping' section missing from {self.mapping_file}"
            )

        mapping = {
            int(native_id): int(unified_id)
            for native_id, unified_id in config["mapping"].items()
        }

        return mapping

    def _build_lookup_table(self) -> np.ndarray:
        """
        Build a fast lookup table so remapping does not require
        Python loops over millions of LiDAR points.
        """

        if not self.mapping:
            raise RuntimeError("SemanticKITTI mapping is empty.")

        max_native_id = max(self.mapping.keys())

        lut = np.zeros(max_native_id + 1, dtype=np.uint8)

        for native_id, unified_id in self.mapping.items():
            lut[native_id] = unified_id

        return lut

    def _discover_samples(self) -> List[Dict]:
        """
        Find all valid scan-label pairs in the selected sequences.
        """

        samples = []

        sequences_root = self.root / "sequences"

        if not sequences_root.exists():
            raise FileNotFoundError(
                f"SemanticKITTI sequences folder not found: {sequences_root}"
            )

        for sequence in self.sequences:
            sequence_dir = sequences_root / sequence
            velodyne_dir = sequence_dir / "velodyne"
            labels_dir = sequence_dir / "labels"

            if not velodyne_dir.exists():
                continue

            if not labels_dir.exists():
                continue

            scan_files = sorted(velodyne_dir.glob("*.bin"))

            for scan_path in scan_files:
                label_path = labels_dir / f"{scan_path.stem}.label"

                if not label_path.exists():
                    continue

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
    def _load_scan(scan_path: Path) -> np.ndarray:
        """
        Read SemanticKITTI Velodyne scan.

        Each point contains:
            x, y, z, intensity
        """

        scan = np.fromfile(scan_path, dtype=np.float32)

        if scan.size % 4 != 0:
            raise ValueError(
                f"Invalid SemanticKITTI scan shape: {scan_path}"
            )

        scan = scan.reshape(-1, 4)

        return scan

    @staticmethod
    def _load_native_labels(label_path: Path) -> np.ndarray:
        """
        Read SemanticKITTI labels.

        SemanticKITTI stores:

        lower 16 bits -> semantic class
        upper 16 bits -> instance ID

        We only need the semantic class here.
        """

        raw_labels = np.fromfile(label_path, dtype=np.uint32)

        semantic_labels = raw_labels & 0xFFFF

        return semantic_labels.astype(np.uint16)

    def _remap_labels(self, native_labels: np.ndarray) -> np.ndarray:
        """
        Convert native SemanticKITTI IDs to unified ontology IDs.

        Any unknown ID automatically becomes class 0 = ignore.
        """

        unified_labels = np.zeros(
            native_labels.shape,
            dtype=np.uint8,
        )

        valid = native_labels < len(self.lookup_table)

        unified_labels[valid] = self.lookup_table[
            native_labels[valid]
        ]

        return unified_labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]

        scan = self._load_scan(sample["scan_path"])

        native_labels = self._load_native_labels(
            sample["label_path"]
        )

        if len(scan) != len(native_labels):
            raise ValueError(
                "Point/label count mismatch\n"
                f"Scan:  {sample['scan_path']}\n"
                f"Label: {sample['label_path']}\n"
                f"Points: {len(scan)}\n"
                f"Labels: {len(native_labels)}"
            )

        unified_labels = self._remap_labels(native_labels)

        xyz = scan[:, :3].astype(np.float32)
        intensity = scan[:, 3].astype(np.float32)

        return {
            "xyz": xyz,
            "intensity": intensity,
            "native_label": native_labels,
            "semantic_label": unified_labels,
            "dataset": "semantic_kitti",
            "sequence": sample["sequence"],
            "frame_id": sample["frame_id"],
            "scan_path": str(sample["scan_path"]),
            "label_path": str(sample["label_path"]),
        }

    def summary(self) -> None:
        """
        Print a short dataset summary.
        """

        sequence_counts = {}

        for sample in self.samples:
            seq = sample["sequence"]
            sequence_counts[seq] = sequence_counts.get(seq, 0) + 1

        print("=" * 60)
        print("SemanticKITTI Dataset")
        print("=" * 60)
        print(f"Root:             {self.root}")
        print(f"Mapping:          {self.mapping_file}")
        print(f"Total samples:    {len(self.samples)}")
        print(f"Sequences:        {', '.join(self.sequences)}")
        print()

        print("Samples per sequence:")

        for seq in sorted(sequence_counts):
            print(f"  {seq}: {sequence_counts[seq]}")

        print("=" * 60)


if __name__ == "__main__":
    print(
        "SemanticKITTIDataset loader module.\n"
        "Import this class from a test script and provide the dataset root."
    )