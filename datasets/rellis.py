from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import yaml


class RELLISDataset:
    """
    RELLIS-3D semantic LiDAR loader.

    Native labels are converted into the project's unified ontology.
    """

    DEFAULT_SEQUENCES = (
        "00000",
        "00001",
        "00002",
        "00003",
        "00004",
    )

    def __init__(
        self,
        root: str,
        mapping_file: str = "datasets/mappings/rellis.yaml",
        sequences: Optional[Sequence[str]] = None,
    ):
        self.root = Path(root)
        self.mapping_file = Path(mapping_file)

        if sequences is None:
            self.sequences = list(self.DEFAULT_SEQUENCES)
        else:
            self.sequences = [
                str(sequence).zfill(5)
                for sequence in sequences
            ]

        self.mapping = self._load_mapping()
        self.lookup_table = self._build_lookup_table()
        self.samples = self._discover_samples()

        if not self.samples:
            raise RuntimeError(
                f"No RELLIS scan-label pairs found under: {self.root}"
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
        if not self.mapping:
            raise RuntimeError("RELLIS mapping is empty.")

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

        for sequence in self.sequences:
            sequence_directories = [
                path
                for path in self.root.rglob(sequence)
                if path.is_dir()
            ]

            if not sequence_directories:
                continue

            sequence_directory = sequence_directories[0]

            point_files = []

            for path in sequence_directory.rglob("*"):
                if not path.is_file():
                    continue

                if path.suffix.lower() not in {
                    ".bin",
                    ".npy",
                }:
                    continue

                if "label" in str(path).lower():
                    continue

                point_files.append(path)

            label_index = {}

            for path in sequence_directory.rglob("*"):
                if not path.is_file():
                    continue

                if path.suffix.lower() not in {
                    ".label",
                    ".npy",
                }:
                    continue

                if "label" not in str(path).lower():
                    continue

                label_index[path.stem] = path

            for scan_path in sorted(point_files):
                label_path = label_index.get(
                    scan_path.stem
                )

                if label_path is None:
                    continue

                samples.append(
                    {
                        "sequence": sequence,
                        "frame_id": scan_path.stem,
                        "scan_path": scan_path,
                        "label_path": label_path,
                    }
                )

        samples.sort(
            key=lambda item: (
                item["sequence"],
                item["frame_id"],
            )
        )

        return samples

    @staticmethod
    def _load_scan(scan_path: Path) -> np.ndarray:
        if scan_path.suffix.lower() == ".npy":
            scan = np.load(scan_path)

            if scan.ndim != 2 or scan.shape[1] < 3:
                raise ValueError(
                    f"Invalid RELLIS scan: {scan_path}"
                )

            return scan.astype(
                np.float32,
                copy=False,
            )

        raw = np.fromfile(
            scan_path,
            dtype=np.float32,
        )

        if raw.size % 4 == 0:
            return raw.reshape(-1, 4)

        if raw.size % 3 == 0:
            return raw.reshape(-1, 3)

        raise ValueError(
            f"Unable to determine point format: {scan_path}"
        )

    @staticmethod
    def _load_native_labels(
        label_path: Path,
    ) -> np.ndarray:

        if label_path.suffix.lower() == ".npy":
            return np.load(
                label_path
            ).reshape(-1).astype(
                np.uint16,
                copy=False,
            )

        return np.fromfile(
            label_path,
            dtype=np.uint32,
        ).astype(
            np.uint16,
            copy=False,
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

        xyz = scan[:, :3].astype(
            np.float32,
            copy=False,
        )

        if scan.shape[1] >= 4:
            intensity = scan[:, 3].astype(
                np.float32,
                copy=False,
            )
        else:
            intensity = np.zeros(
                len(scan),
                dtype=np.float32,
            )

        return {
            "xyz": xyz,
            "intensity": intensity,
            "native_label": native_labels,
            "semantic_label": self._remap_labels(
                native_labels
            ),
            "dataset": "rellis_3d",
            "sequence": sample["sequence"],
            "frame_id": sample["frame_id"],
            "scan_path": str(sample["scan_path"]),
            "label_path": str(sample["label_path"]),
        }


if __name__ == "__main__":
    print("RELLISDataset module OK")