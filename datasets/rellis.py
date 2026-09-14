from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import yaml


class RELLISDataset:
    """
    RELLIS-3D LiDAR semantic segmentation loader.

    Canonical SIH26053 root:

    .../RELLIS_3D/extracted/Rellis-3D
    """

    DEFAULT_SEQUENCES = (
        "00000",
        "00001",
        "00002",
        "00003",
        "00004",
    )

    POINT_DIRECTORY_CANDIDATES = (
        "os1_cloud_node_kitti_bin",
        "velodyne",
        "points",
        "lidar",
    )

    LABEL_DIRECTORY_CANDIDATES = (
        "os1_cloud_node_semantickitti_label_id",
        "labels",
        "label",
    )

    def __init__(
        self,
        root,
        mapping_file="datasets/mappings/rellis.yaml",
        sequences: Optional[Sequence[str]] = None,
        strict_labels: bool = True,
        filter_invalid_geometry: bool = True,
        minimum_valid_range: float = 1e-6,
    ):
        self.root = Path(root)
        self.mapping_file = Path(mapping_file)
        self.strict_labels = strict_labels

        self.filter_invalid_geometry = bool(
            filter_invalid_geometry
        )

        self.minimum_valid_range = float(
            minimum_valid_range
        )

        if self.minimum_valid_range < 0:
            raise ValueError(
                "minimum_valid_range must be >= 0"
            )

        if not self.root.exists():
            raise FileNotFoundError(
                f"RELLIS root not found: {self.root}"
            )

        if sequences is None:
            self.sequences = list(
                self.DEFAULT_SEQUENCES
            )
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
                "No RELLIS scan-label pairs found under:\n"
                f"{self.root}"
            )

    def _load_mapping(self) -> Dict[int, int]:
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
                f"Empty RELLIS mapping: {self.mapping_file}"
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

    @staticmethod
    def _find_named_directory(
        sequence_root: Path,
        candidates,
    ):
        for name in candidates:
            direct = (
                sequence_root
                / name
            )

            if direct.is_dir():
                return direct

        for child in sequence_root.iterdir():
            if (
                child.is_dir()
                and child.name.lower()
                in {
                    name.lower()
                    for name in candidates
                }
            ):
                return child

        return None

    def _get_sequence_root(
        self,
        sequence: str,
    ) -> Path:

        direct = (
            self.root
            / sequence
        )

        if direct.is_dir():
            return direct

        matches = [
            path
            for path in self.root.glob(
                f"*/{sequence}"
            )
            if path.is_dir()
        ]

        if len(matches) == 1:
            return matches[0]

        if not matches:
            raise FileNotFoundError(
                f"RELLIS sequence not found: {sequence}"
            )

        raise RuntimeError(
            "Multiple RELLIS sequence directories found for "
            f"{sequence}:\n"
            + "\n".join(
                str(path)
                for path in matches
            )
        )

    def _discover_sequence(
        self,
        sequence: str,
    ) -> List[Dict]:

        sequence_root = (
            self._get_sequence_root(
                sequence
            )
        )

        point_dir = (
            self._find_named_directory(
                sequence_root,
                self.POINT_DIRECTORY_CANDIDATES,
            )
        )

        label_dir = (
            self._find_named_directory(
                sequence_root,
                self.LABEL_DIRECTORY_CANDIDATES,
            )
        )

        if point_dir is not None:
            point_files = sorted(
                point_dir.glob("*.bin")
            )
        else:
            point_files = sorted(
                sequence_root.rglob("*.bin")
            )

        if label_dir is not None:
            label_files = sorted(
                label_dir.glob("*.label")
            )
        else:
            label_files = sorted(
                sequence_root.rglob("*.label")
            )

        if not point_files:
            raise RuntimeError(
                f"No RELLIS .bin files in {sequence_root}"
            )

        if not label_files:
            raise RuntimeError(
                f"No RELLIS .label files in {sequence_root}"
            )

        label_index = {}

        for label_path in label_files:
            stem = label_path.stem

            if stem in label_index:
                raise RuntimeError(
                    "Duplicate RELLIS label filename within "
                    f"sequence {sequence}: {stem}"
                )

            label_index[stem] = label_path

        samples = []

        for scan_path in point_files:
            label_path = label_index.get(
                scan_path.stem
            )

            if label_path is None:
                raise FileNotFoundError(
                    "Missing RELLIS label for scan:\n"
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

    def _discover_samples(self):
        samples = []

        for sequence in self.sequences:
            samples.extend(
                self._discover_sequence(
                    sequence
                )
            )

        samples.sort(
            key=lambda item: (
                item["sequence"],
                item["frame_id"],
            )
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
                "RELLIS scan is not Nx4:\n"
                f"{scan_path}"
            )

        scan = raw.reshape(-1, 4)

        if not np.isfinite(scan).all():
            raise ValueError(
                "RELLIS scan contains NaN/Inf:\n"
                f"{scan_path}"
            )

        return scan

    @staticmethod
    def _load_native_labels(
        label_path: Path,
    ) -> np.ndarray:

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
                "RELLIS native IDs missing from mapping:\n"
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
                "RELLIS point/label mismatch:\n"
                f"Frame: {sample['frame_id']}\n"
                f"Points: {len(scan)}\n"
                f"Labels: {len(native_labels)}"
            )

        xyz = scan[:, :3].astype(
            np.float32,
            copy=False,
        )

        intensity = scan[:, 3].astype(
            np.float32,
            copy=False,
        )

        semantic_labels = self._remap_labels(
            native_labels
        )

        raw_point_count = len(xyz)

        if self.filter_invalid_geometry:

            ranges = np.linalg.norm(
                xyz,
                axis=1,
            )

            valid_geometry = (
                ranges
                > self.minimum_valid_range
            )

            xyz = xyz[valid_geometry]
            intensity = intensity[valid_geometry]
            native_labels = native_labels[
                valid_geometry
            ]
            semantic_labels = semantic_labels[
                valid_geometry
            ]

        filtered_point_count = len(xyz)

        return {
            "xyz": xyz,
            "intensity": intensity,
            "native_label": native_labels,
            "semantic_label": semantic_labels,
            "raw_point_count": raw_point_count,
            "filtered_point_count": filtered_point_count,
            "removed_invalid_points": (
                raw_point_count
                - filtered_point_count
            ),
            "dataset": "rellis_3d",
            "sequence": sample["sequence"],
            "frame_id": sample["frame_id"],
            "scan_path": str(
                sample["scan_path"]
            ),
            "label_path": str(
                sample["label_path"]
            ),
        }


if __name__ == "__main__":
    print("RELLISDataset module OK")