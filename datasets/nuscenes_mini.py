from pathlib import Path
from typing import Dict, List
import json

import numpy as np
import yaml


class NuScenesMiniDataset:
    """
    nuScenes mini lidarseg loader.

    Canonical SIH26053 top-level folder:

        .../datasets/nuScenes

    Metadata is read from v1.0-mini.
    """

    def __init__(
        self,
        root,
        mapping_file="datasets/mappings/nuscenes.yaml",
        strict_labels: bool = True,
    ):
        self.root = Path(root)
        self.mapping_file = Path(mapping_file)
        self.strict_labels = strict_labels

        if not self.root.exists():
            raise FileNotFoundError(
                f"nuScenes root not found: {self.root}"
            )

        self.mapping = self._load_mapping()
        self.lookup_table = self._build_lookup_table()

        self.metadata_dir = (
            self._find_metadata_directory()
        )

        self.dataset_root = (
            self.metadata_dir.parent
        )

        self.samples = self._discover_samples()

        if not self.samples:
            raise RuntimeError(
                "No nuScenes lidarseg samples found under:\n"
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
                f"Empty nuScenes mapping: {self.mapping_file}"
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

    def _find_metadata_directory(
        self,
    ) -> Path:

        direct = (
            self.root
            / "v1.0-mini"
        )

        if (
            direct.is_dir()
            and (
                direct
                / "sample_data.json"
            ).exists()
        ):
            return direct

        candidates = [
            path
            for path in self.root.rglob(
                "v1.0-mini"
            )
            if (
                path.is_dir()
                and (
                    path
                    / "sample_data.json"
                ).exists()
            )
        ]

        if len(candidates) == 1:
            return candidates[0]

        if not candidates:
            raise FileNotFoundError(
                "Could not locate nuScenes "
                "v1.0-mini metadata."
            )

        raise RuntimeError(
            "Multiple nuScenes v1.0-mini directories found:\n"
            + "\n".join(
                str(path)
                for path in candidates
            )
        )

    def _resolve_metadata_path(
        self,
        relative_path: str,
    ) -> Path:

        relative = Path(relative_path)

        candidates = [
            self.dataset_root / relative,
            self.root / relative,
            self.metadata_dir / relative,
        ]

        for path in candidates:
            if path.exists():
                return path

        raise FileNotFoundError(
            "nuScenes metadata references a missing file:\n"
            f"Relative path: {relative_path}\n"
            "Tried:\n"
            + "\n".join(
                f"  {path}"
                for path in candidates
            )
        )

    @staticmethod
    def _load_json(
        path: Path,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing nuScenes metadata: {path}"
            )

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    def _discover_samples(self) -> List[Dict]:
        sample_data = self._load_json(
            self.metadata_dir
            / "sample_data.json"
        )

        lidarseg = self._load_json(
            self.metadata_dir
            / "lidarseg.json"
        )

        sample_index = {
            item["token"]: item
            for item in sample_data
        }

        samples = []

        missing_tokens = []

        for item in lidarseg:
            token = item[
                "sample_data_token"
            ]

            sample_info = (
                sample_index.get(token)
            )

            if sample_info is None:
                missing_tokens.append(
                    token
                )
                continue

            scan_path = (
                self._resolve_metadata_path(
                    sample_info[
                        "filename"
                    ]
                )
            )

            label_path = (
                self._resolve_metadata_path(
                    item["filename"]
                )
            )

            samples.append(
                {
                    "token": token,
                    "scan_path": scan_path,
                    "label_path": label_path,
                }
            )

        if missing_tokens:
            raise RuntimeError(
                "nuScenes lidarseg contains "
                "sample_data_token values absent from "
                "sample_data.json.\n"
                f"Count: {len(missing_tokens)}"
            )

        samples.sort(
            key=lambda item: item["token"]
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
                "nuScenes LiDAR scan is not Nx5:\n"
                f"{scan_path}"
            )

        scan = raw.reshape(-1, 5)

        if not np.isfinite(scan).all():
            raise ValueError(
                "nuScenes scan contains NaN/Inf:\n"
                f"{scan_path}"
            )

        return scan

    @staticmethod
    def _load_native_labels(
        label_path: Path,
    ):
        return np.fromfile(
            label_path,
            dtype=np.uint8,
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
                "nuScenes native IDs missing from mapping:\n"
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
                "nuScenes point/label mismatch:\n"
                f"Token: {sample['token']}\n"
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
            "scan_path": str(
                sample["scan_path"]
            ),
            "label_path": str(
                sample["label_path"]
            ),
        }


if __name__ == "__main__":
    print("NuScenesMiniDataset module OK")