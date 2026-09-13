from __future__ import annotations

import json
import math
import random
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import numpy as np
import yaml

try:
    import torch
    from torch.utils.data import Dataset, Sampler
except Exception:  # Allows metadata/verification use without torch installed.
    torch = None

    class Dataset:  # type: ignore
        pass

    class Sampler:  # type: ignore
        pass


NUM_CLASSES = 23
IGNORE_INDEX = 0

_SCAN_WIDTH = {
    "semantic_kitti": 4,
    "rellis_3d": 4,
    "semantic_stf": 5,
    "nuscenes_mini": 5,
}

_LABEL_DTYPE = {
    "semantic_kitti": np.uint32,
    "rellis_3d": np.uint32,
    "semantic_stf": np.uint32,
    "nuscenes_mini": np.uint8,
}

_MAPPING_FILE = {
    "semantic_kitti": "datasets/mappings/semantic_kitti.yaml",
    "rellis_3d": "datasets/mappings/rellis.yaml",
    "semantic_stf": "datasets/mappings/semantic_stf.yaml",
    "nuscenes_mini": "datasets/mappings/nuscenes.yaml",
}


def _load_mapping(path: Path) -> Dict[int, int]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    mapping = cfg.get("mapping")
    if not mapping:
        raise RuntimeError(f"Mapping is empty: {path}")
    return {int(k): int(v) for k, v in mapping.items()}


def _build_lookup(mapping: Mapping[int, int]) -> np.ndarray:
    maximum = max(mapping) if mapping else 0
    table = np.zeros(maximum + 1, dtype=np.uint8)
    for native, unified in mapping.items():
        table[int(native)] = int(unified)
    return table


def _decode_scan(dataset_name: str, raw: bytes) -> np.ndarray:
    width = _SCAN_WIDTH[dataset_name]
    values = np.frombuffer(raw, dtype=np.float32)
    if values.size % width != 0:
        raise ValueError(
            f"{dataset_name}: scan float count {values.size} is not divisible by {width}"
        )
    scan = values.reshape(-1, width)
    if not np.isfinite(scan).all():
        raise ValueError(f"{dataset_name}: scan contains NaN/Inf")
    return scan


def _decode_native_labels(dataset_name: str, raw: bytes) -> np.ndarray:
    labels = np.frombuffer(raw, dtype=_LABEL_DTYPE[dataset_name])
    if dataset_name == "semantic_kitti":
        labels = (labels & 0xFFFF).astype(np.uint16, copy=False)
    return labels


class F0TarDataset(Dataset):
    """
    Random-access dataset over the lossless F0 TAR cache.

    The cache stores original scan/label bytes. This class decodes those bytes
    exactly as the original dataset loaders do and applies the same unified
    ontology mappings.

    Cache layout expected in cache_dir:
      f0_shard_00000.tar
      f0_shard_00000.json
      ...

    Each worker/process opens TAR handles lazily and keeps only a bounded
    number of handles open.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        repo_root: str | Path,
        split: Optional[str] = None,
        max_open_tars: int = 8,
        strict: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.repo_root = Path(repo_root)
        self.split = split
        self.max_open_tars = max(1, int(max_open_tars))
        self.strict = bool(strict)

        if not self.cache_dir.is_dir():
            raise FileNotFoundError(f"F0 TAR cache directory not found: {self.cache_dir}")

        self.mappings: Dict[str, Dict[int, int]] = {}
        self.lookup_tables: Dict[str, np.ndarray] = {}
        for dataset_name, rel in _MAPPING_FILE.items():
            mapping = _load_mapping(self.repo_root / rel)
            self.mappings[dataset_name] = mapping
            self.lookup_tables[dataset_name] = _build_lookup(mapping)

        self.rows: List[Dict] = []
        self.dataset_to_indices: Dict[str, List[int]] = defaultdict(list)
        self._tar_handles: Dict[int, tarfile.TarFile] = {}
        self._tar_lru: List[int] = []

        sidecars = sorted(self.cache_dir.glob("f0_shard_*.json"))
        if not sidecars:
            raise RuntimeError(f"No f0_shard_*.json sidecars in {self.cache_dir}")

        for sidecar_path in sidecars:
            with sidecar_path.open("r", encoding="utf-8") as f:
                sidecar = json.load(f)

            shard_id = int(sidecar["shard_id"])
            tar_path = self.cache_dir / f"f0_shard_{shard_id:05d}.tar"
            if not tar_path.is_file():
                raise FileNotFoundError(
                    f"Sidecar exists but TAR is missing: {sidecar_path.name} -> {tar_path.name}"
                )
            expected_size = int(sidecar.get("tar_size_bytes", -1))
            if expected_size >= 0 and tar_path.stat().st_size != expected_size:
                raise RuntimeError(
                    f"TAR size mismatch for {tar_path}: "
                    f"expected {expected_size}, got {tar_path.stat().st_size}"
                )

            for sample in sidecar.get("samples", []):
                if split is not None and sample.get("split") != split:
                    continue
                dataset_name = str(sample["dataset"])
                if dataset_name not in _SCAN_WIDTH:
                    raise ValueError(f"Unknown dataset in cache: {dataset_name}")
                row = dict(sample)
                row["shard_id"] = shard_id
                row["tar_path"] = str(tar_path)
                idx = len(self.rows)
                self.rows.append(row)
                self.dataset_to_indices[dataset_name].append(idx)

        if not self.rows:
            raise RuntimeError(
                f"No samples matched split={split!r} in cache {self.cache_dir}"
            )

    def __len__(self) -> int:
        return len(self.rows)

    def _touch_lru(self, shard_id: int) -> None:
        if shard_id in self._tar_lru:
            self._tar_lru.remove(shard_id)
        self._tar_lru.append(shard_id)
        while len(self._tar_lru) > self.max_open_tars:
            old = self._tar_lru.pop(0)
            handle = self._tar_handles.pop(old, None)
            if handle is not None:
                handle.close()

    def _get_tar(self, shard_id: int, tar_path: str) -> tarfile.TarFile:
        handle = self._tar_handles.get(shard_id)
        if handle is None:
            handle = tarfile.open(tar_path, mode="r")
            self._tar_handles[shard_id] = handle
        self._touch_lru(shard_id)
        return handle

    @staticmethod
    def _extract_bytes(tar: tarfile.TarFile, name: str) -> bytes:
        member = tar.getmember(name)
        stream = tar.extractfile(member)
        if stream is None:
            raise RuntimeError(f"Could not extract TAR member: {name}")
        return stream.read()

    def _remap(self, dataset_name: str, native: np.ndarray) -> np.ndarray:
        if self.strict:
            observed = set(map(int, np.unique(native)))
            unknown = observed - set(self.mappings[dataset_name])
            if unknown:
                raise ValueError(
                    f"{dataset_name}: native IDs absent from mapping: {sorted(unknown)}"
                )

        table = self.lookup_tables[dataset_name]
        unified = np.zeros(native.shape, dtype=np.uint8)
        valid = native < len(table)
        unified[valid] = table[native[valid]]
        return unified

    def __getitem__(self, index: int) -> Dict:
        row = self.rows[index]
        dataset_name = str(row["dataset"])
        shard_id = int(row["shard_id"])

        tar = self._get_tar(shard_id, str(row["tar_path"]))
        scan_bytes = self._extract_bytes(tar, str(row["scan_arcname"]))
        label_bytes = self._extract_bytes(tar, str(row["label_arcname"]))

        scan = _decode_scan(dataset_name, scan_bytes)
        native = _decode_native_labels(dataset_name, label_bytes)
        if len(scan) != len(native):
            raise ValueError(
                f"{dataset_name}/{row['identity']}: "
                f"{len(scan)} points != {len(native)} labels"
            )

        semantic = self._remap(dataset_name, native)

        return {
            "xyz": scan[:, :3].astype(np.float32, copy=False),
            "intensity": scan[:, 3].astype(np.float32, copy=False),
            "semantic_label": semantic,
            "native_label": native,
            "dataset": dataset_name,
            "split": row.get("split"),
            "frame_id": row.get("identity"),
            "identity": row.get("identity"),
            "shard_id": shard_id,
        }

    def close(self) -> None:
        for handle in self._tar_handles.values():
            handle.close()
        self._tar_handles.clear()
        self._tar_lru.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self):
        # PyTorch workers must not inherit live TarFile handles.
        state = self.__dict__.copy()
        state["_tar_handles"] = {}
        state["_tar_lru"] = []
        return state


class ExactMixtureSampler(Sampler):
    """
    Deterministic per-epoch sampler with exact dataset target proportions.

    For a dataset with fewer items than its requested quota, indices are cycled
    through shuffled permutations. For a dataset with more items than its quota,
    a shuffled subset is used. This avoids uncontrolled WeightedRandomSampler
    variance while intentionally oversampling the small datasets.
    """

    def __init__(
        self,
        dataset: F0TarDataset,
        proportions: Mapping[str, float],
        epoch_size: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.proportions = {str(k): float(v) for k, v in proportions.items()}
        self.epoch_size = int(epoch_size or len(dataset))
        self.seed = int(seed)
        self.epoch = 0

        if self.epoch_size <= 0:
            raise ValueError("epoch_size must be > 0")

        total_p = sum(self.proportions.values())
        if not math.isclose(total_p, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"Sampling proportions must sum to 1.0, got {total_p}")

        missing = set(self.proportions) - set(dataset.dataset_to_indices)
        if missing:
            raise ValueError(f"Sampling requested datasets absent from cache: {sorted(missing)}")

        self.quotas = self._compute_quotas()

    def _compute_quotas(self) -> Dict[str, int]:
        names = list(self.proportions)
        raw = {n: self.epoch_size * self.proportions[n] for n in names}
        quotas = {n: int(math.floor(raw[n])) for n in names}
        remaining = self.epoch_size - sum(quotas.values())
        order = sorted(names, key=lambda n: (raw[n] - quotas[n], n), reverse=True)
        for name in order[:remaining]:
            quotas[name] += 1
        return quotas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _draw(pool: Sequence[int], quota: int, rng: random.Random) -> List[int]:
        if not pool:
            raise RuntimeError("Cannot draw from an empty dataset pool")
        out: List[int] = []
        while len(out) < quota:
            block = list(pool)
            rng.shuffle(block)
            need = quota - len(out)
            out.extend(block[:need])
        return out

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        epoch_indices: List[int] = []
        for dataset_name, quota in self.quotas.items():
            epoch_indices.extend(
                self._draw(
                    self.dataset.dataset_to_indices[dataset_name],
                    quota,
                    rng,
                )
            )
        rng.shuffle(epoch_indices)
        return iter(epoch_indices)

    def __len__(self) -> int:
        return self.epoch_size


def variable_point_collate(batch: Sequence[Dict]) -> List[Dict]:
    """
    Deliberately does not stack variable-length clouds.
    Sparse voxelization/collation belongs in the SPVCNN adapter.
    """
    return list(batch)
