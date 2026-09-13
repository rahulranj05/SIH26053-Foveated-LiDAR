from __future__ import annotations

import hashlib
import json
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import yaml

from datasets.f0_tar_dataset import F0TarDataset, ExactMixtureSampler


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _write_mapping(root: Path, rel: str, mapping):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"mapping": mapping}, f)


def _build_fixture(root: Path):
    for rel in (
        "datasets/mappings/semantic_kitti.yaml",
        "datasets/mappings/rellis.yaml",
        "datasets/mappings/semantic_stf.yaml",
        "datasets/mappings/nuscenes.yaml",
    ):
        _write_mapping(root, rel, {0: 0, 1: 1, 10: 16})

    cache = root / "cache"
    cache.mkdir()

    examples = [
        ("semantic_kitti", np.array([[1,2,3,0.5],[4,5,6,0.7]], np.float32),
         np.array([10,1], np.uint32)),
        ("rellis_3d", np.array([[1,2,3,0.5],[4,5,6,0.7]], np.float32),
         np.array([1,0], np.uint32)),
        ("semantic_stf", np.array([[1,2,3,0.5,9],[4,5,6,0.7,8]], np.float32),
         np.array([1,0], np.uint32)),
        ("nuscenes_mini", np.array([[1,2,3,0.5,9],[4,5,6,0.7,8]], np.float32),
         np.array([1,0], np.uint8)),
    ]

    side = {
        "shard_id": 0,
        "sample_count": len(examples),
        "samples": [],
    }
    tar_path = cache / "f0_shard_00000.tar"

    with tarfile.open(tar_path, "w") as tar:
        for i, (name, scan, labels) in enumerate(examples):
            scan_bytes = scan.tobytes()
            label_bytes = labels.tobytes()
            base = f"{name}/train/{i}"
            scan_arc = f"{base}/scan.bin"
            label_arc = f"{base}/label.label"

            for arc, payload in ((scan_arc, scan_bytes), (label_arc, label_bytes)):
                import io
                info = tarfile.TarInfo(arc)
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))

            side["samples"].append({
                "dataset": name,
                "split": "train",
                "identity": str(i),
                "scan_arcname": scan_arc,
                "label_arcname": label_arc,
                "scan_sha256": _sha(scan_bytes),
                "label_sha256": _sha(label_bytes),
                "scan_size": len(scan_bytes),
                "label_size": len(label_bytes),
            })

    side["tar_size_bytes"] = tar_path.stat().st_size
    with (cache / "f0_shard_00000.json").open("w", encoding="utf-8") as f:
        json.dump(side, f)
    return cache


def main():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cache = _build_fixture(root)
        ds = F0TarDataset(cache, root, split="train")

        assert len(ds) == 4
        assert ds[0]["xyz"].shape == (2, 3)
        assert ds[0]["semantic_label"].tolist() == [16, 1]
        assert ds[1]["semantic_label"].tolist() == [1, 0]
        assert ds[2]["intensity"].tolist() == np.array([0.5, 0.7], np.float32).tolist()
        assert ds[3]["semantic_label"].tolist() == [1, 0]

        sampler = ExactMixtureSampler(
            ds,
            {
                "semantic_kitti": 0.25,
                "rellis_3d": 0.25,
                "semantic_stf": 0.25,
                "nuscenes_mini": 0.25,
            },
            epoch_size=40,
            seed=42,
        )
        indices = list(iter(sampler))
        assert len(indices) == 40
        names = [ds.rows[i]["dataset"] for i in indices]
        for name in sampler.quotas:
            assert names.count(name) == 10

        ds.close()

    print("PASS: test_f0_tar_dataset")


if __name__ == "__main__":
    main()
