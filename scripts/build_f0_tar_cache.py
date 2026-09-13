from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


# ============================================================
# Repository setup
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from datasets.paths import validate_dataset_roots
from scripts.prepare_f0_data import (
    load_yaml,
    prepare_nuscenes,
    prepare_rellis,
    prepare_semantic_kitti,
    prepare_semantic_stf,
    resolve_repo_path,
)


# ============================================================
# Constants
# ============================================================

NUM_CLASSES = 23

EXPECTED_TRAIN_FRAMES = {
    "semantic_kitti": 19130,
    "semantic_stf": 1326,
    "rellis_3d": 7800,
    "nuscenes_mini": 282,
}


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build lossless, resumable TAR shards for F0 training. "
            "Original scan and label bytes are preserved exactly. "
            "Exact TRAIN class frequencies are computed during the "
            "same pass."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/f0_data.yaml",
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        choices=(
            "train",
            "val",
            "test",
        ),
        default=[
            "train",
        ],
        help=(
            "Splits to cache. "
            "Start with train. Add val/test later."
        ),
    )

    parser.add_argument(
        "--frames-per-shard",
        type=int,
        default=250,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--read-batch-size",
        type=int,
        default=8,
        help=(
            "How many samples are prefetched before "
            "serial TAR writing."
        ),
    )

    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=2.0,
        help=(
            "Pause after each completed shard to reduce "
            "Google Drive throttling."
        ),
    )

    parser.add_argument(
        "--local-dir",
        default=None,
        help=(
            "Default on Colab: /content/f0_tar_cache"
        ),
    )

    parser.add_argument(
        "--mirror-dir",
        default=None,
        help=(
            "Persistent mirror. "
            "Default on Colab: "
            "/content/drive/MyDrive/SIH26053/f0_tar_cache. "
            "Use 'none' to disable."
        ),
    )

    parser.add_argument(
        "--keep-local",
        action="store_true",
        help=(
            "Keep local shards after copying them "
            "to the persistent mirror."
        ),
    )

    return parser.parse_args()


# ============================================================
# Generic helpers
# ============================================================

def save_json(
    path: Path,
    payload: Dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with open(
        temp_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
        )

    temp_path.replace(
        path
    )


def sha256_bytes(
    data: bytes,
) -> str:
    return hashlib.sha256(
        data
    ).hexdigest()


def add_bytes_to_tar(
    tar: tarfile.TarFile,
    arcname: str,
    data: bytes,
) -> None:
    info = tarfile.TarInfo(
        name=arcname
    )

    info.size = len(
        data
    )

    # Keep TAR deterministic.
    info.mtime = 0
    info.mode = 0o644

    tar.addfile(
        info,
        io.BytesIO(
            data
        ),
    )


# ============================================================
# Sample identity
# ============================================================

def sample_identity(
    dataset_name: str,
    sample: Dict,
    index: int,
) -> str:

    if dataset_name == "semantic_kitti":
        return (
            f"{sample['sequence']}_"
            f"{sample['frame_id']}"
        )

    if dataset_name == "rellis_3d":
        return (
            f"{str(sample['sequence']).zfill(5)}_"
            f"{sample['frame_id']}"
        )

    if dataset_name == "semantic_stf":
        return (
            f"{sample.get('split', 'unknown')}_"
            f"{sample['frame_id']}"
        )

    if dataset_name == "nuscenes_mini":
        return str(
            sample.get(
                "token",
                index,
            )
        )

    return f"{index:08d}"


# ============================================================
# Native label decoding
# ============================================================

def decode_native_labels(
    dataset_name: str,
    label_bytes: bytes,
) -> np.ndarray:

    # nuScenes lidarseg labels are uint8.
    if dataset_name == "nuscenes_mini":
        return np.frombuffer(
            label_bytes,
            dtype=np.uint8,
        )

    # SemanticKITTI, SemanticSTF and RELLIS
    # use uint32 label storage.
    packed = np.frombuffer(
        label_bytes,
        dtype=np.uint32,
    )

    # SemanticKITTI stores semantic ID in lower 16 bits.
    if dataset_name == "semantic_kitti":
        return (
            packed & 0xFFFF
        ).astype(
            np.uint16,
            copy=False,
        )

    return packed


# ============================================================
# Read one complete sample
# ============================================================

def read_sample_payload(
    task: Tuple[
        str,
        str,
        object,
        int,
    ],
) -> Dict:

    (
        dataset_name,
        split,
        dataset,
        index,
    ) = task

    sample = dataset.samples[
        index
    ]

    scan_path = Path(
        sample["scan_path"]
    )

    label_path = Path(
        sample["label_path"]
    )

    # --------------------------------------------------------
    # Important:
    # These bytes are copied EXACTLY.
    #
    # No conversion.
    # No normalization.
    # No resampling.
    # No concatenation.
    # --------------------------------------------------------

    scan_bytes = (
        scan_path.read_bytes()
    )

    label_bytes = (
        label_path.read_bytes()
    )

    # --------------------------------------------------------
    # Exact class histogram
    # --------------------------------------------------------

    native_labels = (
        decode_native_labels(
            dataset_name,
            label_bytes,
        )
    )

    semantic_labels = (
        dataset._remap_labels(
            native_labels
        )
    )

    counts = np.bincount(
        np.asarray(
            semantic_labels,
            dtype=np.int64,
        ),
        minlength=NUM_CLASSES,
    )[:NUM_CLASSES]

    identity = sample_identity(
        dataset_name,
        sample,
        index,
    )

    safe_identity = (
        identity
        .replace("/", "_")
        .replace("\\", "_")
    )

    base = (
        f"{dataset_name}/"
        f"{split}/"
        f"{safe_identity}"
    )

    return {
        "dataset_name":
            dataset_name,

        "split":
            split,

        "index":
            int(index),

        "identity":
            identity,

        "scan_path":
            str(scan_path),

        "label_path":
            str(label_path),

        "scan_arcname":
            (
                f"{base}/"
                f"scan{scan_path.suffix}"
            ),

        "label_arcname":
            (
                f"{base}/"
                f"label{label_path.suffix}"
            ),

        "scan_bytes":
            scan_bytes,

        "label_bytes":
            label_bytes,

        "scan_sha256":
            sha256_bytes(
                scan_bytes
            ),

        "label_sha256":
            sha256_bytes(
                label_bytes
            ),

        "scan_size":
            len(
                scan_bytes
            ),

        "label_size":
            len(
                label_bytes
            ),

        "counts":
            counts,
    }


# ============================================================
# Prepare frozen F0 splits
# ============================================================

def prepare_all_splits(
    config: Dict,
    roots: Dict,
):
    semantic_kitti = (
        prepare_semantic_kitti(
            config,
            roots,
        )
    )

    semantic_stf = (
        prepare_semantic_stf(
            config,
            roots,
        )
    )

    rellis = (
        prepare_rellis(
            config,
            roots,
        )
    )

    nuscenes = (
        prepare_nuscenes(
            config,
            roots,
        )
    )

    actual_train = {
        "semantic_kitti":
            len(
                semantic_kitti[
                    "train"
                ]
            ),

        "semantic_stf":
            len(
                semantic_stf[
                    "train"
                ]
            ),

        "rellis_3d":
            len(
                rellis[
                    "indices"
                ][
                    "train"
                ]
            ),

        "nuscenes_mini":
            len(
                nuscenes[
                    "indices"
                ][
                    "train"
                ]
            ),
    }

    print()
    print(
        "Verifying frozen TRAIN counts..."
    )

    for (
        dataset_name,
        expected,
    ) in EXPECTED_TRAIN_FRAMES.items():

        actual = (
            actual_train[
                dataset_name
            ]
        )

        print(
            f"  "
            f"{dataset_name:18s}: "
            f"{actual:,}"
        )

        if actual != expected:
            raise RuntimeError(
                f"Frozen TRAIN count changed "
                f"for {dataset_name}. "
                f"Expected {expected:,}, "
                f"got {actual:,}."
            )

    print(
        "PASS: frozen TRAIN counts unchanged."
    )

    return (
        semantic_kitti,
        semantic_stf,
        rellis,
        nuscenes,
    )


# ============================================================
# Build deterministic task list
# ============================================================

def build_tasks(
    config: Dict,
    roots: Dict,
    requested_splits: Iterable[str],
):

    (
        semantic_kitti,
        semantic_stf,
        rellis,
        nuscenes,
    ) = prepare_all_splits(
        config,
        roots,
    )

    tasks: List[
        Tuple[
            str,
            str,
            object,
            int,
        ]
    ] = []

    requested_splits = list(
        requested_splits
    )

    for split in requested_splits:

        # ----------------------------------------------------
        # SemanticKITTI
        # ----------------------------------------------------

        semantic_kitti_dataset = (
            semantic_kitti.get(
                split
            )
        )

        if (
            semantic_kitti_dataset
            is not None
        ):
            tasks.extend(
                (
                    "semantic_kitti",
                    split,
                    semantic_kitti_dataset,
                    index,
                )
                for index in range(
                    len(
                        semantic_kitti_dataset
                    )
                )
            )

        # ----------------------------------------------------
        # SemanticSTF
        # ----------------------------------------------------

        semantic_stf_dataset = (
            semantic_stf.get(
                split
            )
        )

        if (
            semantic_stf_dataset
            is not None
        ):
            tasks.extend(
                (
                    "semantic_stf",
                    split,
                    semantic_stf_dataset,
                    index,
                )
                for index in range(
                    len(
                        semantic_stf_dataset
                    )
                )
            )

        # ----------------------------------------------------
        # RELLIS
        # ----------------------------------------------------

        tasks.extend(
            (
                "rellis_3d",
                split,
                rellis[
                    "dataset"
                ],
                index,
            )
            for index in (
                rellis[
                    "indices"
                ][
                    split
                ]
            )
        )

        # ----------------------------------------------------
        # nuScenes
        # ----------------------------------------------------

        tasks.extend(
            (
                "nuscenes_mini",
                split,
                nuscenes[
                    "dataset"
                ],
                index,
            )
            for index in (
                nuscenes[
                    "indices"
                ][
                    split
                ]
            )
        )

    return tasks


# ============================================================
# Resolve local and persistent storage
# ============================================================

def resolve_dirs(
    args: argparse.Namespace,
) -> Tuple[
    Path,
    Path | None,
]:

    # --------------------------------------------------------
    # Local working cache
    # --------------------------------------------------------

    if args.local_dir:
        local_dir = Path(
            args.local_dir
        )

    elif Path(
        "/content"
    ).exists():
        local_dir = Path(
            "/content/f0_tar_cache"
        )

    else:
        local_dir = (
            REPO_ROOT
            / "outputs"
            / "f0_tar_cache"
        )

    # --------------------------------------------------------
    # Persistent mirror
    # --------------------------------------------------------

    if (
        args.mirror_dir
        is not None
    ):

        if (
            args.mirror_dir.lower()
            == "none"
        ):
            mirror_dir = None

        else:
            mirror_dir = Path(
                args.mirror_dir
            )

    elif Path(
        "/content/drive/MyDrive"
    ).exists():

        mirror_dir = Path(
            "/content/drive/MyDrive/"
            "SIH26053/f0_tar_cache"
        )

    else:
        mirror_dir = None

    local_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if mirror_dir is not None:
        mirror_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    return (
        local_dir,
        mirror_dir,
    )


# ============================================================
# Shard path helpers
# ============================================================

def shard_paths(
    base_dir: Path,
    shard_id: int,
) -> Tuple[
    Path,
    Path,
]:

    stem = (
        f"f0_shard_"
        f"{shard_id:05d}"
    )

    return (
        base_dir
        / f"{stem}.tar",

        base_dir
        / f"{stem}.json",
    )


def existing_sidecar(
    local_dir: Path,
    mirror_dir: Path | None,
    shard_id: int,
) -> Path | None:

    (
        _,
        local_json,
    ) = shard_paths(
        local_dir,
        shard_id,
    )

    if local_json.exists():
        return local_json

    if mirror_dir is not None:

        (
            _,
            mirror_json,
        ) = shard_paths(
            mirror_dir,
            shard_id,
        )

        if mirror_json.exists():
            return mirror_json

    return None


# ============================================================
# Build one shard
# ============================================================

def build_one_shard(
    shard_id: int,
    shard_tasks: List[
        Tuple[
            str,
            str,
            object,
            int,
        ]
    ],
    local_dir: Path,
    mirror_dir: Path | None,
    workers: int,
    read_batch_size: int,
    keep_local: bool,
) -> Dict:

    (
        tar_path,
        sidecar_path,
    ) = shard_paths(
        local_dir,
        shard_id,
    )

    tar_tmp = (
        tar_path.with_suffix(
            ".tar.partial"
        )
    )

    # Remove incomplete leftovers.
    if tar_tmp.exists():
        tar_tmp.unlink()

    if tar_path.exists():
        tar_path.unlink()

    if sidecar_path.exists():
        sidecar_path.unlink()

    shard_counts = np.zeros(
        NUM_CLASSES,
        dtype=np.int64,
    )

    sample_rows = []

    started = (
        time.perf_counter()
    )

    # --------------------------------------------------------
    # Write uncompressed TAR.
    #
    # Compression is intentionally disabled.
    # We want fast sequential access and no CPU overhead.
    # --------------------------------------------------------

    with tarfile.open(
        tar_tmp,
        mode="w",
    ) as tar:

        with ThreadPoolExecutor(
            max_workers=workers
        ) as executor:

            for batch_start in range(
                0,
                len(shard_tasks),
                read_batch_size,
            ):

                batch = (
                    shard_tasks[
                        batch_start:
                        batch_start
                        + read_batch_size
                    ]
                )

                # executor.map preserves batch order.
                payloads = list(
                    executor.map(
                        read_sample_payload,
                        batch,
                    )
                )

                for payload in payloads:

                    add_bytes_to_tar(
                        tar,
                        payload[
                            "scan_arcname"
                        ],
                        payload[
                            "scan_bytes"
                        ],
                    )

                    add_bytes_to_tar(
                        tar,
                        payload[
                            "label_arcname"
                        ],
                        payload[
                            "label_bytes"
                        ],
                    )

                    # Exact TRAIN distribution only.
                    if (
                        payload[
                            "split"
                        ]
                        == "train"
                    ):
                        shard_counts += (
                            payload[
                                "counts"
                            ]
                        )

                    sample_rows.append(
                        {
                            "dataset":
                                payload[
                                    "dataset_name"
                                ],

                            "split":
                                payload[
                                    "split"
                                ],

                            "source_index":
                                payload[
                                    "index"
                                ],

                            "identity":
                                payload[
                                    "identity"
                                ],

                            "scan_arcname":
                                payload[
                                    "scan_arcname"
                                ],

                            "label_arcname":
                                payload[
                                    "label_arcname"
                                ],

                            "scan_sha256":
                                payload[
                                    "scan_sha256"
                                ],

                            "label_sha256":
                                payload[
                                    "label_sha256"
                                ],

                            "scan_size":
                                payload[
                                    "scan_size"
                                ],

                            "label_size":
                                payload[
                                    "label_size"
                                ],
                        }
                    )

                completed = min(
                    batch_start
                    + len(batch),
                    len(
                        shard_tasks
                    ),
                )

                print(
                    f"      shard "
                    f"{shard_id:05d}: "
                    f"{completed:4d}/"
                    f"{len(shard_tasks):4d} "
                    f"samples read + written",
                    flush=True,
                )

    # Rename only after the TAR completed successfully.
    tar_tmp.replace(
        tar_path
    )

    sidecar = {
        "version":
            1,

        "shard_id":
            shard_id,

        "format":
            "lossless_raw_tar",

        "compression":
            "none",

        "sample_count":
            len(
                sample_rows
            ),

        "tar_size_bytes":
            tar_path.stat().st_size,

        "train_class_counts":
            {
                str(class_id):
                    int(
                        shard_counts[
                            class_id
                        ]
                    )
                for class_id in range(
                    NUM_CLASSES
                )
            },

        "samples":
            sample_rows,

        "build_seconds":
            (
                time.perf_counter()
                - started
            ),
    }

    save_json(
        sidecar_path,
        sidecar,
    )

    # --------------------------------------------------------
    # Mirror completed shard to persistent Drive.
    # --------------------------------------------------------

    if mirror_dir is not None:

        (
            mirror_tar,
            mirror_json,
        ) = shard_paths(
            mirror_dir,
            shard_id,
        )

        print(
            f"      mirroring shard "
            f"{shard_id:05d} "
            f"to Drive...",
            flush=True,
        )

        shutil.copy2(
            tar_path,
            mirror_tar,
        )

        shutil.copy2(
            sidecar_path,
            mirror_json,
        )

        # Basic transfer validation.
        if (
            mirror_tar.stat().st_size
            != tar_path.stat().st_size
        ):
            raise RuntimeError(
                f"Mirror size mismatch "
                f"for shard "
                f"{shard_id:05d}"
            )

        # Free Colab local disk unless explicitly retained.
        if not keep_local:

            tar_path.unlink()

            sidecar_path.unlink()

    return sidecar


# ============================================================
# Load completed sidecars
# ============================================================

def collect_sidecars(
    local_dir: Path,
    mirror_dir: Path | None,
    total_shards: int,
) -> List[Dict]:

    rows = []

    for shard_id in range(
        total_shards
    ):

        path = existing_sidecar(
            local_dir,
            mirror_dir,
            shard_id,
        )

        if path is None:
            raise RuntimeError(
                f"Missing sidecar for "
                f"completed shard "
                f"{shard_id:05d}"
            )

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:

            rows.append(
                json.load(
                    file
                )
            )

    return rows


# ============================================================
# Exact class distribution from shard metadata
# ============================================================

def build_exact_distribution(
    sidecars: List[Dict],
    config: Dict,
) -> Dict:

    counts = np.zeros(
        NUM_CLASSES,
        dtype=np.int64,
    )

    for sidecar in sidecars:

        shard_counts = (
            sidecar.get(
                "train_class_counts",
                {},
            )
        )

        for class_id in range(
            NUM_CLASSES
        ):

            counts[
                class_id
            ] += int(
                shard_counts.get(
                    str(
                        class_id
                    ),
                    0,
                )
            )

    total = int(
        counts.sum()
    )

    ontology = load_yaml(
        resolve_repo_path(
            config[
                "ontology"
            ][
                "path"
            ]
        )
    )

    classes = ontology[
        "classes"
    ]

    combined = {}

    print()
    print("=" * 80)
    print(
        "EXACT TRAIN CLASS DISTRIBUTION"
    )
    print("=" * 80)

    print(
        f"{'ID':>3}  "
        f"{'Class':25s}  "
        f"{'Points':>15s}  "
        f"{'%':>10s}"
    )

    print("-" * 62)

    for class_id in range(
        NUM_CLASSES
    ):

        points = int(
            counts[
                class_id
            ]
        )

        percentage = (
            100.0
            * points
            / total
            if total
            else 0.0
        )

        name = (
            classes[
                class_id
            ][
                "name"
            ]
        )

        combined[
            str(
                class_id
            )
        ] = {
            "name":
                name,

            "points":
                points,

            "percentage":
                percentage,
        }

        print(
            f"{class_id:3d}  "
            f"{name:25s}  "
            f"{points:15,d}  "
            f"{percentage:9.4f}%"
        )

    return {
        "distribution_type":
            "exact_from_lossless_tar_build",

        "num_classes":
            NUM_CLASSES,

        "ignore_index":
            0,

        "total_points":
            total,

        "combined":
            combined,
    }


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    if (
        args.frames_per_shard
        <= 0
    ):
        raise ValueError(
            "--frames-per-shard "
            "must be > 0"
        )

    if args.workers <= 0:
        raise ValueError(
            "--workers must be > 0"
        )

    if (
        args.read_batch_size
        <= 0
    ):
        raise ValueError(
            "--read-batch-size "
            "must be > 0"
        )

    config = load_yaml(
        resolve_repo_path(
            args.config
        )
    )

    roots = (
        validate_dataset_roots()
    )

    (
        local_dir,
        mirror_dir,
    ) = resolve_dirs(
        args
    )

    print("=" * 80)
    print(
        "SIH26053 - LOSSLESS F0 TAR CACHE BUILDER"
    )
    print("=" * 80)

    print(
        f"Splits            : "
        f"{', '.join(args.splits)}"
    )

    print(
        f"Frames / shard    : "
        f"{args.frames_per_shard}"
    )

    print(
        f"Read workers      : "
        f"{args.workers}"
    )

    print(
        f"Read batch size   : "
        f"{args.read_batch_size}"
    )

    print(
        f"Cooldown          : "
        f"{args.cooldown_seconds:.1f}s"
    )

    print(
        f"Local directory   : "
        f"{local_dir}"
    )

    print(
        f"Persistent mirror : "
        f"{mirror_dir if mirror_dir else 'disabled'}"
    )

    print()

    print(
        "IMPORTANT: TAR is storage-only."
    )

    print(
        "Scan and label bytes are NOT "
        "converted, normalized, resampled "
        "or concatenated."
    )

    # --------------------------------------------------------
    # Create deterministic frozen task list
    # --------------------------------------------------------

    tasks = build_tasks(
        config,
        roots,
        args.splits,
    )

    total_frames = len(
        tasks
    )

    total_shards = (
        total_frames
        + args.frames_per_shard
        - 1
    ) // args.frames_per_shard

    print()

    print(
        f"Frames selected    : "
        f"{total_frames:,}"
    )

    print(
        f"Shards required    : "
        f"{total_shards:,}"
    )

    overall_start = (
        time.perf_counter()
    )

    # --------------------------------------------------------
    # Build shards
    # --------------------------------------------------------

    for shard_id in range(
        total_shards
    ):

        start = (
            shard_id
            * args.frames_per_shard
        )

        end = min(
            start
            + args.frames_per_shard,
            total_frames,
        )

        shard_tasks = (
            tasks[
                start:end
            ]
        )

        found = existing_sidecar(
            local_dir,
            mirror_dir,
            shard_id,
        )

        if found is not None:

            print(
                f"["
                f"{shard_id + 1:4d}/"
                f"{total_shards:4d}"
                f"] "
                f"SKIP completed shard "
                f"{shard_id:05d} "
                f"({found})"
            )

            continue

        print()

        print(
            f"["
            f"{shard_id + 1:4d}/"
            f"{total_shards:4d}"
            f"] "
            f"BUILD shard "
            f"{shard_id:05d}"
        )

        print(
            f"      samples "
            f"{start:,}.."
            f"{end - 1:,}"
        )

        shard_start = (
            time.perf_counter()
        )

        build_one_shard(
            shard_id=
                shard_id,

            shard_tasks=
                shard_tasks,

            local_dir=
                local_dir,

            mirror_dir=
                mirror_dir,

            workers=
                args.workers,

            read_batch_size=
                args.read_batch_size,

            keep_local=
                args.keep_local,
        )

        elapsed = (
            time.perf_counter()
            - shard_start
        )

        print(
            f"      PASS shard "
            f"{shard_id:05d} "
            f"| {elapsed:.1f}s"
        )

        if (
            args.cooldown_seconds
            > 0
        ):
            time.sleep(
                args.cooldown_seconds
            )

    # --------------------------------------------------------
    # Reconstruct global metadata
    # --------------------------------------------------------

    sidecars = collect_sidecars(
        local_dir,
        mirror_dir,
        total_shards,
    )

    manifest = {
        "version":
            1,

        "format":
            "lossless_raw_tar",

        "compression":
            "none",

        "splits":
            args.splits,

        "frames_per_shard":
            args.frames_per_shard,

        "total_frames":
            total_frames,

        "total_shards":
            total_shards,

        "shards":
            [
                {
                    "shard_id":
                        row[
                            "shard_id"
                        ],

                    "sample_count":
                        row[
                            "sample_count"
                        ],

                    "tar_size_bytes":
                        row[
                            "tar_size_bytes"
                        ],
                }

                for row in sidecars
            ],
    }

    distribution = (
        build_exact_distribution(
            sidecars,
            config,
        )
    )

    # --------------------------------------------------------
    # Store final metadata
    # --------------------------------------------------------

    if mirror_dir is not None:
        output_base = (
            mirror_dir
        )
    else:
        output_base = (
            local_dir
        )

    manifest_path = (
        output_base
        / "f0_tar_manifest.json"
    )

    distribution_path = (
        output_base
        / "f0_class_distribution_exact.json"
    )

    save_json(
        manifest_path,
        manifest,
    )

    save_json(
        distribution_path,
        distribution,
    )

    runtime = (
        time.perf_counter()
        - overall_start
    )

    print()
    print("=" * 80)
    print(
        "F0 TAR CACHE: PASS"
    )
    print("=" * 80)

    print(
        f"Runtime      : "
        f"{runtime:.1f}s"
    )

    print(
        f"Manifest     : "
        f"{manifest_path}"
    )

    print(
        f"Exact counts : "
        f"{distribution_path}"
    )

    print()

    print(
        "Each sample sidecar contains "
        "SHA256 hashes for the original "
        "scan and label bytes."
    )

    print(
        "This allows byte-for-byte "
        "verification after extraction."
    )


if __name__ == "__main__":
    main()