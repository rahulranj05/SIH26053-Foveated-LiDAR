from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import yaml


# ============================================================
# Repository imports
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from datasets.paths import validate_dataset_roots
from datasets.semantic_kitti import SemanticKITTIDataset
from datasets.rellis import RELLISDataset
from datasets.semantic_stf import SemanticSTFDataset
from datasets.nuscenes_mini import NuScenesMiniDataset


# ============================================================
# General helpers
# ============================================================

def load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid YAML document: {path}"
        )

    return data


def resolve_repo_path(path_string: str) -> Path:
    path = Path(path_string)

    if path.is_absolute():
        return path

    return REPO_ROOT / path


def is_colab() -> bool:
    return Path("/content").exists()


def get_output_directory(config: Dict) -> Path:
    if is_colab():
        output = Path(
            config["output"]["colab_directory"]
        )
    else:
        output = resolve_repo_path(
            config["output"]["local_directory"]
        )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output


def save_json(
    path: Path,
    data: Dict,
):
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
        )


def load_mapping(
    config: Dict,
    dataset_name: str,
) -> Dict[int, int]:

    path = resolve_repo_path(
        config["mappings"][dataset_name]
    )

    raw = load_yaml(path)["mapping"]

    return {
        int(native): int(unified)
        for native, unified in raw.items()
    }


def distributed_indices(
    length: int,
    count: int,
    seed: int,
) -> List[int]:

    if count >= length:
        return list(range(length))

    rng = random.Random(seed)

    anchors = {
        0,
        length // 4,
        length // 2,
        3 * length // 4,
        length - 1,
    }

    remaining = [
        index
        for index in range(length)
        if index not in anchors
    ]

    required = max(
        0,
        count - len(anchors),
    )

    anchors.update(
        rng.sample(
            remaining,
            min(required, len(remaining)),
        )
    )

    return sorted(anchors)[:count]


# ============================================================
# SemanticKITTI split
# ============================================================

def prepare_semantic_kitti(
    config: Dict,
    roots: Dict,
):
    split_cfg = (
        config["splits"]["semantic_kitti"]
    )

    mapping_file = resolve_repo_path(
        config["mappings"]["semantic_kitti"]
    )

    train_sequences = [
        str(x).zfill(2)
        for x
        in split_cfg["train_sequences"]
    ]

    val_sequences = [
        str(x).zfill(2)
        for x
        in split_cfg["val_sequences"]
    ]

    test_sequences = [
        str(x).zfill(2)
        for x
        in split_cfg["test_sequences"]
    ]

    if (
        set(train_sequences)
        & set(val_sequences)
    ):
        raise RuntimeError(
            "SemanticKITTI train/val overlap."
        )

    train = SemanticKITTIDataset(
        root=roots["semantic_kitti"],
        mapping_file=mapping_file,
        sequences=train_sequences,
        strict_labels=True,
    )

    val = SemanticKITTIDataset(
        root=roots["semantic_kitti"],
        mapping_file=mapping_file,
        sequences=val_sequences,
        strict_labels=True,
    )

    test = None

    if test_sequences:
        test = SemanticKITTIDataset(
            root=roots["semantic_kitti"],
            mapping_file=mapping_file,
            sequences=test_sequences,
            strict_labels=True,
        )

    return {
        "train": train,
        "val": val,
        "test": test,
    }


# ============================================================
# SemanticSTF split
# ============================================================

def prepare_semantic_stf(
    config: Dict,
    roots: Dict,
):
    split_cfg = (
        config["splits"]["semantic_stf"]
    )

    mapping_file = resolve_repo_path(
        config["mappings"]["semantic_stf"]
    )

    return {
        "train": SemanticSTFDataset(
            root=roots["semantic_stf"],
            mapping_file=mapping_file,
            splits=[
                split_cfg["train_split"]
            ],
            strict_labels=True,
        ),

        "val": SemanticSTFDataset(
            root=roots["semantic_stf"],
            mapping_file=mapping_file,
            splits=[
                split_cfg["val_split"]
            ],
            strict_labels=True,
        ),

        "test": SemanticSTFDataset(
            root=roots["semantic_stf"],
            mapping_file=mapping_file,
            splits=[
                split_cfg["test_split"]
            ],
            strict_labels=True,
        ),
    }


# ============================================================
# RELLIS split discovery
# ============================================================
def find_rellis_split_lists(
    loader_root: Path,
) -> Dict[str, Path]:
    """
    Discover official RELLIS split-list files.

    The LiDAR loader root is usually:

        .../RELLIS_3D/extracted/Rellis-3D

    But the split-list files may live elsewhere under the
    broader RELLIS_3D dataset directory.

    Therefore search progressively upward from the loader root.
    """

    search_roots = [
        loader_root,
        loader_root.parent,
        loader_root.parent.parent,
    ]

    list_files = []

    for search_root in search_roots:
        if not search_root.exists():
            continue

        found = sorted(
            search_root.rglob("*.lst")
        )

        if found:
            list_files = found
            break

    if not list_files:
        raise FileNotFoundError(
            "No RELLIS .lst split files found.\n"
            "Searched:\n"
            + "\n".join(
                f"  {path}"
                for path in search_roots
            )
        )

    print()
    print("RELLIS split files discovered:")

    for path in list_files:
        print(
            f"  {path}"
        )

    result = {}

    for path in list_files:
        filename = (
            path.name
            .lower()
            .replace("-", "_")
        )

        if "train" in filename:
            split_name = "train"

        elif (
            "val" in filename
            or "valid" in filename
        ):
            split_name = "val"

        elif "test" in filename:
            split_name = "test"

        else:
            continue

        if split_name in result:
            raise RuntimeError(
                "Multiple RELLIS split-list files matched "
                f"'{split_name}'.\n"
                f"First : {result[split_name]}\n"
                f"Second: {path}"
            )

        result[split_name] = path

    missing = (
        {"train", "val", "test"}
        - set(result)
    )

    if missing:
        raise RuntimeError(
            "RELLIS .lst files were found, but the "
            "following split(s) could not be identified: "
            f"{sorted(missing)}\n\n"
            "Discovered files:\n"
            + "\n".join(
                f"  {path}"
                for path in list_files
            )
        )

    print()
    print("RELLIS split mapping:")

    for split_name in (
        "train",
        "val",
        "test",
    ):
        print(
            f"  {split_name:5s}: "
            f"{result[split_name]}"
        )

    return result


def parse_rellis_frame_reference(
    value: str,
) -> Tuple[str, str]:
    """
    Parse a RELLIS file reference into a unique frame key:

        (sequence, frame_id)

    Official RELLIS split-list entries look like:

        00000/os1_cloud_node_kitti_bin/000307.bin
        00000/os1_cloud_node_semantickitti_label_id/000307.label

    Frame numbers are reused across sequences, therefore frame_id alone
    is NOT a unique identifier.
    """

    value = value.strip().replace("\\", "/")

    if not value:
        raise ValueError(
            "Empty RELLIS frame reference."
        )

    parts = [
        part
        for part in value.split("/")
        if part
    ]

    if len(parts) < 2:
        raise ValueError(
            "Unable to extract RELLIS sequence from reference: "
            f"{value}"
        )

    sequence = None

    for part in parts:
        if (
            len(part) == 5
            and part.isdigit()
        ):
            sequence = part
            break

    if sequence is None:
        raise ValueError(
            "Unable to determine RELLIS sequence from reference: "
            f"{value}"
        )

    frame_id = Path(
        parts[-1]
    ).stem

    if not frame_id:
        raise ValueError(
            "Unable to determine RELLIS frame ID from reference: "
            f"{value}"
        )

    return (
        sequence,
        frame_id,
    )


def read_rellis_split_ids(
    path: Path,
) -> Set[Tuple[str, str]]:
    """
    Read one official RELLIS split list.

    Each line is expected to contain:

        <point-cloud-path> <label-path>

    Both references must identify the same (sequence, frame_id).

    Returns a set of unique:
        (sequence, frame_id)
    """

    frame_keys: Set[
        Tuple[str, str]
    ] = set()

    duplicate_count = 0
    line_count = 0

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):
            stripped = line.strip()

            if not stripped:
                continue

            line_count += 1

            parts = stripped.split()

            if len(parts) < 2:
                raise ValueError(
                    "Invalid RELLIS split-list line.\n"
                    f"File: {path}\n"
                    f"Line: {line_number}\n"
                    f"Content: {stripped}"
                )

            scan_reference = parts[0]
            label_reference = parts[1]

            scan_key = (
                parse_rellis_frame_reference(
                    scan_reference
                )
            )

            label_key = (
                parse_rellis_frame_reference(
                    label_reference
                )
            )

            if scan_key != label_key:
                raise RuntimeError(
                    "RELLIS scan/label reference mismatch.\n"
                    f"File : {path}\n"
                    f"Line : {line_number}\n"
                    f"Scan : {scan_reference}\n"
                    f"Label: {label_reference}\n"
                    f"Scan key : {scan_key}\n"
                    f"Label key: {label_key}"
                )

            if scan_key in frame_keys:
                duplicate_count += 1

            frame_keys.add(
                scan_key
            )

    if not frame_keys:
        raise RuntimeError(
            f"RELLIS split list is empty: {path}"
        )

    print(
        f"  Parsed {path.name:15s}: "
        f"{len(frame_keys):,} unique frames"
    )

    if duplicate_count:
        print(
            f"    WARNING: {duplicate_count:,} "
            "duplicate entries were present."
        )

    if len(frame_keys) != (
        line_count - duplicate_count
    ):
        raise RuntimeError(
            "Unexpected RELLIS split parsing inconsistency."
        )

    return frame_keys


def subset_rellis_dataset(
    dataset: RELLISDataset,
    allowed_ids: Set[Tuple[str, str]],
) -> List[int]:
    """
    Match RELLIS loader samples against an official split.

    Identity is:
        (sequence, frame_id)

    NOT frame_id alone.
    """

    indices = []

    matched_keys = set()

    for index, sample in enumerate(
        dataset.samples
    ):
        key = (
            str(sample["sequence"]).zfill(5),
            str(sample["frame_id"]),
        )

        if key in allowed_ids:
            indices.append(
                index
            )

            matched_keys.add(
                key
            )

    missing = (
        allowed_ids
        - matched_keys
    )

    if missing:
        examples = sorted(
            missing
        )[:10]

        raise RuntimeError(
            "Some official RELLIS split frames were not "
            "found by RELLISDataset.\n"
            f"Expected : {len(allowed_ids):,}\n"
            f"Matched  : {len(matched_keys):,}\n"
            f"Missing  : {len(missing):,}\n"
            f"Examples : {examples}"
        )

    return indices

def prepare_rellis(
    config: Dict,
    roots: Dict,
):
    """
    Prepare RELLIS using its official train/val/test split lists.

    Split identity is based on:
        (sequence, frame_id)

    This prevents false leakage caused by identical frame numbers
    occurring in different RELLIS sequences.
    """

    root = Path(
        roots["rellis_3d"]
    )

    mapping_file = resolve_repo_path(
        config["mappings"]["rellis_3d"]
    )

    print()
    print("Preparing official RELLIS splits...")

    dataset = RELLISDataset(
        root=root,
        mapping_file=mapping_file,
        strict_labels=True,
    )

    print(
        f"  Loader samples : {len(dataset):,}"
    )

    lists = find_rellis_split_lists(
        root
    )

    print()
    print("Parsing official RELLIS split lists:")

    ids = {
        split: read_rellis_split_ids(
            path
        )
        for split, path
        in lists.items()
    }

    # ========================================================
    # Strict leakage checks
    # ========================================================

    train_val_overlap = (
        ids["train"]
        & ids["val"]
    )

    train_test_overlap = (
        ids["train"]
        & ids["test"]
    )

    val_test_overlap = (
        ids["val"]
        & ids["test"]
    )

    if train_val_overlap:
        raise RuntimeError(
            "REAL RELLIS train/val overlap detected.\n"
            f"Count: {len(train_val_overlap):,}\n"
            f"Examples: "
            f"{sorted(train_val_overlap)[:10]}"
        )

    if train_test_overlap:
        raise RuntimeError(
            "REAL RELLIS train/test overlap detected.\n"
            f"Count: {len(train_test_overlap):,}\n"
            f"Examples: "
            f"{sorted(train_test_overlap)[:10]}"
        )

    if val_test_overlap:
        raise RuntimeError(
            "REAL RELLIS val/test overlap detected.\n"
            f"Count: {len(val_test_overlap):,}\n"
            f"Examples: "
            f"{sorted(val_test_overlap)[:10]}"
        )

    print()
    print(
        "  PASS: train/val overlap  = 0"
    )
    print(
        "  PASS: train/test overlap = 0"
    )
    print(
        "  PASS: val/test overlap   = 0"
    )

    # ========================================================
    # Match official lists against loader samples
    # ========================================================

    indices = {
        split: subset_rellis_dataset(
            dataset,
            split_ids,
        )
        for split, split_ids
        in ids.items()
    }

    print()
    print("RELLIS matched split counts:")

    for split in (
        "train",
        "val",
        "test",
    ):
        expected = len(
            ids[split]
        )

        matched = len(
            indices[split]
        )

        print(
            f"  {split:5s}: "
            f"{matched:,} / {expected:,}"
        )

        if matched != expected:
            raise RuntimeError(
                "RELLIS split count mismatch.\n"
                f"Split    : {split}\n"
                f"Expected : {expected:,}\n"
                f"Matched  : {matched:,}"
            )

        if matched == 0:
            raise RuntimeError(
                "RELLIS official split produced "
                f"zero frames for '{split}'."
            )

    # ========================================================
    # Verify split union does not contain duplicate identities
    # ========================================================

    total_split_frames = (
        len(ids["train"])
        + len(ids["val"])
        + len(ids["test"])
    )

    union = (
        ids["train"]
        | ids["val"]
        | ids["test"]
    )

    if len(union) != total_split_frames:
        raise RuntimeError(
            "RELLIS split union contains duplicate "
            "frame identities."
        )

    print(
        f"  PASS: total official split frames "
        f"= {total_split_frames:,}"
    )

    return {
        "dataset": dataset,

        "indices": indices,

        "frame_keys": {
            split: sorted(
                split_ids
            )
            for split, split_ids
            in ids.items()
        },

        "list_files": {
            key: str(path)
            for key, path
            in lists.items()
        },
    }

# ============================================================
# nuScenes scene-level split
# ============================================================

def load_json(path: Path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def prepare_nuscenes(
    config: Dict,
    roots: Dict,
):
    seed = int(
        config["seed"]
    )

    split_cfg = (
        config["splits"]["nuscenes_mini"]
    )

    mapping_file = resolve_repo_path(
        config["mappings"]["nuscenes_mini"]
    )

    dataset = NuScenesMiniDataset(
        root=roots["nuscenes_mini"],
        mapping_file=mapping_file,
        strict_labels=True,
    )

    metadata_dir = (
        dataset.metadata_dir
    )

    sample_data = load_json(
        metadata_dir
        / "sample_data.json"
    )

    sample_table = load_json(
        metadata_dir
        / "sample.json"
    )

    # sample_data token -> sample token
    sample_data_to_sample = {
        row["token"]:
            row["sample_token"]
        for row in sample_data
    }

    # sample token -> scene token
    sample_to_scene = {
        row["token"]:
            row["scene_token"]
        for row in sample_table
    }

    frame_scene = {}

    for index, sample in enumerate(
        dataset.samples
    ):
        sample_data_token = (
            sample["token"]
        )

        sample_token = (
            sample_data_to_sample[
                sample_data_token
            ]
        )

        scene_token = (
            sample_to_scene[
                sample_token
            ]
        )

        frame_scene[index] = scene_token

    scenes = sorted(
        set(frame_scene.values())
    )

    rng = random.Random(seed)
    rng.shuffle(scenes)

    train_fraction = float(
        split_cfg["train_fraction"]
    )

    val_fraction = float(
        split_cfg["val_fraction"]
    )

    test_fraction = float(
        split_cfg["test_fraction"]
    )

    total_fraction = (
        train_fraction
        + val_fraction
        + test_fraction
    )

    if not np.isclose(
        total_fraction,
        1.0,
    ):
        raise RuntimeError(
            "nuScenes split fractions "
            "must sum to 1.0."
        )

    n_scenes = len(scenes)

    n_train = max(
        1,
        round(
            n_scenes
            * train_fraction
        ),
    )

    n_val = max(
        1,
        round(
            n_scenes
            * val_fraction
        ),
    )

    # Let test receive the remainder.
    if (
        n_train
        + n_val
        >= n_scenes
    ):
        n_val = max(
            1,
            n_scenes
            - n_train
            - 1,
        )

    train_scenes = set(
        scenes[:n_train]
    )

    val_scenes = set(
        scenes[
            n_train:
            n_train + n_val
        ]
    )

    test_scenes = set(
        scenes[
            n_train + n_val:
        ]
    )

    if (
        train_scenes & val_scenes
        or train_scenes & test_scenes
        or val_scenes & test_scenes
    ):
        raise RuntimeError(
            "nuScenes scene-level split leakage."
        )

    indices = {
        "train": [],
        "val": [],
        "test": [],
    }

    for index, scene in (
        frame_scene.items()
    ):
        if scene in train_scenes:
            indices["train"].append(
                index
            )
        elif scene in val_scenes:
            indices["val"].append(
                index
            )
        elif scene in test_scenes:
            indices["test"].append(
                index
            )

    for split in indices:
        if not indices[split]:
            raise RuntimeError(
                "nuScenes split produced "
                f"zero frames for {split}."
            )

    return {
        "dataset": dataset,
        "indices": indices,

        "scenes": {
            "train": sorted(
                train_scenes
            ),
            "val": sorted(
                val_scenes
            ),
            "test": sorted(
                test_scenes
            ),
        },
    }


# ============================================================
# Class distribution
# ============================================================

def count_loader_labels(
    dataset,
    indices: Iterable[int],
) -> Counter:
    """
    Count unified semantic labels without loading point-cloud geometry.

    This avoids calling dataset[index], which can read large LiDAR scans
    from Google Drive. For class-frequency estimation we only need the
    label files.
    """

    counter = Counter()

    indices = list(indices)
    total = len(indices)

    if total == 0:
        return counter

    start_time = time.perf_counter()

    for position, index in enumerate(
        indices,
        start=1,
    ):
        sample = dataset.samples[index]

        # ----------------------------------------------------
        # Fast path: dataset exposes label_path plus its own
        # native-label loader/remapper.
        # ----------------------------------------------------
        if (
            "label_path" in sample
            and hasattr(
                dataset,
                "_load_native_labels",
            )
            and hasattr(
                dataset,
                "_remap_labels",
            )
        ):
            native_labels = (
                dataset._load_native_labels(
                    Path(
                        sample["label_path"]
                    )
                )
            )

            semantic_labels = (
                dataset._remap_labels(
                    native_labels
                )
            )

        # ----------------------------------------------------
        # Fallback for loaders whose label representation is
        # not compatible with the generic fast path.
        # ----------------------------------------------------
        else:
            loaded = dataset[index]

            semantic_labels = np.asarray(
                loaded["semantic_label"]
            )

        unique, counts = np.unique(
            semantic_labels,
            return_counts=True,
        )

        for class_id, count in zip(
            unique,
            counts,
        ):
            counter[
                int(class_id)
            ] += int(count)

        # Progress report every 25 frames
        if (
            position % 25 == 0
            or position == total
        ):
            elapsed = (
                time.perf_counter()
                - start_time
            )

            rate = (
                position / elapsed
                if elapsed > 0
                else 0.0
            )

            print(
                f"    {position:4d}/{total:4d} "
                f"frames | "
                f"{elapsed:7.1f}s | "
                f"{rate:5.2f} frames/s",
                flush=True,
            )

    return counter

def select_distribution_indices(
    length: int,
    config: Dict,
    seed_offset: int,
) -> List[int]:

    mode = (
        config["distribution"]["mode"]
    )

    if mode == "exact":
        return list(
            range(length)
        )

    if mode != "sampled":
        raise RuntimeError(
            "distribution.mode must be "
            "'exact' or 'sampled'."
        )

    count = int(
        config["distribution"][
            "sampled_frames_per_dataset"
        ]
    )

    return distributed_indices(
        length,
        min(count, length),
        int(config["seed"])
        + seed_offset,
    )


def calculate_class_weights(
    counts: Dict[int, int],
    config: Dict,
) -> Dict[int, float]:

    weight_cfg = (
        config["class_weights"]
    )

    if (
        weight_cfg["method"]
        != "log_inverse_frequency"
    ):
        raise RuntimeError(
            "Unsupported class-weight method."
        )

    ignore_index = int(
        config["ontology"]["ignore_index"]
    )

    total = sum(
        count
        for class_id, count
        in counts.items()
        if class_id != ignore_index
    )

    offset = float(
        weight_cfg["log_offset"]
    )

    minimum = float(
        weight_cfg["min_weight"]
    )

    maximum = float(
        weight_cfg["max_weight"]
    )

    weights = {}

    for class_id in range(
        int(
            config["ontology"][
                "num_classes"
            ]
        )
    ):
        if class_id == ignore_index:
            weights[class_id] = float(
                weight_cfg[
                    "ignore_weight"
                ]
            )

            continue

        count = int(
            counts.get(
                class_id,
                0,
            )
        )

        if count == 0:
            # No direct training supervision.
            weights[class_id] = 0.0
            continue

        probability = (
            count
            / total
        )

        weight = (
            1.0
            / np.log(
                offset
                + probability
            )
        )

        weight = float(
            np.clip(
                weight,
                minimum,
                maximum,
            )
        )

        weights[
            class_id
        ] = weight

    return weights


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and freeze the SIH26053 "
            "F0 semantic training data protocol."
        )
    )

    parser.add_argument(
        "--config",
        default=(
            "configs/f0_data.yaml"
        ),
    )

    parser.add_argument(
        "--distribution-mode",
        choices=[
            "exact",
            "sampled",
        ],
        default=None,
        help=(
            "Override distribution.mode from YAML."
        ),
    )

    args = parser.parse_args()

    config_path = resolve_repo_path(
        args.config
    )

    config = load_yaml(
        config_path
    )

    if args.distribution_mode:
        config[
            "distribution"
        ][
            "mode"
        ] = args.distribution_mode

    output_dir = get_output_directory(
        config
    )

    roots = validate_dataset_roots()

    print("=" * 80)
    print(
        "SIH26053 — F0 DATA PREPARATION"
    )
    print("=" * 80)

    start = time.perf_counter()

    # --------------------------------------------------------
    # Build/freeze splits
    # --------------------------------------------------------

    print(
        "\n[1/5] Preparing splits..."
    )

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

    rellis = prepare_rellis(
        config,
        roots,
    )

    nuscenes = prepare_nuscenes(
        config,
        roots,
    )

    split_counts = {
        "semantic_kitti": {
            "train": len(
                semantic_kitti["train"]
            ),
            "val": len(
                semantic_kitti["val"]
            ),
            "test": 0,
        },

        "semantic_stf": {
            "train": len(
                semantic_stf["train"]
            ),
            "val": len(
                semantic_stf["val"]
            ),
            "test": len(
                semantic_stf["test"]
            ),
        },

        "rellis_3d": {
            split: len(indices)
            for split, indices
            in rellis["indices"].items()
        },

        "nuscenes_mini": {
            split: len(indices)
            for split, indices
            in nuscenes["indices"].items()
        },
    }

    for dataset_name, values in (
        split_counts.items()
    ):
        print(
            f"\n{dataset_name}"
        )

        for split, count in (
            values.items()
        ):
            print(
                f"  {split:5s}: "
                f"{count:,}"
            )

    # --------------------------------------------------------
    # Sampling policy
    # --------------------------------------------------------

    print(
        "\n[2/5] Training sampling mixture..."
    )

    probabilities = {
        name: float(value)
        for name, value
        in config[
            "sampling"
        ].items()
    }

    if not np.isclose(
        sum(probabilities.values()),
        1.0,
    ):
        raise RuntimeError(
            "Training sampling ratios "
            "must sum to 1.0."
        )

    for name, value in (
        probabilities.items()
    ):
        print(
            f"  {name:18s}: "
            f"{100 * value:.1f}%"
        )

    # --------------------------------------------------------
    # Unified class distribution
    # --------------------------------------------------------

    print(
        "\n[3/5] Computing unified TRAIN "
        "class distribution..."
    )

    combined = Counter()
    per_dataset = {}

    # SemanticKITTI
    sk_indices = (
        select_distribution_indices(
            len(
                semantic_kitti[
                    "train"
                ]
            ),
            config,
            100,
        )
    )

    sk_counts = count_loader_labels(
        semantic_kitti["train"],
        sk_indices,
    )

    per_dataset[
        "semantic_kitti"
    ] = sk_counts

    combined.update(
        sk_counts
    )

    # SemanticSTF
    stf_indices = (
        select_distribution_indices(
            len(
                semantic_stf[
                    "train"
                ]
            ),
            config,
            200,
        )
    )

    stf_counts = count_loader_labels(
        semantic_stf["train"],
        stf_indices,
    )

    per_dataset[
        "semantic_stf"
    ] = stf_counts

    combined.update(
        stf_counts
    )

    # RELLIS
    rellis_train_indices = (
        rellis[
            "indices"
        ][
            "train"
        ]
    )

    if (
        config[
            "distribution"
        ][
            "mode"
        ]
        == "sampled"
    ):
        local_indices = (
            distributed_indices(
                len(
                    rellis_train_indices
                ),
                min(
                    int(
                        config[
                            "distribution"
                        ][
                            "sampled_frames_per_dataset"
                        ]
                    ),
                    len(
                        rellis_train_indices
                    ),
                ),
                int(
                    config["seed"]
                )
                + 300,
            )
        )

        rellis_count_indices = [
            rellis_train_indices[
                index
            ]
            for index
            in local_indices
        ]

    else:
        rellis_count_indices = (
            rellis_train_indices
        )

    rellis_counts = (
        count_loader_labels(
            rellis[
                "dataset"
            ],
            rellis_count_indices,
        )
    )

    per_dataset[
        "rellis_3d"
    ] = rellis_counts

    combined.update(
        rellis_counts
    )

    # nuScenes
    nusc_train_indices = (
        nuscenes[
            "indices"
        ][
            "train"
        ]
    )

    if (
        config[
            "distribution"
        ][
            "mode"
        ]
        == "sampled"
    ):
        local_indices = (
            distributed_indices(
                len(
                    nusc_train_indices
                ),
                min(
                    int(
                        config[
                            "distribution"
                        ][
                            "sampled_frames_per_dataset"
                        ]
                    ),
                    len(
                        nusc_train_indices
                    ),
                ),
                int(
                    config["seed"]
                )
                + 400,
            )
        )

        nusc_count_indices = [
            nusc_train_indices[
                index
            ]
            for index
            in local_indices
        ]

    else:
        nusc_count_indices = (
            nusc_train_indices
        )

    nusc_counts = (
        count_loader_labels(
            nuscenes[
                "dataset"
            ],
            nusc_count_indices,
        )
    )

    per_dataset[
        "nuscenes_mini"
    ] = nusc_counts

    combined.update(
        nusc_counts
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

    classes = {
        int(class_id): info
        for class_id, info
        in ontology[
            "classes"
        ].items()
    }

    print()

    total_points = sum(
        combined.values()
    )

    print(
        f"{'ID':>3}  "
        f"{'Class':22s} "
        f"{'Points':>15s} "
        f"{'%':>9s}"
    )

    print("-" * 60)

    for class_id in range(
        int(
            config["ontology"][
                "num_classes"
            ]
        )
    ):
        count = int(
            combined.get(
                class_id,
                0,
            )
        )

        ratio = (
            100.0
            * count
            / total_points
        ) if total_points else 0.0

        print(
            f"{class_id:3d}  "
            f"{classes[class_id]['name']:22s} "
            f"{count:15,d} "
            f"{ratio:8.4f}%"
        )

    # --------------------------------------------------------
    # Weights
    # --------------------------------------------------------

    print(
        "\n[4/5] Deriving class weights..."
    )

    weights = (
        calculate_class_weights(
            combined,
            config,
        )
    )

    for class_id in range(
        len(classes)
    ):
        print(
            f"  {class_id:2d} "
            f"{classes[class_id]['name']:22s} "
            f"{weights[class_id]:.4f}"
        )

    # --------------------------------------------------------
    # Save frozen manifest
    # --------------------------------------------------------

    print(
        "\n[5/5] Saving F0 data manifest..."
    )

    manifest = {
        "version": 1,
        "seed": int(
            config["seed"]
        ),

        "sampling":
            probabilities,

        "split_counts":
            split_counts,

        "semantic_kitti": {
            "train_sequences":
                config[
                    "splits"
                ][
                    "semantic_kitti"
                ][
                    "train_sequences"
                ],

            "val_sequences":
                config[
                    "splits"
                ][
                    "semantic_kitti"
                ][
                    "val_sequences"
                ],

            "supervised_test_available":
                False,
        },

        "semantic_stf": {
            "train": "train",
            "val": "val",
            "test": "test",
        },

        "rellis_3d": {
            "official_split_lists":
                rellis[
                    "list_files"
                ],
        },

        "nuscenes_mini": {
            "split_level":
                "scene",

            "scenes":
                nuscenes[
                    "scenes"
                ],
        },

        "distribution_mode":
            config[
                "distribution"
            ][
                "mode"
            ],
    }

    distribution_json = {
        "combined": {
            str(class_id):
                int(
                    combined.get(
                        class_id,
                        0,
                    )
                )
            for class_id
            in range(
                len(classes)
            )
        },

        "per_dataset": {
            dataset_name: {
                str(class_id):
                    int(
                        counts.get(
                            class_id,
                            0,
                        )
                    )
                for class_id
                in range(
                    len(classes)
                )
            }
            for dataset_name, counts
            in per_dataset.items()
        },
    }

    weights_json = {
        str(class_id):
            float(weight)
        for class_id, weight
        in weights.items()
    }

    manifest_path = (
        output_dir
        / config[
            "output"
        ][
            "manifest_filename"
        ]
    )

    distribution_path = (
        output_dir
        / config[
            "output"
        ][
            "distribution_filename"
        ]
    )

    weights_path = (
        output_dir
        / config[
            "output"
        ][
            "weights_filename"
        ]
    )

    save_json(
        manifest_path,
        manifest,
    )

    save_json(
        distribution_path,
        distribution_json,
    )

    save_json(
        weights_path,
        weights_json,
    )

    runtime = (
        time.perf_counter()
        - start
    )

    print()
    print("=" * 80)
    print(
        "F0 DATA PREPARATION: PASS"
    )
    print("=" * 80)

    print(
        f"Runtime      : "
        f"{runtime:.1f} sec"
    )

    print(
        f"Manifest     : "
        f"{manifest_path}"
    )

    print(
        f"Distribution : "
        f"{distribution_path}"
    )

    print(
        f"Class weights: "
        f"{weights_path}"
    )

    print()
    print(
        "NEXT: inspect split counts and "
        "class distribution before Gate 2."
    )


if __name__ == "__main__":
    main()