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


def normalize_frame_reference(
    value: str,
) -> str:
    """
    Convert a split-list entry into a filename stem.

    Examples:
        /path/000123.bin -> 000123
        000123.label     -> 000123
        000123           -> 000123
    """

    value = value.strip()

    if not value:
        return ""

    return Path(value).stem


def read_rellis_split_ids(
    path: Path,
) -> Set[str]:

    ids = set()

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as file:

        for line in file:
            stripped = line.strip()

            if not stripped:
                continue

            # Some split files contain more than one column.
            parts = stripped.split()

            candidates = {
                normalize_frame_reference(part)
                for part in parts
            }

            candidates.discard("")

            ids.update(candidates)

    return ids


def subset_rellis_dataset(
    dataset: RELLISDataset,
    allowed_ids: Set[str],
) -> List[int]:

    indices = []

    for index, sample in enumerate(
        dataset.samples
    ):
        frame_id = str(
            sample["frame_id"]
        )

        if frame_id in allowed_ids:
            indices.append(index)

    return indices


def prepare_rellis(
    config: Dict,
    roots: Dict,
):
    root = roots["rellis_3d"]

    mapping_file = resolve_repo_path(
        config["mappings"]["rellis_3d"]
    )

    dataset = RELLISDataset(
        root=root,
        mapping_file=mapping_file,
        strict_labels=True,
    )

    lists = find_rellis_split_lists(
        root
    )

    ids = {
        split: read_rellis_split_ids(path)
        for split, path in lists.items()
    }

    # --------------------------------------------------------
    # Leakage check on list references
    # --------------------------------------------------------

    if ids["train"] & ids["val"]:
        raise RuntimeError(
            "RELLIS train/val overlap detected."
        )

    if ids["train"] & ids["test"]:
        raise RuntimeError(
            "RELLIS train/test overlap detected."
        )

    if ids["val"] & ids["test"]:
        raise RuntimeError(
            "RELLIS val/test overlap detected."
        )

    indices = {
        split: subset_rellis_dataset(
            dataset,
            split_ids,
        )
        for split, split_ids in ids.items()
    }

    for split in (
        "train",
        "val",
        "test",
    ):
        if not indices[split]:
            raise RuntimeError(
                "RELLIS official split produced "
                f"zero matched frames for '{split}'."
            )

    return {
        "dataset": dataset,
        "indices": indices,
        "list_files": {
            key: str(path)
            for key, path in lists.items()
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

    counter = Counter()

    for index in indices:
        sample = dataset[index]

        labels = np.asarray(
            sample[
                "semantic_label"
            ]
        )

        ids, counts = np.unique(
            labels,
            return_counts=True,
        )

        for class_id, count in zip(
            ids,
            counts,
        ):
            counter[
                int(class_id)
            ] += int(count)

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