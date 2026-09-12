from __future__ import annotations

import argparse
import errno
import json
import random
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import yaml


# ============================================================
# Ensure repository root is importable
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from datasets.paths import (  # noqa: E402
    validate_dataset_roots,
)

from datasets.semantic_kitti import (  # noqa: E402
    SemanticKITTIDataset,
)

from datasets.rellis import (  # noqa: E402
    RELLISDataset,
)

from datasets.semantic_stf import (  # noqa: E402
    SemanticSTFDataset,
)

from datasets.nuscenes_mini import (  # noqa: E402
    NuScenesMiniDataset,
)


# ============================================================
# Constants
# ============================================================

REQUIRED_SAMPLE_KEYS = {
    "xyz",
    "intensity",
    "native_label",
    "semantic_label",
    "dataset",
    "frame_id",
}


class Gate1Failure(RuntimeError):
    """
    Raised when a genuine Gate 1 validation check fails.
    """


# ============================================================
# Utility functions
# ============================================================

def load_yaml(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(
            f"YAML file not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
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


def get_output_directory(
    config: Dict,
) -> Path:

    output_config = config["output"]

    if is_colab():
        path = Path(
            output_config[
                "colab_directory"
            ]
        )
    else:
        path = resolve_repo_path(
            output_config[
                "local_directory"
            ]
        )

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def distributed_indices(
    length: int,
    count: int,
    seed: int,
) -> List[int]:
    """
    Select distributed indices:
    - beginning
    - middle
    - end
    - deterministic random samples
    """

    if length <= 0:
        return []

    if count >= length:
        return list(
            range(length)
        )

    selected = {
        0,
        length // 4,
        length // 2,
        (3 * length) // 4,
        length - 1,
    }

    selected = {
        index
        for index in selected
        if 0 <= index < length
    }

    rng = random.Random(seed)

    remaining = [
        index
        for index in range(length)
        if index not in selected
    ]

    needed = (
        count
        - len(selected)
    )

    if needed > 0:
        selected.update(
            rng.sample(
                remaining,
                min(
                    needed,
                    len(remaining),
                ),
            )
        )

    return sorted(selected)[:count]


def percentage(
    numerator: int,
    denominator: int,
) -> float:

    if denominator == 0:
        return 0.0

    return (
        100.0
        * numerator
        / denominator
    )


# ============================================================
# Ontology
# ============================================================

def load_and_validate_ontology(
    config: Dict,
):
    ontology_path = resolve_repo_path(
        config["ontology"]["path"]
    )

    ontology = load_yaml(
        ontology_path
    )

    raw_classes = ontology.get(
        "classes",
        {}
    )

    if not raw_classes:
        raise Gate1Failure(
            "Ontology contains no classes."
        )

    classes = {
        int(class_id): info
        for class_id, info
        in raw_classes.items()
    }

    expected_num = int(
        config["ontology"][
            "expected_num_classes"
        ]
    )

    expected_ids = set(
        range(expected_num)
    )

    observed_ids = set(
        classes
    )

    if observed_ids != expected_ids:
        raise Gate1Failure(
            "Ontology IDs are not exactly "
            f"0-{expected_num - 1}.\n"
            f"Observed: {sorted(observed_ids)}"
        )

    ignore_index = int(
        config["ontology"][
            "ignore_index"
        ]
    )

    if ignore_index not in classes:
        raise Gate1Failure(
            "Configured ignore index does "
            "not exist in ontology."
        )

    if (
        classes[ignore_index]["name"]
        != "ignore"
    ):
        raise Gate1Failure(
            "Ignore class must be named "
            "'ignore'."
        )

    return classes


# ============================================================
# Mapping validation
# ============================================================

def load_mapping(
    config: Dict,
    dataset_name: str,
):
    mapping_path = resolve_repo_path(
        config["mappings"][
            dataset_name
        ]
    )

    mapping_config = load_yaml(
        mapping_path
    )

    mapping = mapping_config.get(
        "mapping",
        {}
    )

    if not mapping:
        raise Gate1Failure(
            f"Empty mapping for "
            f"{dataset_name}"
        )

    return {
        int(native): int(unified)
        for native, unified
        in mapping.items()
    }


def validate_mapping_targets(
    config: Dict,
    classes: Dict,
):
    ontology_ids = set(
        classes
    )

    result = {}

    for dataset_name in (
        "semantic_kitti",
        "rellis_3d",
        "semantic_stf",
        "nuscenes_mini",
    ):
        mapping = load_mapping(
            config,
            dataset_name,
        )

        targets = set(
            mapping.values()
        )

        invalid = (
            targets
            - ontology_ids
        )

        if invalid:
            raise Gate1Failure(
                f"{dataset_name} mapping "
                "contains invalid ontology IDs: "
                f"{sorted(invalid)}"
            )

        result[dataset_name] = {
            "native_ids": len(mapping),
            "unified_ids_used": sorted(
                targets
            ),
        }

    return result


# ============================================================
# Dataset creation
# ============================================================

def build_datasets(
    config: Dict,
    roots: Dict,
):
    return {
        "semantic_kitti":
            SemanticKITTIDataset(
                root=roots[
                    "semantic_kitti"
                ],
                mapping_file=resolve_repo_path(
                    config["mappings"][
                        "semantic_kitti"
                    ]
                ),
                strict_labels=True,
            ),

        "rellis_3d":
            RELLISDataset(
                root=roots[
                    "rellis_3d"
                ],
                mapping_file=resolve_repo_path(
                    config["mappings"][
                        "rellis_3d"
                    ]
                ),
                strict_labels=True,
            ),

        "semantic_stf":
            SemanticSTFDataset(
                root=roots[
                    "semantic_stf"
                ],
                mapping_file=resolve_repo_path(
                    config["mappings"][
                        "semantic_stf"
                    ]
                ),
                strict_labels=True,
            ),

        "nuscenes_mini":
            NuScenesMiniDataset(
                root=roots[
                    "nuscenes_mini"
                ],
                mapping_file=resolve_repo_path(
                    config["mappings"][
                        "nuscenes_mini"
                    ]
                ),
                strict_labels=True,
            ),
    }


# ============================================================
# Sample validation
# ============================================================

def validate_sample(
    sample: Dict,
    expected_dataset_name: str,
    ontology_ids: set,
    mapping: Dict[int, int],
):
    missing = (
        REQUIRED_SAMPLE_KEYS
        - set(sample)
    )

    if missing:
        raise Gate1Failure(
            "Sample missing required keys: "
            f"{sorted(missing)}"
        )

    if (
        sample["dataset"]
        != expected_dataset_name
    ):
        raise Gate1Failure(
            "Incorrect internal dataset name.\n"
            f"Expected: {expected_dataset_name}\n"
            f"Observed: {sample['dataset']}"
        )

    xyz = np.asarray(
        sample["xyz"]
    )

    intensity = np.asarray(
        sample["intensity"]
    )

    native = np.asarray(
        sample["native_label"]
    )

    semantic = np.asarray(
        sample["semantic_label"]
    )

    if (
        xyz.ndim != 2
        or xyz.shape[1] != 3
    ):
        raise Gate1Failure(
            f"Invalid xyz shape: {xyz.shape}"
        )

    n_points = len(xyz)

    if n_points == 0:
        raise Gate1Failure(
            "Frame contains zero points."
        )

    if intensity.shape != (
        n_points,
    ):
        raise Gate1Failure(
            "Intensity length does not "
            "match point count."
        )

    if native.shape != (
        n_points,
    ):
        raise Gate1Failure(
            "Native-label length does not "
            "match point count."
        )

    if semantic.shape != (
        n_points,
    ):
        raise Gate1Failure(
            "Semantic-label length does not "
            "match point count."
        )

    if not np.isfinite(
        xyz
    ).all():
        raise Gate1Failure(
            "XYZ contains NaN or Inf."
        )

    if not np.isfinite(
        intensity
    ).all():
        raise Gate1Failure(
            "Intensity contains NaN or Inf."
        )

    native_ids = set(
        map(
            int,
            np.unique(native),
        )
    )

    unknown_native = (
        native_ids
        - set(mapping)
    )

    if unknown_native:
        raise Gate1Failure(
            "Native IDs are missing from "
            "the dataset mapping: "
            f"{sorted(unknown_native)}"
        )

    semantic_ids = set(
        map(
            int,
            np.unique(semantic),
        )
    )

    invalid_semantic = (
        semantic_ids
        - ontology_ids
    )

    if invalid_semantic:
        raise Gate1Failure(
            "Unified labels outside ontology: "
            f"{sorted(invalid_semantic)}"
        )

    return {
        "points": n_points,
        "native_ids": sorted(
            native_ids
        ),
        "semantic_ids": sorted(
            semantic_ids
        ),
        "semantic_counts": Counter(
            map(
                int,
                semantic,
            )
        ),
    }


# ============================================================
# Dataset validation
# ============================================================

def validate_dataset(
    dataset_name: str,
    dataset,
    config: Dict,
    classes: Dict,
):
    expected_samples = int(
        config["datasets"][
            dataset_name
        ][
            "expected_samples"
        ]
    )

    internal_name = (
        config["datasets"][
            dataset_name
        ][
            "internal_name"
        ]
    )

    actual_samples = len(
        dataset
    )

    if (
        actual_samples
        != expected_samples
    ):
        raise Gate1Failure(
            f"{dataset_name}: expected "
            f"{expected_samples:,} samples, "
            f"found {actual_samples:,}"
        )

    count = int(
        config["validation"][
            "frames_per_dataset"
        ]
    )

    indices = distributed_indices(
        actual_samples,
        count,
        int(config["seed"]),
    )

    mapping = load_mapping(
        config,
        dataset_name,
    )

    ontology_ids = set(
        classes
    )

    total_points = 0
    combined_counts = Counter()
    observed_native_ids = set()
    observed_semantic_ids = set()

    first_sample = None

    frames = []

    for index in indices:
        sample = dataset[
            index
        ]

        validation = (
            validate_sample(
                sample=sample,
                expected_dataset_name=(
                    internal_name
                ),
                ontology_ids=(
                    ontology_ids
                ),
                mapping=mapping,
            )
        )

        if first_sample is None:
            first_sample = sample

        total_points += (
            validation["points"]
        )

        observed_native_ids.update(
            validation[
                "native_ids"
            ]
        )

        observed_semantic_ids.update(
            validation[
                "semantic_ids"
            ]
        )

        combined_counts.update(
            validation[
                "semantic_counts"
            ]
        )

        frames.append(
            {
                "index": index,
                "frame_id": str(
                    sample[
                        "frame_id"
                    ]
                ),
                "points": int(
                    validation[
                        "points"
                    ]
                ),
            }
        )

    return {
        "expected_samples":
            expected_samples,

        "actual_samples":
            actual_samples,

        "validated_frames":
            len(indices),

        "validated_points":
            total_points,

        "native_ids_observed":
            sorted(
                observed_native_ids
            ),

        "unified_ids_observed":
            sorted(
                observed_semantic_ids
            ),

        "class_counts":
            {
                str(class_id):
                    int(count)
                for class_id, count
                in sorted(
                    combined_counts.items()
                )
            },

        "frames":
            frames,

        "_visual_sample":
            first_sample,
    }


# ============================================================
# Split checks
# ============================================================

def validate_semantic_kitti_split(
    config: Dict,
):
    split_config = (
        config["splits"][
            "semantic_kitti"
        ]
    )

    train = set(
        str(sequence).zfill(2)
        for sequence
        in split_config[
            "train_sequences"
        ]
    )

    val = set(
        str(sequence).zfill(2)
        for sequence
        in split_config[
            "val_sequences"
        ]
    )

    overlap = (
        train
        & val
    )

    if overlap:
        raise Gate1Failure(
            "SemanticKITTI train/val "
            "sequence overlap: "
            f"{sorted(overlap)}"
        )

    labelled = set(
        f"{i:02d}"
        for i in range(11)
    )

    covered = (
        train
        | val
    )

    if covered != labelled:
        raise Gate1Failure(
            "SemanticKITTI train/val split "
            "does not cover exactly labelled "
            "sequences 00-10.\n"
            f"Covered: {sorted(covered)}"
        )

    return {
        "train_sequences":
            sorted(train),

        "val_sequences":
            sorted(val),

        "overlap":
            [],

        "note":
            (
                "Official SemanticKITTI "
                "sequences 11-21 are unlabeled "
                "and are not used for supervised "
                "semantic validation."
            ),
    }


def validate_semantic_stf_split(
    config: Dict,
    roots: Dict,
):
    expected = (
        config["splits"][
            "semantic_stf"
        ][
            "expected"
        ]
    )

    mapping_file = (
        resolve_repo_path(
            config["mappings"][
                "semantic_stf"
            ]
        )
    )

    split_ids = {}
    split_counts = {}

    for split in (
        "train",
        "val",
        "test",
    ):
        dataset = (
            SemanticSTFDataset(
                root=roots[
                    "semantic_stf"
                ],
                mapping_file=(
                    mapping_file
                ),
                splits=[
                    split
                ],
                strict_labels=True,
            )
        )

        actual = len(
            dataset
        )

        expected_count = int(
            expected[
                split
            ]
        )

        if (
            actual
            != expected_count
        ):
            raise Gate1Failure(
                f"SemanticSTF {split}: "
                f"expected {expected_count}, "
                f"found {actual}"
            )

        identifiers = {
            str(
                item[
                    "frame_id"
                ]
            )
            for item
            in dataset.samples
        }

        if (
            len(identifiers)
            != actual
        ):
            raise Gate1Failure(
                "Duplicate SemanticSTF "
                f"frame IDs in {split}."
            )

        split_ids[
            split
        ] = identifiers

        split_counts[
            split
        ] = actual

    train_val = (
        split_ids["train"]
        & split_ids["val"]
    )

    train_test = (
        split_ids["train"]
        & split_ids["test"]
    )

    val_test = (
        split_ids["val"]
        & split_ids["test"]
    )

    if (
        train_val
        or train_test
        or val_test
    ):
        raise Gate1Failure(
            "SemanticSTF split leakage detected."
        )

    return {
        "counts":
            split_counts,

        "train_val_overlap":
            0,

        "train_test_overlap":
            0,

        "val_test_overlap":
            0,
    }


# ============================================================
# Benchmark
# ============================================================

def benchmark_dataset(
    dataset_name: str,
    dataset,
    config: Dict,
):
    count = int(
        config["benchmark"][
            "frames_per_dataset"
        ]
    )

    indices = distributed_indices(
        len(dataset),
        count,
        int(config["seed"]) + 999,
    )

    total_points = 0

    start = time.perf_counter()

    for index in indices:
        sample = dataset[
            index
        ]

        total_points += len(
            sample["xyz"]
        )

    elapsed = (
        time.perf_counter()
        - start
    )

    frames = len(indices)

    return {
        "frames":
            frames,

        "points":
            total_points,

        "seconds":
            elapsed,

        "milliseconds_per_frame":
            (
                1000.0
                * elapsed
                / frames
            )
            if frames
            else 0.0,

        "points_per_second":
            (
                total_points
                / elapsed
            )
            if elapsed > 0
            else 0.0,
    }


# ============================================================
# Visualization
# ============================================================

def build_color_table(
    classes: Dict,
):
    """
    Deterministic shared color table.
    Same ontology ID = same color in every dataset.
    """

    num_classes = len(
        classes
    )

    tab20 = plt.get_cmap(
        "tab20"
    )

    tab10 = plt.get_cmap(
        "tab10"
    )

    colors = {}

    # Ignore = neutral gray.
    colors[0] = (
        0.45,
        0.45,
        0.45,
        1.0,
    )

    for class_id in range(
        1,
        num_classes,
    ):
        if class_id <= 20:
            colors[
                class_id
            ] = tab20(
                (class_id - 1)
                % 20
            )
        else:
            colors[
                class_id
            ] = tab10(
                (class_id - 21)
                % 10
            )

    return colors


def save_visualization(
    dataset_name: str,
    sample: Dict,
    classes: Dict,
    config: Dict,
    output_dir: Path,
):
    xyz = np.asarray(
        sample["xyz"]
    )

    labels = np.asarray(
        sample[
            "semantic_label"
        ]
    )

    max_points = int(
        config[
            "visualization"
        ][
            "max_points_per_plot"
        ]
    )

    if (
        len(xyz)
        > max_points
    ):
        rng = np.random.default_rng(
            int(config["seed"])
        )

        indices = rng.choice(
            len(xyz),
            size=max_points,
            replace=False,
        )

        xyz = xyz[
            indices
        ]

        labels = labels[
            indices
        ]

    colors = (
        build_color_table(
            classes
        )
    )

    point_colors = np.array(
        [
            colors[
                int(label)
            ]
            for label
            in labels
        ]
    )

    figure = plt.figure(
        figsize=(12, 10)
    )

    axis = figure.add_subplot(
        111
    )

    axis.scatter(
        xyz[:, 0],
        xyz[:, 1],
        c=point_colors,
        s=float(
            config[
                "visualization"
            ][
                "point_size"
            ]
        ),
        linewidths=0,
    )

    axis.set_aspect(
        "equal",
        adjustable="box",
    )

    axis.set_xlabel(
        "X (m)"
    )

    axis.set_ylabel(
        "Y (m)"
    )

    axis.set_title(
        f"{dataset_name} — "
        "Unified Ontology"
    )

    # Build only legend classes actually
    # visible in the selected frame.
    visible_ids = sorted(
        set(
            map(
                int,
                np.unique(labels),
            )
        )
    )

    handles = []

    for class_id in visible_ids:
        handle = plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=6,
            markerfacecolor=(
                colors[
                    class_id
                ]
            ),
            markeredgecolor=(
                colors[
                    class_id
                ]
            ),
            label=(
                f"{class_id}: "
                f"{classes[class_id]['name']}"
            ),
        )

        handles.append(
            handle
        )

    axis.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(
            1.01,
            1.0,
        ),
        fontsize=8,
    )

    figure.tight_layout()

    visualization_dir = (
        output_dir
        / "visualizations"
    )

    visualization_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        visualization_dir
        / f"{dataset_name}.png"
    )

    figure.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return path


# ============================================================
# Reporting
# ============================================================

def make_json_safe(
    value,
):
    if isinstance(
        value,
        Path,
    ):
        return str(value)

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key):
                make_json_safe(
                    item
                )
            for key, item
            in value.items()
            if not str(key).startswith(
                "_"
            )
        }

    if isinstance(
        value,
        list,
    ):
        return [
            make_json_safe(
                item
            )
            for item in value
        ]

    if isinstance(
        value,
        tuple,
    ):
        return [
            make_json_safe(
                item
            )
            for item in value
        ]

    if isinstance(
        value,
        set,
    ):
        return sorted(
            value
        )

    if isinstance(
        value,
        (
            np.integer,
            np.floating,
        ),
    ):
        return value.item()

    return value


def save_report(
    report: Dict,
    output_dir: Path,
):
    json_path = (
        output_dir
        / "gate1_report.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            make_json_safe(
                report
            ),
            file,
            indent=2,
        )

    text_path = (
        output_dir
        / "gate1_report.txt"
    )

    with open(
        text_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "SIH26053 GATE 1 PREFLIGHT\n"
        )

        file.write(
            "=" * 70
            + "\n"
        )

        file.write(
            f"Result: "
            f"{report['result']}\n"
        )

        file.write(
            f"Mode: "
            f"{report['mode']}\n"
        )

        file.write(
            f"Runtime: "
            f"{report['runtime_seconds']:.2f} sec\n"
        )

        file.write(
            "\nDataset validation:\n"
        )

        for name, result in (
            report.get(
                "datasets",
                {}
            ).items()
        ):
            file.write(
                f"\n{name}\n"
            )

            file.write(
                f"  samples: "
                f"{result['actual_samples']}\n"
            )

            file.write(
                f"  validated frames: "
                f"{result['validated_frames']}\n"
            )

            file.write(
                f"  validated points: "
                f"{result['validated_points']}\n"
            )

            file.write(
                f"  native IDs: "
                f"{result['native_ids_observed']}\n"
            )

            file.write(
                f"  unified IDs: "
                f"{result['unified_ids_observed']}\n"
            )

        if "benchmark" in report:
            file.write(
                "\nBenchmark:\n"
            )

            for name, result in (
                report[
                    "benchmark"
                ].items()
            ):
                file.write(
                    f"  {name}: "
                    f"{result['milliseconds_per_frame']:.1f} ms/frame, "
                    f"{result['points_per_second']:,.0f} points/s\n"
                )

    return (
        json_path,
        text_path,
    )


# ============================================================
# Console display
# ============================================================

def print_dataset_result(
    name: str,
    result: Dict,
):
    print(
        f"\n{name}"
    )

    print(
        f"  Samples              : "
        f"{result['actual_samples']:,}"
    )

    print(
        f"  Frames validated     : "
        f"{result['validated_frames']}"
    )

    print(
        f"  Points validated     : "
        f"{result['validated_points']:,}"
    )

    print(
        f"  Native IDs observed  : "
        f"{result['native_ids_observed']}"
    )

    print(
        f"  Unified IDs observed : "
        f"{result['unified_ids_observed']}"
    )

    print(
        "  Result               : PASS"
    )


# ============================================================
# Main Gate 1 execution
# ============================================================

def run_gate1(
    config_path: Path,
    mode: str,
):
    start_time = (
        time.perf_counter()
    )

    config = load_yaml(
        config_path
    )

    output_dir = (
        get_output_directory(
            config
        )
    )

    report = {
        "mode": mode,
        "result": "FAIL",
        "output_directory":
            str(output_dir),
    }

    print(
        "=" * 80
    )

    print(
        "SIH26053 — GATE 1 PREFLIGHT"
    )

    print(
        "=" * 80
    )

    print(
        f"Mode: {mode}"
    )

    print()

    # --------------------------------------------------------
    # Dataset roots
    # --------------------------------------------------------

    print(
        "[1/7] Dataset roots"
    )

    roots = (
        validate_dataset_roots()
    )

    report["roots"] = {
        key: str(value)
        for key, value
        in roots.items()
    }

    for name, path in (
        roots.items()
    ):
        print(
            f"  PASS  "
            f"{name:18s} "
            f"{path}"
        )

    # --------------------------------------------------------
    # Ontology
    # --------------------------------------------------------

    print()
    print(
        "[2/7] Ontology"
    )

    classes = (
        load_and_validate_ontology(
            config
        )
    )

    print(
        f"  PASS  "
        f"{len(classes)} classes"
    )

    print(
        f"  PASS  IDs "
        f"0-{len(classes) - 1}"
    )

    print(
        f"  PASS  ignore_index="
        f"{config['ontology']['ignore_index']}"
    )

    # --------------------------------------------------------
    # Mapping files
    # --------------------------------------------------------

    print()
    print(
        "[3/7] Mapping validation"
    )

    mapping_report = (
        validate_mapping_targets(
            config,
            classes,
        )
    )

    report[
        "mappings"
    ] = mapping_report

    for name, info in (
        mapping_report.items()
    ):
        print(
            f"  PASS  "
            f"{name:18s} "
            f"{info['native_ids']} native IDs"
        )

    # --------------------------------------------------------
    # Initialize real loaders
    # --------------------------------------------------------

    print()
    print(
        "[4/7] Real loader initialization"
    )

    init_start = (
        time.perf_counter()
    )

    datasets = (
        build_datasets(
            config,
            roots,
        )
    )

    initialization_time = (
        time.perf_counter()
        - init_start
    )

    for name, dataset in (
        datasets.items()
    ):
        print(
            f"  PASS  "
            f"{name:18s} "
            f"{len(dataset):,} samples"
        )

    print(
        f"  Loader initialization "
        f"time: {initialization_time:.2f}s"
    )

    report[
        "loader_initialization_seconds"
    ] = initialization_time

    # --------------------------------------------------------
    # Validate real samples
    # --------------------------------------------------------

    print()
    print(
        "[5/7] Real-frame validation"
    )

    dataset_report = {}

    visual_samples = {}

    for name, dataset in (
        datasets.items()
    ):
        result = validate_dataset(
            dataset_name=name,
            dataset=dataset,
            config=config,
            classes=classes,
        )

        visual_samples[
            name
        ] = result.pop(
            "_visual_sample"
        )

        dataset_report[
            name
        ] = result

        print_dataset_result(
            name,
            result,
        )

    report[
        "datasets"
    ] = dataset_report

    # --------------------------------------------------------
    # Split checks
    # --------------------------------------------------------

    print()
    print(
        "[6/7] Split integrity"
    )

    split_report = {}

    split_report[
        "semantic_kitti"
    ] = (
        validate_semantic_kitti_split(
            config
        )
    )

    print(
        "  PASS  SemanticKITTI "
        "train/validation sequences disjoint"
    )

    split_report[
        "semantic_stf"
    ] = (
        validate_semantic_stf_split(
            config,
            roots,
        )
    )

    print(
        "  PASS  SemanticSTF "
        "train/val/test counts"
    )

    print(
        "  PASS  SemanticSTF "
        "train/val/test overlap = 0"
    )

    print(
        "  INFO  RELLIS split policy "
        "will be frozen before Gate 2"
    )

    print(
        "  INFO  nuScenes mini split "
        "policy will be frozen before Gate 2"
    )

    report[
        "splits"
    ] = split_report

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    if (
        mode
        in {
            "validate",
            "full",
        }
        and bool(
            config[
                "visualization"
            ][
                "enabled"
            ]
        )
    ):
        print()
        print(
            "[7/7] Unified ontology "
            "visualizations"
        )

        visualization_report = {}

        for name, sample in (
            visual_samples.items()
        ):
            path = (
                save_visualization(
                    dataset_name=name,
                    sample=sample,
                    classes=classes,
                    config=config,
                    output_dir=(
                        output_dir
                    ),
                )
            )

            visualization_report[
                name
            ] = str(path)

            print(
                f"  PASS  {name:18s} "
                f"{path.name}"
            )

        report[
            "visualizations"
        ] = visualization_report

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    if mode in {
        "benchmark",
        "full",
    }:
        print()
        print(
            "BENCHMARK MODE"
        )

        print(
            "-" * 80
        )

        benchmark_report = {}

        for name, dataset in (
            datasets.items()
        ):
            result = (
                benchmark_dataset(
                    dataset_name=name,
                    dataset=dataset,
                    config=config,
                )
            )

            benchmark_report[
                name
            ] = result

            print(
                f"{name:18s} | "
                f"{result['frames']} frames | "
                f"{result['points']:,} points | "
                f"{result['milliseconds_per_frame']:.1f} ms/frame | "
                f"{result['points_per_second']:,.0f} points/s"
            )

        report[
            "benchmark"
        ] = benchmark_report

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    runtime = (
        time.perf_counter()
        - start_time
    )

    report[
        "runtime_seconds"
    ] = runtime

    report[
        "result"
    ] = "PASS"

    json_path, text_path = (
        save_report(
            report,
            output_dir,
        )
    )

    print()
    print(
        "=" * 80
    )

    print(
        "GATE 1 AUTOMATED RESULT: PASS"
    )

    print(
        "=" * 80
    )

    print(
        f"Runtime       : "
        f"{runtime:.1f} sec"
    )

    print(
        f"JSON report   : "
        f"{json_path}"
    )

    print(
        f"Text report   : "
        f"{text_path}"
    )

    print()

    print(
        "NEXT REQUIRED ACTION:"
    )

    print(
        "Open the four generated semantic "
        "visualizations and perform the "
        "manual semantic sanity check."
    )

    print()

    print(
        "Gate 1 is CLOSED only after:"
    )

    print(
        "  1. automated result = PASS"
    )

    print(
        "  2. all four visualizations look "
        "semantically sensible"
    )

    return report


# ============================================================
# Error handling
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "SIH26053 Gate 1 "
            "dataset preflight"
        )
    )

    parser.add_argument(
        "--config",
        default=(
            "configs/gate1.yaml"
        ),
        help=(
            "Path to Gate 1 YAML config."
        ),
    )

    parser.add_argument(
        "--mode",
        choices=(
            "validate",
            "benchmark",
            "full",
        ),
        default="full",
        help=(
            "validate = correctness + plots, "
            "benchmark = correctness + I/O benchmark, "
            "full = everything"
        ),
    )

    args = parser.parse_args()

    config_path = (
        resolve_repo_path(
            args.config
        )
    )

    try:
        run_gate1(
            config_path=config_path,
            mode=args.mode,
        )

    except OSError as error:
        print()
        print(
            "=" * 80
        )

        if (
            getattr(
                error,
                "errno",
                None,
            )
            == errno.ENOTCONN
            or "Transport endpoint is not connected"
            in str(error)
        ):
            print(
                "GOOGLE DRIVE CONNECTION FAILURE"
            )

            print(
                "=" * 80
            )

            print(
                "The Google Drive FUSE mount "
                "disconnected."
            )

            print(
                "This is NOT evidence that the "
                "dataset is corrupted."
            )

            print(
                "Remount Google Drive and rerun "
                "Gate 1."
            )

        else:
            print(
                "GATE 1 I/O FAILURE"
            )

            print(
                "=" * 80
            )

            print(
                f"{type(error).__name__}: "
                f"{error}"
            )

        sys.exit(2)

    except Exception as error:
        print()
        print(
            "=" * 80
        )

        print(
            "GATE 1 RESULT: FAIL"
        )

        print(
            "=" * 80
        )

        print(
            f"{type(error).__name__}: "
            f"{error}"
        )

        print()

        traceback.print_exc()

        sys.exit(1)


if __name__ == "__main__":
    main()