from __future__ import annotations

import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from datasets.paths import validate_dataset_roots
from scripts.prepare_f0_data import (
    load_yaml,
    resolve_repo_path,
    prepare_semantic_kitti,
    prepare_semantic_stf,
    prepare_rellis,
    prepare_nuscenes,
)


NUM_CLASSES = 23
IGNORE_INDEX = 0

EXPECTED_TRAIN_FRAMES = {
    "semantic_kitti": 19130,
    "semantic_stf": 1326,
    "rellis_3d": 7800,
    "nuscenes_mini": 282,
}


def count_one_frame(
    dataset,
    index: int,
) -> np.ndarray:
    """
    Read only the semantic label file for one frame.

    Point-cloud geometry is NOT loaded.
    """

    sample = dataset.samples[index]

    if "label_path" not in sample:
        raise RuntimeError(
            f"Sample index {index} has no label_path."
        )

    if not hasattr(
        dataset,
        "_load_native_labels",
    ):
        raise RuntimeError(
            f"{type(dataset).__name__} "
            "does not expose _load_native_labels()."
        )

    if not hasattr(
        dataset,
        "_remap_labels",
    ):
        raise RuntimeError(
            f"{type(dataset).__name__} "
            "does not expose _remap_labels()."
        )

    label_path = Path(
        sample["label_path"]
    )

    native_labels = (
        dataset._load_native_labels(
            label_path
        )
    )

    semantic_labels = (
        dataset._remap_labels(
            native_labels
        )
    )

    semantic_labels = np.asarray(
        semantic_labels,
        dtype=np.int64,
    )

    counts = np.bincount(
        semantic_labels,
        minlength=NUM_CLASSES,
    )

    return counts[:NUM_CLASSES]


def count_dataset_parallel(
    dataset_name: str,
    dataset,
    indices: Iterable[int],
    workers: int = 12,
) -> Counter:
    """
    Count exact class frequencies using parallel label-file reads.
    """

    indices = list(indices)

    total_frames = len(indices)

    print()
    print("=" * 80)
    print(
        f"EXACT LABEL COUNT - {dataset_name}"
    )
    print("=" * 80)

    print(
        f"Frames  : {total_frames:,}"
    )

    print(
        f"Workers : {workers}"
    )

    total_counts = np.zeros(
        NUM_CLASSES,
        dtype=np.int64,
    )

    start_time = time.perf_counter()

    completed = 0

    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:

        futures = {}

        for index in indices:

            future = executor.submit(
                count_one_frame,
                dataset,
                index,
            )

            futures[future] = index

        for future in as_completed(
            futures
        ):

            index = futures[future]

            try:

                frame_counts = (
                    future.result()
                )

            except Exception as exc:

                sample = (
                    dataset.samples[index]
                )

                print()
                print(
                    "ERROR while reading label."
                )
                print(
                    f"Dataset : {dataset_name}"
                )
                print(
                    f"Index   : {index}"
                )
                print(
                    f"Sample  : {sample}"
                )
                print(
                    f"Error   : {exc}"
                )

                raise

            total_counts += frame_counts

            completed += 1

            if (
                completed % 250 == 0
                or completed == total_frames
            ):

                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                if elapsed > 0:
                    rate = (
                        completed
                        / elapsed
                    )
                else:
                    rate = 0.0

                print(
                    f"  "
                    f"{completed:6,d}"
                    f"/{total_frames:6,d}"
                    f" | "
                    f"{elapsed:8.1f}s"
                    f" | "
                    f"{rate:7.2f} frames/s",
                    flush=True,
                )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print(
        f"PASS: {dataset_name}"
    )

    print(
        f"Runtime: {elapsed:.1f}s"
    )

    result = Counter()

    for class_id in range(
        NUM_CLASSES
    ):

        result[class_id] = int(
            total_counts[class_id]
        )

    return result


def save_json(
    path: Path,
    data: dict,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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


def main() -> None:

    config_path = (
        REPO_ROOT
        / "configs"
        / "f0_data.yaml"
    )

    config = load_yaml(
        config_path
    )

    roots = (
        validate_dataset_roots()
    )

    print("=" * 80)
    print(
        "SIH26053 - EXACT F0 TRAIN LABEL DISTRIBUTION"
    )
    print("=" * 80)

    overall_start = (
        time.perf_counter()
    )

    # ========================================================
    # STEP 1
    # ========================================================

    print()
    print(
        "[1/4] Recreating frozen splits..."
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

    # ========================================================
    # STEP 2
    # ========================================================

    print()
    print(
        "[2/4] Verifying frozen TRAIN counts..."
    )

    train_counts = {
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

    for (
        dataset_name,
        expected_count,
    ) in EXPECTED_TRAIN_FRAMES.items():

        actual_count = (
            train_counts[
                dataset_name
            ]
        )

        print(
            f"  "
            f"{dataset_name:18s}: "
            f"{actual_count:,}"
        )

        if (
            actual_count
            != expected_count
        ):

            raise RuntimeError(
                f"{dataset_name} TRAIN "
                f"count changed. "
                f"Expected "
                f"{expected_count:,}, "
                f"got "
                f"{actual_count:,}."
            )

    print(
        "PASS: frozen TRAIN counts unchanged."
    )

    # ========================================================
    # STEP 3
    # ========================================================

    print()
    print(
        "[3/4] Counting exact unified labels..."
    )

    workers = 12

    per_dataset = {}

    # --------------------------------------------------------
    # SemanticKITTI
    # --------------------------------------------------------

    per_dataset[
        "semantic_kitti"
    ] = count_dataset_parallel(
        dataset_name="SemanticKITTI",
        dataset=semantic_kitti[
            "train"
        ],
        indices=range(
            len(
                semantic_kitti[
                    "train"
                ]
            )
        ),
        workers=workers,
    )

    # --------------------------------------------------------
    # SemanticSTF
    # --------------------------------------------------------

    per_dataset[
        "semantic_stf"
    ] = count_dataset_parallel(
        dataset_name="SemanticSTF",
        dataset=semantic_stf[
            "train"
        ],
        indices=range(
            len(
                semantic_stf[
                    "train"
                ]
            )
        ),
        workers=workers,
    )

    # --------------------------------------------------------
    # RELLIS
    # --------------------------------------------------------

    per_dataset[
        "rellis_3d"
    ] = count_dataset_parallel(
        dataset_name="RELLIS-3D",
        dataset=rellis[
            "dataset"
        ],
        indices=rellis[
            "indices"
        ][
            "train"
        ],
        workers=workers,
    )

    # --------------------------------------------------------
    # nuScenes
    # --------------------------------------------------------

    per_dataset[
        "nuscenes_mini"
    ] = count_dataset_parallel(
        dataset_name="nuScenes-mini",
        dataset=nuscenes[
            "dataset"
        ],
        indices=nuscenes[
            "indices"
        ][
            "train"
        ],
        workers=workers,
    )

    # ========================================================
    # Combine datasets
    # ========================================================

    combined = Counter()

    for counts in (
        per_dataset.values()
    ):

        combined.update(
            counts
        )

    total_points = sum(
        combined.values()
    )

    ontology_path = (
        resolve_repo_path(
            config[
                "ontology"
            ][
                "path"
            ]
        )
    )

    ontology = load_yaml(
        ontology_path
    )

    classes = ontology[
        "classes"
    ]

    print()
    print("=" * 80)
    print(
        "EXACT UNIFIED TRAIN DISTRIBUTION"
    )
    print("=" * 80)

    print(
        f"{'ID':>3}  "
        f"{'Class':25s}  "
        f"{'Points':>15s}  "
        f"{'%':>10s}"
    )

    print("-" * 62)

    exact_rows = {}

    for class_id in range(
        NUM_CLASSES
    ):

        class_info = (
            classes[
                class_id
            ]
        )

        class_name = (
            class_info[
                "name"
            ]
        )

        point_count = int(
            combined.get(
                class_id,
                0,
            )
        )

        if total_points > 0:

            percentage = (
                100.0
                * point_count
                / total_points
            )

        else:

            percentage = 0.0

        print(
            f"{class_id:3d}  "
            f"{class_name:25s}  "
            f"{point_count:15,d}  "
            f"{percentage:9.4f}%"
        )

        exact_rows[
            str(class_id)
        ] = {
            "name":
                class_name,

            "points":
                point_count,

            "percentage":
                percentage,
        }

    # ========================================================
    # STEP 4
    # ========================================================

    print()
    print(
        "[4/4] Saving exact distribution..."
    )

    if Path(
        "/content"
    ).exists():

        output_dir = Path(
            "/content/drive/MyDrive/"
            "SIH26053/f0_preparation"
        )

    else:

        output_dir = (
            REPO_ROOT
            / "outputs"
            / "f0_preparation"
        )

    per_dataset_json = {}

    for (
        dataset_name,
        counts,
    ) in per_dataset.items():

        dataset_counts = {}

        for class_id in range(
            NUM_CLASSES
        ):

            dataset_counts[
                str(class_id)
            ] = int(
                counts.get(
                    class_id,
                    0,
                )
            )

        per_dataset_json[
            dataset_name
        ] = dataset_counts

    output = {
        "distribution_type":
            "exact",

        "num_classes":
            NUM_CLASSES,

        "ignore_index":
            IGNORE_INDEX,

        "train_frames":
            train_counts,

        "total_points":
            int(
                total_points
            ),

        "per_dataset":
            per_dataset_json,

        "combined":
            exact_rows,
    }

    output_path = (
        output_dir
        / "f0_class_distribution_exact.json"
    )

    save_json(
        output_path,
        output,
    )

    runtime = (
        time.perf_counter()
        - overall_start
    )

    print()
    print("=" * 80)
    print(
        "EXACT F0 DISTRIBUTION: PASS"
    )
    print("=" * 80)

    print(
        f"Runtime : "
        f"{runtime:.1f}s"
    )

    print(
        f"Output  : "
        f"{output_path}"
    )

    print()
    print(
        "NEXT: compare class-weight strategies."
    )


if __name__ == "__main__":
    main()