from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from datasets.f0_tar_dataset import F0TarDataset


NUM_CLASSES = 23
IGNORE_INDEX = 0


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Recompute exact F0 TRAIN class distribution "
            "from the existing lossless TAR cache using "
            "the CURRENT native-to-unified mappings."
        )
    )

    p.add_argument(
        "--cache-dir",
        required=True,
        help="TRAIN F0 TAR cache directory.",
    )

    p.add_argument(
        "--repo-root",
        default=".",
        help="Repository root.",
    )

    p.add_argument(
        "--ontology",
        default="configs/ontology.yaml",
        help="Unified ontology YAML.",
    )

    p.add_argument(
        "--output",
        required=True,
        help="Output JSON path.",
    )

    p.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help="Print progress every N frames.",
    )

    return p.parse_args()


def load_class_names(path: Path):
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    classes = cfg["classes"]

    names = {}

    # Support either:
    #
    # classes:
    #   0:
    #     name: ignore
    #
    # OR
    #
    # classes:
    #   - id: 0
    #     name: ignore

    if isinstance(classes, dict):
        for class_id, item in classes.items():
            cid = int(class_id)

            if isinstance(item, dict):
                names[cid] = str(item["name"])
            else:
                names[cid] = str(item)

    elif isinstance(classes, list):
        for item in classes:
            cid = int(item["id"])
            names[cid] = str(item["name"])

    else:
        raise RuntimeError(
            "Unsupported ontology classes format."
        )

    expected = set(range(NUM_CLASSES))
    observed = set(names)

    if observed != expected:
        raise RuntimeError(
            "Ontology class IDs do not match 0..22.\n"
            f"Expected: {sorted(expected)}\n"
            f"Observed: {sorted(observed)}"
        )

    return names


def empty_counts():
    return np.zeros(
        NUM_CLASSES,
        dtype=np.int64,
    )


def main():
    args = parse_args()

    repo_root = Path(args.repo_root).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    ontology_path = repo_root / args.ontology
    output_path = Path(args.output).resolve()

    if not cache_dir.is_dir():
        raise FileNotFoundError(
            f"Cache directory not found: {cache_dir}"
        )

    if not ontology_path.is_file():
        raise FileNotFoundError(
            f"Ontology not found: {ontology_path}"
        )

    class_names = load_class_names(
        ontology_path
    )

    print("=" * 88)
    print(
        "SIH26053 — EXACT F0 TRAIN DISTRIBUTION RECOUNT"
    )
    print("=" * 88)

    print()
    print(f"Cache     : {cache_dir}")
    print(f"Repo root : {repo_root}")
    print(f"Ontology  : {ontology_path}")
    print(f"Output    : {output_path}")

    dataset = F0TarDataset(
        cache_dir=cache_dir,
        repo_root=repo_root,
        split="train",
        strict=True,
    )

    print()
    print(
        f"TRAIN frames indexed: {len(dataset):,}"
    )

    expected_dataset_counts = {
        name: len(indices)
        for name, indices
        in dataset.dataset_to_indices.items()
    }

    print()
    print("Frames by dataset:")

    for name in sorted(expected_dataset_counts):
        print(
            f"  {name:20s}: "
            f"{expected_dataset_counts[name]:,}"
        )

    combined_counts = empty_counts()

    dataset_counts = defaultdict(
        empty_counts
    )

    dataset_frames_processed = defaultdict(int)

    total_points = 0

    start = time.perf_counter()

    for index in range(len(dataset)):
        sample = dataset[index]

        labels = np.asarray(
            sample["semantic_label"],
            dtype=np.int64,
        )

        if labels.ndim != 1:
            raise RuntimeError(
                f"Frame {index}: semantic labels "
                f"must be 1-D, got {labels.shape}"
            )

        if labels.size == 0:
            raise RuntimeError(
                f"Frame {index}: empty semantic labels"
            )

        minimum = int(labels.min())
        maximum = int(labels.max())

        if minimum < 0 or maximum >= NUM_CLASSES:
            raise RuntimeError(
                f"Frame {index}: unified label range "
                f"{minimum}..{maximum} outside 0.."
                f"{NUM_CLASSES - 1}"
            )

        bincount = np.bincount(
            labels,
            minlength=NUM_CLASSES,
        )[:NUM_CLASSES]

        dataset_name = str(
            sample["dataset"]
        )

        combined_counts += bincount
        dataset_counts[dataset_name] += bincount
        dataset_frames_processed[dataset_name] += 1

        total_points += int(labels.size)

        done = index + 1

        if (
            done % args.progress_every == 0
            or done == len(dataset)
        ):
            elapsed = (
                time.perf_counter()
                - start
            )

            rate = (
                done / elapsed
                if elapsed > 0
                else 0.0
            )

            print(
                f"  {done:6,d}/{len(dataset):6,d} "
                f"frames | "
                f"{total_points:14,d} points | "
                f"{elapsed:8.1f}s | "
                f"{rate:6.2f} frames/s"
            )

    dataset.close()

    elapsed = (
        time.perf_counter()
        - start
    )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if sum(dataset_frames_processed.values()) != len(dataset):
        raise RuntimeError(
            "Processed frame total does not match "
            "dataset length."
        )

    for dataset_name, expected in expected_dataset_counts.items():
        actual = int(
            dataset_frames_processed[
                dataset_name
            ]
        )

        if actual != expected:
            raise RuntimeError(
                f"{dataset_name}: processed "
                f"{actual} frames but expected "
                f"{expected}"
            )

    if int(combined_counts.sum()) != total_points:
        raise RuntimeError(
            "Combined class counts do not equal "
            "total processed points."
        )

    # --------------------------------------------------------
    # Build output
    # --------------------------------------------------------

    def make_distribution(counts):
        total = int(counts.sum())

        result = {}

        for class_id in range(NUM_CLASSES):
            points = int(
                counts[class_id]
            )

            percentage = (
                100.0 * points / total
                if total > 0
                else 0.0
            )

            result[str(class_id)] = {
                "name": class_names[class_id],
                "points": points,
                "percentage": percentage,
            }

        return result

    payload = {
        "source": (
            "Existing lossless TRAIN F0 TAR cache, "
            "remapped using current repository mappings."
        ),
        "split": "train",
        "num_classes": NUM_CLASSES,
        "ignore_index": IGNORE_INDEX,
        "frames": int(len(dataset)),
        "points": int(total_points),
        "runtime_seconds": float(elapsed),
        "frames_by_dataset": {
            name: int(
                dataset_frames_processed[name]
            )
            for name
            in sorted(dataset_frames_processed)
        },
        "combined": make_distribution(
            combined_counts
        ),
        "datasets": {
            name: make_distribution(
                dataset_counts[name]
            )
            for name
            in sorted(dataset_counts)
        },
    }

    # --------------------------------------------------------
    # Print exact distribution
    # --------------------------------------------------------

    print()
    print(
        f"{'ID':>3}  "
        f"{'Class':25s} "
        f"{'Points':>15s} "
        f"{'%':>9s}"
    )

    print("-" * 58)

    for class_id in range(NUM_CLASSES):
        points = int(
            combined_counts[class_id]
        )

        percentage = (
            100.0 * points / total_points
            if total_points > 0
            else 0.0
        )

        print(
            f"{class_id:3d}  "
            f"{class_names[class_id]:25s} "
            f"{points:15,d} "
            f"{percentage:8.4f}%"
        )

    # --------------------------------------------------------
    # Important ontology checks
    # --------------------------------------------------------

    if combined_counts[IGNORE_INDEX] == 0:
        print()
        print(
            "WARNING: ignore class has zero points."
        )

    if combined_counts[5] != 0:
        raise RuntimeError(
            "Class 5 rough_ground was expected to "
            "have zero direct semantic supervision, "
            f"but found {int(combined_counts[5]):,} points."
        )

    # --------------------------------------------------------
    # Atomic save
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    with temp_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
        )

    temp_path.replace(
        output_path
    )

    print()
    print("=" * 88)
    print(
        "EXACT F0 TRAIN DISTRIBUTION RECOUNT: PASS"
    )
    print("=" * 88)

    print(
        f"Frames : {len(dataset):,}"
    )

    print(
        f"Points : {total_points:,}"
    )

    print(
        f"Runtime: {elapsed:.1f} sec"
    )

    print(
        f"Saved  : {output_path}"
    )


if __name__ == "__main__":
    main()