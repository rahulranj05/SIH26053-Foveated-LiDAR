from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from datasets.f0_tar_dataset import F0TarDataset


DATASETS = (
    "semantic_kitti",
    "rellis_3d",
    "semantic_stf",
    "nuscenes_mini",
)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Audit TRAIN-only LiDAR intensity distributions "
            "from the existing F0 TAR cache."
        )
    )

    p.add_argument(
        "--cache-dir",
        required=True,
    )

    p.add_argument(
        "--repo-root",
        default=".",
    )

    p.add_argument(
        "--output",
        required=True,
    )

    p.add_argument(
        "--sample-per-frame",
        type=int,
        default=4096,
        help=(
            "Maximum intensity values retained per frame "
            "for percentile estimation."
        ),
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    p.add_argument(
        "--progress-every",
        type=int,
        default=500,
    )

    return p.parse_args()


def main():
    args = parse_args()

    cache_dir = Path(args.cache_dir).resolve()
    repo_root = Path(args.repo_root).resolve()
    output_path = Path(args.output).resolve()

    rng = np.random.default_rng(args.seed)

    dataset = F0TarDataset(
        cache_dir=cache_dir,
        repo_root=repo_root,
        split="train",
        strict=True,
    )

    print("=" * 88)
    print("SIH26053 — TRAIN INTENSITY AUDIT")
    print("=" * 88)

    print(f"TRAIN frames: {len(dataset):,}")

    stats = {}

    for name in DATASETS:
        stats[name] = {
            "frames": 0,
            "points": 0,
            "finite": 0,
            "nonfinite": 0,
            "zeros": 0,
            "min": np.inf,
            "max": -np.inf,
            "sum": 0.0,
            "sum_sq": 0.0,
            "samples": [],
        }

    start = time.perf_counter()

    for idx in range(len(dataset)):
        sample = dataset[idx]

        name = str(sample["dataset"])

        x = np.asarray(
            sample["intensity"],
            dtype=np.float64,
        )

        s = stats[name]

        s["frames"] += 1
        s["points"] += int(x.size)

        finite_mask = np.isfinite(x)

        finite = x[finite_mask]

        s["finite"] += int(finite.size)
        s["nonfinite"] += int(
            x.size - finite.size
        )

        if finite.size:
            s["zeros"] += int(
                np.count_nonzero(finite == 0.0)
            )

            s["min"] = min(
                s["min"],
                float(finite.min()),
            )

            s["max"] = max(
                s["max"],
                float(finite.max()),
            )

            s["sum"] += float(
                finite.sum(dtype=np.float64)
            )

            s["sum_sq"] += float(
                np.square(
                    finite,
                    dtype=np.float64,
                ).sum(dtype=np.float64)
            )

            keep = min(
                args.sample_per_frame,
                finite.size,
            )

            if keep == finite.size:
                sampled = finite
            else:
                inds = rng.choice(
                    finite.size,
                    size=keep,
                    replace=False,
                )
                sampled = finite[inds]

            s["samples"].append(
                sampled.astype(
                    np.float32,
                    copy=False,
                )
            )

        done = idx + 1

        if (
            done % args.progress_every == 0
            or done == len(dataset)
        ):
            elapsed = (
                time.perf_counter()
                - start
            )

            print(
                f"{done:6,d}/{len(dataset):6,d} "
                f"frames | "
                f"{elapsed:8.1f}s"
            )

    dataset.close()

    percentiles = [
        0.1,
        1.0,
        5.0,
        50.0,
        95.0,
        99.0,
        99.9,
    ]

    payload = {
        "split": "train",
        "seed": args.seed,
        "sample_per_frame": (
            args.sample_per_frame
        ),
        "datasets": {},
    }

    print()
    print("=" * 88)
    print("INTENSITY SUMMARY")
    print("=" * 88)

    for name in DATASETS:
        s = stats[name]

        if s["finite"] == 0:
            raise RuntimeError(
                f"{name}: no finite intensity values."
            )

        sample_values = np.concatenate(
            s["samples"]
        )

        mean = (
            s["sum"]
            / s["finite"]
        )

        variance = max(
            (
                s["sum_sq"]
                / s["finite"]
            )
            - mean * mean,
            0.0,
        )

        std = float(
            np.sqrt(variance)
        )

        q = np.percentile(
            sample_values,
            percentiles,
        )

        zero_fraction = (
            s["zeros"]
            / s["finite"]
        )

        result = {
            "frames": int(s["frames"]),
            "points": int(s["points"]),
            "finite": int(s["finite"]),
            "nonfinite": int(
                s["nonfinite"]
            ),
            "min": float(s["min"]),
            "p0_1": float(q[0]),
            "p1": float(q[1]),
            "p5": float(q[2]),
            "median": float(q[3]),
            "mean": float(mean),
            "std": float(std),
            "p95": float(q[4]),
            "p99": float(q[5]),
            "p99_9": float(q[6]),
            "max": float(s["max"]),
            "zero_fraction": float(
                zero_fraction
            ),
            "percentile_sample_count": int(
                sample_values.size
            ),
        }

        payload["datasets"][name] = result

        print()
        print(name)
        print("-" * 88)

        for key, value in result.items():
            if isinstance(value, float):
                print(
                    f"  {key:26s}: "
                    f"{value:.8f}"
                )
            else:
                print(
                    f"  {key:26s}: "
                    f"{value:,}"
                )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    with temp.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
        )

    temp.replace(output_path)

    runtime = (
        time.perf_counter()
        - start
    )

    print()
    print("=" * 88)
    print("TRAIN INTENSITY AUDIT: PASS")
    print("=" * 88)
    print(f"Runtime: {runtime:.1f} sec")
    print(f"Saved  : {output_path}")


if __name__ == "__main__":
    main()