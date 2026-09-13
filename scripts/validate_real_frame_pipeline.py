from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from datasets.f0_tar_dataset import F0TarDataset
from models.feature_adapter import FeatureAdapter
from models.voxel_adapter import (
    VoxelAdapter,
    project_voxel_predictions_to_points,
)


DATASET_ORDER = (
    "semantic_kitti",
    "rellis_3d",
    "semantic_stf",
    "nuscenes_mini",
)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Validate real F0 TAR frame -> feature adapter -> "
            "5 cm voxelization -> inverse projection."
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
        "--normalization-config",
        default="configs/intensity_normalization.yaml",
    )

    p.add_argument(
        "--frames-per-dataset",
        type=int,
        default=2,
    )

    p.add_argument(
        "--voxel-size",
        type=float,
        default=0.05,
    )

    return p.parse_args()


def validate_mode(
    ds,
    feature_adapter,
    voxel_adapter,
    indices,
    mode,
):
    summary = defaultdict(
        lambda: {
            "frames": 0,
            "points": 0,
            "voxels": 0,
            "agreement_sum": 0.0,
        }
    )

    print()
    print("=" * 100)
    print(f"FEATURE MODE: {mode}")
    print("=" * 100)

    for dataset_name in DATASET_ORDER:

        print()
        print(dataset_name)
        print("-" * 100)

        for idx in indices[dataset_name]:

            sample = ds[idx]

            xyz = np.asarray(
                sample["xyz"],
                dtype=np.float32,
            )

            intensity = np.asarray(
                sample["intensity"],
                dtype=np.float32,
            )

            labels = np.asarray(
                sample["semantic_label"],
                dtype=np.int64,
            )

            # ------------------------------------------------
            # Raw sample checks
            # ------------------------------------------------

            n_points = xyz.shape[0]

            assert xyz.shape == (n_points, 3)
            assert intensity.shape == (n_points,)
            assert labels.shape == (n_points,)

            assert np.isfinite(xyz).all()
            assert np.isfinite(intensity).all()

            assert labels.min() >= 0
            assert labels.max() <= 22

            # ------------------------------------------------
            # Feature adapter
            # ------------------------------------------------

            features = feature_adapter(
                intensity,
                dataset_name,
            )

            assert features.shape == (
                n_points,
                1,
            )

            assert features.dtype == np.float32
            assert np.isfinite(features).all()

            if mode == "constant":
                assert np.allclose(
                    features,
                    1.0,
                )

            elif mode == "robust_intensity":
                assert features.min() >= -1e-6
                assert features.max() <= 1.0 + 1e-6

            # ------------------------------------------------
            # Voxelization
            # ------------------------------------------------

            vox = voxel_adapter(
                xyz,
                features,
                labels,
            )

            n_voxels = len(
                vox.voxel_coords
            )

            assert n_voxels > 0
            assert n_voxels <= n_points

            assert vox.voxel_coords.shape == (
                n_voxels,
                3,
            )

            assert vox.voxel_features.shape == (
                n_voxels,
                1,
            )

            assert vox.voxel_labels.shape == (
                n_voxels,
            )

            assert vox.inverse_map.shape == (
                n_points,
            )

            assert (
                vox.point_to_voxel_counts.sum()
                == n_points
            )

            assert vox.inverse_map.min() >= 0
            assert vox.inverse_map.max() < n_voxels

            assert np.isfinite(
                vox.voxel_features
            ).all()

            assert vox.voxel_labels.min() >= 0
            assert vox.voxel_labels.max() <= 22

            # ------------------------------------------------
            # Inverse projection
            # ------------------------------------------------

            projected = (
                project_voxel_predictions_to_points(
                    vox.voxel_labels,
                    vox.inverse_map,
                )
            )

            # This must be exact.
            assert projected.shape == labels.shape
            assert len(projected) == n_points

            # This is diagnostic only.
            # Mixed-class voxels can legitimately disagree.
            agreement = float(
                np.mean(
                    projected == labels
                )
            )

            compression = (
                1.0
                - n_voxels / n_points
            )

            print(
                f"  frame={str(sample['frame_id']):24s} "
                f"points={n_points:8,d} "
                f"voxels={n_voxels:8,d} "
                f"reduction={100.0 * compression:6.2f}% "
                f"label_agreement={100.0 * agreement:6.2f}% "
                f"feat=[{features.min():.4f},"
                f"{features.max():.4f}]"
            )

            s = summary[dataset_name]

            s["frames"] += 1
            s["points"] += n_points
            s["voxels"] += n_voxels
            s["agreement_sum"] += agreement

    print()
    print("=" * 100)
    print(f"SUMMARY — {mode}")
    print("=" * 100)

    for dataset_name in DATASET_ORDER:

        s = summary[dataset_name]

        reduction = (
            1.0
            - s["voxels"] / s["points"]
        )

        agreement = (
            s["agreement_sum"]
            / s["frames"]
        )

        print(
            f"{dataset_name:20s} "
            f"frames={s['frames']:2d} "
            f"points={s['points']:10,d} "
            f"voxels={s['voxels']:10,d} "
            f"reduction={100.0 * reduction:6.2f}% "
            f"label_agreement={100.0 * agreement:6.2f}%"
        )


def main():

    args = parse_args()

    repo_root = Path(
        args.repo_root
    ).resolve()

    cache_dir = Path(
        args.cache_dir
    ).resolve()

    normalization_config = (
        repo_root
        / args.normalization_config
    )

    print("=" * 100)
    print(
        "SIH26053 — REAL F0 FEATURE/VOXEL PIPELINE VALIDATION"
    )
    print("=" * 100)

    print(f"Cache      : {cache_dir}")
    print(f"Repo root  : {repo_root}")
    print(
        f"Voxel size : "
        f"{args.voxel_size:.3f} m"
    )

    ds = F0TarDataset(
        cache_dir=cache_dir,
        repo_root=repo_root,
        split="train",
        strict=True,
    )

    print(
        f"TRAIN frames indexed: "
        f"{len(ds):,}"
    )

    # --------------------------------------------------------
    # Select deterministic real frames from every dataset
    # --------------------------------------------------------

    indices = {}

    for dataset_name in DATASET_ORDER:

        pool = ds.dataset_to_indices[
            dataset_name
        ]

        if len(pool) < args.frames_per_dataset:
            raise RuntimeError(
                f"{dataset_name}: only "
                f"{len(pool)} frames available"
            )

        # Beginning + end when requesting 2 frames.
        if args.frames_per_dataset == 1:
            chosen = [
                pool[0]
            ]

        elif args.frames_per_dataset == 2:
            chosen = [
                pool[0],
                pool[-1],
            ]

        else:
            positions = np.linspace(
                0,
                len(pool) - 1,
                args.frames_per_dataset,
                dtype=int,
            )

            chosen = [
                pool[int(i)]
                for i in positions
            ]

        indices[
            dataset_name
        ] = chosen

    voxel_adapter = VoxelAdapter(
        voxel_size=args.voxel_size,
        ignore_index=0,
    )

    # --------------------------------------------------------
    # Mode A: geometry-only
    # --------------------------------------------------------

    constant_adapter = FeatureAdapter(
        mode="constant",
    )

    validate_mode(
        ds,
        constant_adapter,
        voxel_adapter,
        indices,
        "constant",
    )

    # --------------------------------------------------------
    # Mode B: robust TRAIN-derived intensity
    # --------------------------------------------------------

    robust_adapter = FeatureAdapter(
        mode="robust_intensity",
        normalization_config=(
            normalization_config
        ),
    )

    validate_mode(
        ds,
        robust_adapter,
        voxel_adapter,
        indices,
        "robust_intensity",
    )

    ds.close()

    print()
    print("=" * 100)
    print(
        "REAL F0 FEATURE/VOXEL PIPELINE: PASS"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()