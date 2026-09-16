from __future__ import annotations

import time

import numpy as np

from perception_v1.src.perception.spvcnn.voxelization import (
    voxelize_points,
)


def reference_voxelize(
    xyz: np.ndarray,
    voxel_size_m: float = 0.10,
):
    xyz = np.asarray(xyz, dtype=np.float32)

    coords = np.floor(
        xyz.astype(np.float64) / voxel_size_m
    ).astype(np.int64)

    voxels, inverse = np.unique(
        coords,
        axis=0,
        return_inverse=True,
    )

    inverse = inverse.astype(np.int64)

    ranges_squared = np.sum(
        xyz.astype(np.float64) ** 2,
        axis=1,
    )

    reps = np.empty(
        len(voxels),
        dtype=np.int64,
    )

    for voxel_id in range(len(voxels)):
        indices = np.flatnonzero(
            inverse == voxel_id
        )

        local_ranges = ranges_squared[indices]
        minimum = np.min(local_ranges)

        tied = indices[
            local_ranges == minimum
        ]

        reps[voxel_id] = np.min(tied)

    return voxels, inverse, reps


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:

    print("M7-B VOXELIZATION RUNTIME CLOSURE")
    print("=" * 50)

    # --------------------------------------------------
    # 1. Random adversarial equivalence
    # --------------------------------------------------
    rng = np.random.default_rng(26053)

    xyz = rng.uniform(
        -20.0,
        20.0,
        size=(5000, 3),
    ).astype(np.float32)

    # Force duplicate/same-voxel cases.
    xyz[100:200] = xyz[0:100]

    old_vox, old_inv, old_rep = reference_voxelize(xyz)
    new = voxelize_points(xyz)

    check(
        np.array_equal(
            new.voxel_coordinates,
            old_vox,
        ),
        "Voxel coordinates differ from reference",
    )

    check(
        np.array_equal(
            new.point_to_voxel_inverse,
            old_inv,
        ),
        "Inverse mapping differs from reference",
    )

    check(
        np.array_equal(
            new.voxel_to_representative_point,
            old_rep,
        ),
        "Representative points differ from reference",
    )

    print("[PASS] Reference implementation equivalence")

    # --------------------------------------------------
    # 2. Exact range tie -> lowest canonical index
    # --------------------------------------------------
    tie_xyz = np.asarray(
        [
            [1.01, 1.01, 1.01],
            [1.01, 1.01, 1.01],
            [1.08, 1.08, 1.08],
        ],
        dtype=np.float32,
    )

    tie = voxelize_points(tie_xyz)

    check(
        len(tie.voxel_coordinates) == 1,
        "Tie test did not remain in one voxel",
    )

    check(
        int(
            tie.voxel_to_representative_point[0]
        ) == 0,
        "Exact tie did not choose lowest canonical index",
    )

    print("[PASS] Exact tie chooses lowest canonical index")

    # --------------------------------------------------
    # 3. Negative signed-floor behavior
    # --------------------------------------------------
    neg_xyz = np.asarray(
        [
            [-0.01, -0.01, -0.01],
            [-0.11, -0.21, -0.31],
            [0.01, 0.01, 0.01],
        ],
        dtype=np.float32,
    )

    neg = voxelize_points(neg_xyz)

    expected = np.unique(
        np.floor(
            neg_xyz.astype(np.float64) / 0.10
        ).astype(np.int64),
        axis=0,
    )

    check(
        np.array_equal(
            neg.voxel_coordinates,
            expected,
        ),
        "Signed-floor negative coordinates changed",
    )

    check(
        np.any(neg.voxel_coordinates < 0),
        "Negative voxel coordinates disappeared",
    )

    print("[PASS] Signed-floor negative coordinates preserved")

    # --------------------------------------------------
    # 4. Determinism
    # --------------------------------------------------
    a = voxelize_points(xyz)
    b = voxelize_points(xyz)

    check(
        np.array_equal(
            a.voxel_coordinates,
            b.voxel_coordinates,
        )
        and np.array_equal(
            a.point_to_voxel_inverse,
            b.point_to_voxel_inverse,
        )
        and np.array_equal(
            a.voxel_to_representative_point,
            b.voxel_to_representative_point,
        ),
        "Vectorized voxelization is nondeterministic",
    )

    print("[PASS] Deterministic replay")

    # --------------------------------------------------
    # 5. Large-frame runtime sanity
    #
    # This is not a hardware-independent benchmark.
    # It only falsifies catastrophic behavior.
    # --------------------------------------------------
    large_xyz = rng.uniform(
        -80.0,
        80.0,
        size=(131072, 3),
    ).astype(np.float32)

    start = time.perf_counter()

    large = voxelize_points(
        large_xyz
    )

    elapsed = time.perf_counter() - start

    check(
        len(large.point_to_voxel_inverse)
        == len(large_xyz),
        "Large-frame point mapping incomplete",
    )

    check(
        len(large.voxel_to_representative_point)
        == len(large.voxel_coordinates),
        "Large-frame representative mapping incomplete",
    )

    # Generous ceiling: catches catastrophic regressions,
    # not ordinary machine-to-machine timing variation.
    check(
        elapsed < 15.0,
        f"Large-frame runtime unexpectedly high: "
        f"{elapsed:.3f}s",
    )

    print(
        f"[PASS] 131072-point runtime sanity: "
        f"{elapsed:.3f}s"
    )

    print("=" * 50)
    print("M7-B VOXELIZATION RUNTIME CLOSURE: PASS")


if __name__ == "__main__":
    main()