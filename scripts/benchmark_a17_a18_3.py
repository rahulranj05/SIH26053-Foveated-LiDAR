from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from mapping.foveation_controller import compute_foveation
from mapping.hierarchical_foveated_mapper import (
    build_hierarchical_foveated_map,
    validate_leaf_partition,
)


FRAME_PATH = ROOT / "datasets" / "TEST" / "000000.bin"


VALID_REASONS = (
    "DISTANCE",
    "KINEMATIC",
    "PREDICTED_PATH",
    "SEMANTIC",
    "DYNAMIC",
)


def load_kitti_frame(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"SemanticKITTI test frame not found:\n{path}"
        )

    raw = np.fromfile(path, dtype=np.float32)

    if raw.size % 4 != 0:
        raise ValueError(
            f"Invalid KITTI .bin file. "
            f"Expected a multiple of 4 floats, got {raw.size}."
        )

    points = raw.reshape(-1, 4)

    return points[:, :3]


def build_test_signals(xyz: np.ndarray) -> dict[str, np.ndarray]:
    """
    Build deterministic controlled inputs for A17.

    These are integration-test signals, NOT semantic or
    dynamic ground truth.
    """

    n = len(xyz)

    distance = np.linalg.norm(
        xyz[:, :2],
        axis=1,
    )

    # ---------------------------------------------------------
    # Distance importance
    # ---------------------------------------------------------

    distance_importance = np.clip(
        1.0 - distance / 100.0,
        0.0,
        1.0,
    )

    # ---------------------------------------------------------
    # Kinematic importance
    #
    # Controlled vehicle state:
    #   speed   = 5 m/s
    #   yawrate = 0.15 rad/s
    # ---------------------------------------------------------

    speed = 5.0
    yaw_rate = 0.15

    forward = xyz[:, 0]

    kinematic_importance = np.clip(
        (forward / 30.0)
        + 0.35
        + (abs(yaw_rate) * 0.5)
        + (speed / 20.0),
        0.0,
        1.0,
    )

    kinematic_importance = np.where(
        forward > 0.0,
        kinematic_importance,
        kinematic_importance * 0.35,
    )

    # ---------------------------------------------------------
    # Predicted-path importance
    #
    # Controlled curved path approximation.
    # ---------------------------------------------------------

    path_center_y = (
        0.12 * forward * forward / 10.0
    )

    path_distance = np.abs(
        xyz[:, 1] - path_center_y
    )

    predicted_path_importance = np.exp(
        -(path_distance ** 2)
        / (2.0 * 2.5 ** 2)
    )

    predicted_path_importance *= np.clip(
        1.0 - distance / 100.0,
        0.0,
        1.0,
    )

    predicted_path_importance = np.clip(
        predicted_path_importance,
        0.0,
        1.0,
    )

    # ---------------------------------------------------------
    # Semantic importance
    #
    # Controlled spatial regions.
    # NOT semantic ground truth.
    # ---------------------------------------------------------

    semantic_importance = np.full(
        n,
        0.05,
        dtype=np.float64,
    )

    obstacle_region = (
        (np.abs(xyz[:, 0] - 12.0) < 2.0)
        & (np.abs(xyz[:, 1]) < 2.5)
    )

    vehicle_region = (
        (np.abs(xyz[:, 0] - 20.0) < 2.0)
        & (np.abs(xyz[:, 1] - 5.0) < 2.0)
    )

    vulnerable_region = (
        (np.abs(xyz[:, 0] - 15.0) < 1.0)
        & (np.abs(xyz[:, 1] + 4.0) < 1.0)
    )

    semantic_importance[obstacle_region] = 0.75
    semantic_importance[vehicle_region] = 0.85
    semantic_importance[vulnerable_region] = 1.0

    # ---------------------------------------------------------
    # Dynamic importance
    #
    # Controlled moving-object regions.
    # NOT dynamic ground truth.
    # ---------------------------------------------------------

    dynamic_importance = np.zeros(
        n,
        dtype=np.float64,
    )

    dynamic_region_1 = (
        (np.abs(xyz[:, 0] - 18.0) < 2.0)
        & (np.abs(xyz[:, 1] - 3.0) < 2.0)
    )

    dynamic_region_2 = (
        (np.abs(xyz[:, 0] - 28.0) < 2.5)
        & (np.abs(xyz[:, 1] + 6.0) < 2.5)
    )

    dynamic_importance[dynamic_region_1] = 0.85
    dynamic_importance[dynamic_region_2] = 1.0

    return {
        "distance": distance_importance,
        "kinematic": kinematic_importance,
        "predicted_path": predicted_path_importance,
        "semantic": semantic_importance,
        "dynamic": dynamic_importance,
    }


def benchmark_controller(
    importance_signals: dict[str, np.ndarray],
    runs: int = 20,
):
    timings = []
    result = None

    # Warmup
    for _ in range(3):
        result = compute_foveation(
            importance_signals
        )

    # Timed runs
    for _ in range(runs):
        start = time.perf_counter()

        result = compute_foveation(
            importance_signals
        )

        elapsed = (
            time.perf_counter() - start
        ) * 1000.0

        timings.append(elapsed)

    return result, np.asarray(timings)


def benchmark_mapper(
    xyz: np.ndarray,
    desired_resolution: np.ndarray,
    dominant_reason: np.ndarray,
    runs: int = 20,
):
    timings = []
    result = None

    # Warmup
    for _ in range(3):
        result = build_hierarchical_foveated_map(
            xyz=xyz,
            desired_resolution=desired_resolution,
            dominant_reason=dominant_reason,
        )

    # Timed runs
    for _ in range(runs):
        start = time.perf_counter()

        result = build_hierarchical_foveated_map(
            xyz=xyz,
            desired_resolution=desired_resolution,
            dominant_reason=dominant_reason,
        )

        elapsed = (
            time.perf_counter() - start
        ) * 1000.0

        timings.append(elapsed)

    return result, np.asarray(timings)


def print_timing(
    label: str,
    timings: np.ndarray,
):
    print(f"\n{label}")
    print("-" * len(label))

    print(
        f"Mean   : {timings.mean():.3f} ms"
    )

    print(
        f"Median : {np.median(timings):.3f} ms"
    )

    print(
        f"Min    : {timings.min():.3f} ms"
    )

    print(
        f"Max    : {timings.max():.3f} ms"
    )

    print(
        f"P95    : "
        f"{np.percentile(timings, 95):.3f} ms"
    )


def print_resolution_distribution(
    title: str,
    resolutions: np.ndarray,
):
    unique_resolutions, counts = np.unique(
        resolutions,
        return_counts=True,
    )

    print(f"\n{title}")
    print("-" * len(title))

    total = len(resolutions)

    for resolution, count in zip(
        unique_resolutions,
        counts,
    ):
        percentage = (
            count / total * 100.0
        )

        print(
            f"  {resolution:.2f} m:"
            f" {count:,}"
            f" ({percentage:.2f}%)"
        )


def main():
    print("=" * 70)
    print("FOVEAMAP — A17 → A18.3 INTEGRATION BENCHMARK")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load real SemanticKITTI frame
    # ---------------------------------------------------------

    xyz = load_kitti_frame(
        FRAME_PATH
    )

    print("\nInput frame")
    print("------------")

    print(
        f"Points loaded : {len(xyz):,}"
    )

    # Keep the same 100 m evaluation region
    # used by previous mapper benchmarks.

    radius = np.linalg.norm(
        xyz[:, :2],
        axis=1,
    )

    mask = radius <= 100.0

    xyz = xyz[mask]

    print(
        f"Points <=100m : {len(xyz):,}"
    )

    # ---------------------------------------------------------
    # Build A17 inputs
    # ---------------------------------------------------------

    importance_signals = build_test_signals(
        xyz
    )

    print("\nA17 input signals")
    print("-----------------")

    for name, values in importance_signals.items():
        print(
            f"  {name:<18}"
            f" min={values.min():.3f}"
            f" max={values.max():.3f}"
            f" mean={values.mean():.3f}"
        )

    # ---------------------------------------------------------
    # A17 controller
    # ---------------------------------------------------------

    (
        foveation_result,
        controller_timings,
    ) = benchmark_controller(
        importance_signals
    )

    print_timing(
        "A17 FOVEATION CONTROLLER",
        controller_timings,
    )

    importance = (
        foveation_result["importance"]
    )

    desired_resolution = (
        foveation_result["resolution"]
    )

    dominant_reason = (
        foveation_result["dominant_reason"]
    )

    # ---------------------------------------------------------
    # A17 output
    # ---------------------------------------------------------

    print(
        "\nA17 importance statistics"
    )
    print("-------------------------")

    print(
        f"Mean   : {importance.mean():.6f}"
    )

    print(
        f"Median : {np.median(importance):.6f}"
    )

    print(
        f"Min    : {importance.min():.6f}"
    )

    print(
        f"Max    : {importance.max():.6f}"
    )

    print_resolution_distribution(
        "A17 desired-resolution distribution",
        desired_resolution,
    )

    unique_reasons, reason_counts = np.unique(
        dominant_reason,
        return_counts=True,
    )

    print(
        "\nA17 dominant reasons"
    )
    print("--------------------")

    for reason, count in zip(
        unique_reasons,
        reason_counts,
    ):
        percentage = (
            count / len(dominant_reason)
            * 100.0
        )

        print(
            f"  {reason:<18}:"
            f" {count:,}"
            f" ({percentage:.2f}%)"
        )

    # ---------------------------------------------------------
    # A18.3 mapper
    # ---------------------------------------------------------

    (
        map_result,
        mapper_timings,
    ) = benchmark_mapper(
        xyz,
        desired_resolution,
        dominant_reason,
    )

    print_timing(
        "A18.3 HIERARCHICAL FOVEATED MAPPER",
        mapper_timings,
    )

    # ---------------------------------------------------------
    # Combined timing
    # ---------------------------------------------------------

    combined_timings = (
        controller_timings
        + mapper_timings
    )

    print_timing(
        "A17 + A18.3 COMBINED",
        combined_timings,
    )

    # ---------------------------------------------------------
    # Final map statistics
    # ---------------------------------------------------------

    print(
        "\nFINAL HIERARCHICAL MAP"
    )
    print("----------------------")

    print(
        f"Input points      : "
        f"{map_result.input_points:,}"
    )

    print(
        f"Active leaf cells : "
        f"{map_result.active_cells:,}"
    )

    mapped_points = int(
        map_result.point_count.sum()
    )

    print(
        f"Mapped points     : "
        f"{mapped_points:,}"
    )

    points_conserved = (
        mapped_points
        == len(xyz)
    )

    print(
        f"Point conservation: "
        f"{points_conserved}"
    )

    # ---------------------------------------------------------
    # Final leaf resolution
    # ---------------------------------------------------------

    print_resolution_distribution(
        "Final leaf-resolution distribution",
        map_result.resolution,
    )

    # ---------------------------------------------------------
    # Hierarchy distribution
    # ---------------------------------------------------------

    unique_levels, level_counts = np.unique(
        map_result.level,
        return_counts=True,
    )

    print(
        "\nHierarchy distribution"
    )
    print("----------------------")

    for level, count in zip(
        unique_levels,
        level_counts,
    ):
        percentage = (
            count
            / map_result.active_cells
            * 100.0
        )

        print(
            f"  Level {int(level)}:"
            f" {count:,}"
            f" ({percentage:.2f}%)"
        )

    # ---------------------------------------------------------
    # Final dominant reasons
    # ---------------------------------------------------------

    unique_leaf_reasons, leaf_reason_counts = (
        np.unique(
            map_result.dominant_reason,
            return_counts=True,
        )
    )

    print(
        "\nFinal dominant foveation reasons"
    )
    print("---------------------------------")

    for reason, count in zip(
        unique_leaf_reasons,
        leaf_reason_counts,
    ):
        percentage = (
            count
            / map_result.active_cells
            * 100.0
        )

        print(
            f"  {reason:<18}:"
            f" {count:,}"
            f" ({percentage:.2f}%)"
        )

    # ---------------------------------------------------------
    # Structural validation
    # ---------------------------------------------------------

    print(
        "\nSTRUCTURAL VALIDATION"
    )
    print("---------------------")

    partition_valid = (
        validate_leaf_partition(
            map_result
        )
    )

    unique_keys = np.unique(
        np.column_stack(
            (
                map_result.level,
                map_result.ix,
                map_result.iy,
            )
        ),
        axis=0,
    )

    no_duplicates = (
        len(unique_keys)
        == map_result.active_cells
    )

    print(
        f"Point conservation : "
        f"{points_conserved}"
    )

    print(
        f"No duplicate leaves: "
        f"{no_duplicates}"
    )

    print(
        f"Leaf partition valid: "
        f"{partition_valid}"
    )

    # ---------------------------------------------------------
    # Final status
    # ---------------------------------------------------------

    all_valid = (
        points_conserved
        and no_duplicates
        and partition_valid
    )

    print(
        "\nINTEGRATION STATUS"
    )
    print("------------------")

    if all_valid:
        print(
            "A17 → A18.3 integration: PASS"
        )
    else:
        print(
            "A17 → A18.3 integration: FAIL"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()