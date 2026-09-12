"""
FOVEAMAP A14 — Predicted-Path Foveation Benchmark

Benchmarks predicted-path foveation on a real SemanticKITTI frame.

Scenarios:
    0 m/s straight
    5 m/s straight
    10 m/s straight
    5 m/s left turn
    5 m/s right turn

Outputs:
    - active points
    - active percentage
    - importance statistics
    - resolution distribution
    - computation latency
    - speed effect
    - turning effect
    - visualization
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mapping.predicted_path_foveation import (
    VehicleState,
    path_importance,
    predict_path,
    resolution_from_path_importance,
)


DATASET_PATH = (
    ROOT
    / "datasets"
    / "TEST"
    / "000000.bin"
)

OUTPUT_PATH = (
    ROOT
    / "a14_predicted_path_foveation.png"
)


def load_kitti_bin(path: Path) -> np.ndarray:
    """Load a SemanticKITTI LiDAR .bin file."""

    if not path.exists():
        raise FileNotFoundError(
            f"LiDAR frame not found: {path}"
        )

    raw = np.fromfile(
        path,
        dtype=np.float32,
    )

    if raw.size % 4 != 0:
        raise ValueError(
            "Invalid SemanticKITTI .bin file"
        )

    points = raw.reshape(
        -1,
        4,
    )

    return points[:, :3].astype(
        np.float64
    )


def run_scenario(
    name: str,
    xyz: np.ndarray,
    state: VehicleState,
    horizon: float = 5.0,
    step: float = 0.25,
    corridor_width: float = 3.0,
    runs: int = 20,
) -> dict:
    """Run one benchmark scenario."""

    # Warm-up
    for _ in range(3):
        importance = path_importance(
            xyz,
            state,
            horizon=horizon,
            step=step,
            corridor_width=corridor_width,
        )

        resolutions = (
            resolution_from_path_importance(
                importance
            )
        )

    timings = []

    for _ in range(runs):
        start = time.perf_counter()

        importance = path_importance(
            xyz,
            state,
            horizon=horizon,
            step=step,
            corridor_width=corridor_width,
        )

        resolutions = (
            resolution_from_path_importance(
                importance
            )
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        timings.append(
            elapsed * 1000.0
        )

    timings = np.asarray(
        timings,
        dtype=np.float64,
    )

    active = importance > 0.0

    active_count = int(
        np.count_nonzero(active)
    )

    total_count = len(xyz)

    active_percentage = (
        100.0
        * active_count
        / total_count
        if total_count
        else 0.0
    )

    counts = {
        "5cm": int(
            np.count_nonzero(
                resolutions == 0.05
            )
        ),
        "10cm": int(
            np.count_nonzero(
                resolutions == 0.10
            )
        ),
        "20cm": int(
            np.count_nonzero(
                resolutions == 0.20
            )
        ),
        "40cm": int(
            np.count_nonzero(
                resolutions == 0.40
            )
        ),
    }

    return {
        "name": name,
        "state": state,
        "importance": importance,
        "resolutions": resolutions,
        "active": active,
        "active_count": active_count,
        "active_percentage": active_percentage,
        "importance_min": float(
            np.min(importance)
        ),
        "importance_max": float(
            np.max(importance)
        ),
        "importance_mean": float(
            np.mean(importance)
        ),
        "importance_median": float(
            np.median(importance)
        ),
        "resolution_counts": counts,
        "latency_mean": float(
            np.mean(timings)
        ),
        "latency_median": float(
            np.median(timings)
        ),
        "latency_p95": float(
            np.percentile(
                timings,
                95,
            )
        ),
    }


def print_result(result: dict) -> None:
    """Print benchmark result."""

    state = result["state"]
    counts = result["resolution_counts"]

    print()
    print("=" * 70)
    print(result["name"])
    print("=" * 70)

    print(
        f"Speed:              "
        f"{state.speed:.2f} m/s"
    )

    print(
        f"Yaw rate:           "
        f"{state.yaw_rate:.3f} rad/s"
    )

    print(
        f"Active points:      "
        f"{result['active_count']:,}"
    )

    print(
        f"Active percentage:  "
        f"{result['active_percentage']:.2f}%"
    )

    print(
        f"Importance min:     "
        f"{result['importance_min']:.6f}"
    )

    print(
        f"Importance max:     "
        f"{result['importance_max']:.6f}"
    )

    print(
        f"Importance mean:    "
        f"{result['importance_mean']:.6f}"
    )

    print(
        f"Importance median:  "
        f"{result['importance_median']:.6f}"
    )

    print()
    print("Resolution allocation:")

    print(
        f"  5 cm:             "
        f"{counts['5cm']:,}"
    )

    print(
        f"  10 cm:            "
        f"{counts['10cm']:,}"
    )

    print(
        f"  20 cm:            "
        f"{counts['20cm']:,}"
    )

    print(
        f"  40 cm:            "
        f"{counts['40cm']:,}"
    )

    print()
    print(
        f"Mean latency:       "
        f"{result['latency_mean']:.3f} ms"
    )

    print(
        f"Median latency:     "
        f"{result['latency_median']:.3f} ms"
    )

    print(
        f"P95 latency:        "
        f"{result['latency_p95']:.3f} ms"
    )


def plot_result(
    ax,
    xyz: np.ndarray,
    result: dict,
    title: str,
) -> None:
    """Plot one predicted-path foveation scenario."""

    active = result["active"]

    resolutions = result["resolutions"]

    masks = [
        resolutions == 0.40,
        resolutions == 0.20,
        resolutions == 0.10,
        resolutions == 0.05,
    ]

    sizes = [
        1,
        3,
        6,
        10,
    ]

    labels = [
        "40 cm",
        "20 cm",
        "10 cm",
        "5 cm",
    ]

    for mask, size, label in zip(
        masks,
        sizes,
        labels,
    ):
        selected = (
            mask
            & active
        )

        if np.any(selected):
            ax.scatter(
                xyz[selected, 0],
                xyz[selected, 1],
                s=size,
                alpha=0.55,
                label=label,
            )

    path = predict_path(
        result["state"],
        horizon=5.0,
        step=0.1,
    )

    ax.plot(
        path[:, 0],
        path[:, 1],
        linewidth=2.0,
        label="Predicted path",
    )

    ax.scatter(
        [0.0],
        [0.0],
        s=45,
        marker="x",
        label="Vehicle",
    )

    ax.set_title(title)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.grid(
        True,
        alpha=0.25,
    )


def main() -> None:
    print()
    print("=" * 70)
    print("FOVEAMAP A14 — PREDICTED-PATH FOVEATION")
    print("=" * 70)

    print()
    print(
        f"Loading: {DATASET_PATH}"
    )

    xyz = load_kitti_bin(
        DATASET_PATH
    )

    print(
        f"Loaded points: {len(xyz):,}"
    )

    print(
        f"XYZ shape: {xyz.shape}"
    )

    scenarios = [
        (
            "0mps_straight",
            VehicleState(
                speed=0.0,
                yaw_rate=0.0,
            ),
        ),
        (
            "5mps_straight",
            VehicleState(
                speed=5.0,
                yaw_rate=0.0,
            ),
        ),
        (
            "10mps_straight",
            VehicleState(
                speed=10.0,
                yaw_rate=0.0,
            ),
        ),
        (
            "5mps_left_turn",
            VehicleState(
                speed=5.0,
                yaw_rate=0.20,
            ),
        ),
        (
            "5mps_right_turn",
            VehicleState(
                speed=5.0,
                yaw_rate=-0.20,
            ),
        ),
    ]

    results = []

    for name, state in scenarios:

        result = run_scenario(
            name,
            xyz,
            state,
        )

        results.append(
            result
        )

        print_result(
            result
        )

    print()
    print("=" * 70)
    print("SPEED EFFECT")
    print("=" * 70)

    speed_results = [
        result
        for result in results
        if result["state"].yaw_rate == 0.0
    ]

    for result in speed_results:
        print(
            f"{result['name']}: "
            f"{result['active_count']:,} active "
            f"({result['active_percentage']:.2f}%)"
        )

    if len(speed_results) >= 3:

        speed_active = [
            result["active_count"]
            for result in speed_results
        ]

        if all(
            speed_active[index]
            <= speed_active[index + 1]
            for index in range(
                len(speed_active) - 1
            )
        ):
            print(
                "PASS: Increasing speed "
                "does not shrink predicted-path "
                "foveation."
            )
        else:
            print(
                "WARNING: Active foveation "
                "did not increase monotonically "
                "with speed."
            )

    print()
    print("=" * 70)
    print("TURNING EFFECT")
    print("=" * 70)

    left = next(
        result
        for result in results
        if result["name"]
        == "5mps_left_turn"
    )

    right = next(
        result
        for result in results
        if result["name"]
        == "5mps_right_turn"
    )

    left_path = predict_path(
        left["state"],
        horizon=5.0,
        step=0.1,
    )

    right_path = predict_path(
        right["state"],
        horizon=5.0,
        step=0.1,
    )

    print(
        "Left final Y:      "
        f"{left_path[-1, 1]:.3f} m"
    )

    print(
        "Right final Y:     "
        f"{right_path[-1, 1]:.3f} m"
    )

    if (
        left_path[-1, 1] > 0.0
        and right_path[-1, 1] < 0.0
    ):
        print(
            "PASS: Left/right turning "
            "produces opposite predicted "
            "path curvature."
        )
    else:
        print(
            "WARNING: Turning curvature "
            "check failed."
        )

    symmetry_error = np.mean(
        np.abs(
            left_path[:, 0]
            - right_path[:, 0]
        )
    )

    y_symmetry_error = np.mean(
        np.abs(
            left_path[:, 1]
            + right_path[:, 1]
        )
    )

    print(
        f"X symmetry error:  "
        f"{symmetry_error:.6f} m"
    )

    print(
        f"Y symmetry error:  "
        f"{y_symmetry_error:.6f} m"
    )

    if y_symmetry_error < 1e-6:
        print(
            "PASS: Left/right paths "
            "are symmetric."
        )

    print()
    print("=" * 70)
    print("VISUALIZATION")
    print("=" * 70)

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(18, 11),
    )

    plot_result(
        axes[0, 0],
        xyz,
        results[0],
        "0 m/s — Straight",
    )

    plot_result(
        axes[0, 1],
        xyz,
        results[1],
        "5 m/s — Straight",
    )

    plot_result(
        axes[0, 2],
        xyz,
        results[2],
        "10 m/s — Straight",
    )

    plot_result(
        axes[1, 0],
        xyz,
        results[3],
        "5 m/s — Left Turn",
    )

    plot_result(
        axes[1, 1],
        xyz,
        results[4],
        "5 m/s — Right Turn",
    )

    axes[1, 2].axis("off")

    handles, labels = (
        axes[0, 0].get_legend_handles_labels()
    )

    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
    )

    figure.suptitle(
        "FoveaMap A14 — Predicted-Path Foveation",
        fontsize=16,
    )

    figure.tight_layout(
        rect=(0, 0.05, 1, 0.95)
    )

    figure.savefig(
        OUTPUT_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        f"Saved visualization: "
        f"{OUTPUT_PATH}"
    )

    print()
    print("=" * 70)
    print("A14 BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()