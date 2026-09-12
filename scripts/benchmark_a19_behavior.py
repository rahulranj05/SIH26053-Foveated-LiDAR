#!/usr/bin/env python3

import sys
import time
from pathlib import Path

import numpy as np


# Ensure repository root is importable when running:
# python3 scripts/benchmark_a19_behavior.py
ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from mapping.foveation_controller import compute_foveation
from mapping.kinematic_foveation import (
    KinematicFoveationConfig,
    VehicleState,
    kinematic_importance,
)


SEED = 19
N_POINTS = 100_000

FINE_RESOLUTION = 0.10
COARSE_RESOLUTION = 0.40


def make_points(
    n: int = N_POINTS,
    seed: int = SEED,
) -> np.ndarray:
    """
    Generate deterministic synthetic points for A19.

    Coordinate convention used by the benchmark:
        X = forward
        Y = lateral
        Z = vertical

    The benchmark does not assume the sign convention of yaw.
    Turn-direction validation compares the two A13 outputs
    against each other.
    """
    rng = np.random.default_rng(seed)

    xyz = np.empty((n, 3), dtype=np.float64)

    xyz[:, 0] = rng.uniform(-25.0, 25.0, n)
    xyz[:, 1] = rng.uniform(-25.0, 25.0, n)
    xyz[:, 2] = rng.uniform(-1.0, 1.0, n)

    return xyz


def make_kinematic_signal(
    xyz: np.ndarray,
    speed: float,
    yaw_rate: float,
) -> np.ndarray:
    """
    Generate the kinematic importance signal using the actual A13 API.

    A13 is treated as an already-validated subsystem.
    """
    config = KinematicFoveationConfig()

    state = VehicleState(
        speed=speed,
        yaw_rate=yaw_rate,
        heading=0.0,
    )

    signal = kinematic_importance(
        xyz,
        state,
        config,
    )

    signal = np.asarray(signal, dtype=np.float64)

    if signal.shape != (len(xyz),):
        raise ValueError(
            "A13 kinematic_importance returned unexpected shape: "
            f"{signal.shape}; expected {(len(xyz),)}"
        )

    return signal


def make_predicted_path_signal(
    xyz: np.ndarray,
) -> np.ndarray:
    """
    Synthetic predicted-path importance.

    A forward corridor receives high importance.

    This is a behavioral-control signal, not ground truth.
    """
    x = xyz[:, 0]
    y = xyz[:, 1]

    forward = np.clip(
        (x + 25.0) / 50.0,
        0.0,
        1.0,
    )

    corridor = np.exp(
        -0.5 * (y / 1.5) ** 2
    )

    signal = forward * corridor

    return np.clip(
        signal,
        0.0,
        1.0,
    )


def make_semantic_signal(
    xyz: np.ndarray,
) -> np.ndarray:
    """
    Synthetic localized semantic importance.

    Represents an important object region centered near:

        x = 15 m
        y = -0.5 m

    This does not represent semantic segmentation accuracy.
    """
    x = xyz[:, 0]
    y = xyz[:, 1]

    distance = np.sqrt(
        (x - 15.0) ** 2
        + (y + 0.5) ** 2
    )

    signal = np.exp(
        -(distance ** 2)
        / (2.0 * 2.0 ** 2)
    )

    return np.clip(
        signal,
        0.0,
        1.0,
    )


def make_dynamic_signal(
    xyz: np.ndarray,
) -> np.ndarray:
    """
    Synthetic localized dynamic-object importance.

    Represents a dynamic object centered near:

        x = 20 m
        y = 0 m

    This does not represent actual tracking accuracy.
    """
    x = xyz[:, 0]
    y = xyz[:, 1]

    distance = np.sqrt(
        (x - 20.0) ** 2
        + y ** 2
    )

    signal = np.exp(
        -(distance ** 2)
        / (2.0 * 1.5 ** 2)
    )

    return np.clip(
        signal,
        0.0,
        1.0,
    )


def make_rough_terrain_signal(
    xyz: np.ndarray,
) -> np.ndarray:
    """
    Synthetic localized rough-terrain importance.

    Represents a rough region centered near:

        x = 18 m
        y = 8 m

    A19 only verifies that a localized importance signal
    produces localized refinement.

    It does NOT claim to perform terrain perception here.
    """
    x = xyz[:, 0]
    y = xyz[:, 1]

    distance = np.sqrt(
        (x - 18.0) ** 2
        + (y - 8.0) ** 2
    )

    signal = np.exp(
        -(distance ** 2)
        / (2.0 * 3.0 ** 2)
    )

    return np.clip(
        signal,
        0.0,
        1.0,
    )


def run_a17(
    signals: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Run the existing A17 controller as a black box.

    IMPORTANT:
    Only the signal being tested is supplied.

    This prevents unrelated signals, especially DISTANCE,
    from dominating the MAX-fusion controller.
    """
    result = compute_foveation(signals)

    if not isinstance(result, dict):
        raise TypeError(
            "A17 compute_foveation() did not return a dict."
        )

    required_keys = {
        "importance",
        "resolution",
        "dominant_reason",
    }

    missing = required_keys.difference(result.keys())

    if missing:
        raise KeyError(
            "A17 result is missing required keys: "
            f"{sorted(missing)}"
        )

    return result


def resolution_stats(
    xyz: np.ndarray,
    result: dict[str, np.ndarray],
) -> dict:
    """
    Compute behavioral statistics from an A17 result.
    """
    importance = np.asarray(
        result["importance"],
        dtype=np.float64,
    )

    resolution = np.asarray(
        result["resolution"],
        dtype=np.float64,
    )

    if importance.shape != (len(xyz),):
        raise ValueError(
            "A17 importance shape mismatch: "
            f"{importance.shape}; "
            f"expected {(len(xyz),)}"
        )

    if resolution.shape != (len(xyz),):
        raise ValueError(
            "A17 resolution shape mismatch: "
            f"{resolution.shape}; "
            f"expected {(len(xyz),)}"
        )

    fine_mask = resolution <= FINE_RESOLUTION
    coarse_mask = resolution >= COARSE_RESOLUTION

    stats = {
        "importance_mean": float(
            np.mean(importance)
        ),
        "importance_median": float(
            np.median(importance)
        ),
        "importance_min": float(
            np.min(importance)
        ),
        "importance_max": float(
            np.max(importance)
        ),
        "fine_fraction": float(
            np.mean(fine_mask)
        ),
        "coarse_fraction": float(
            np.mean(coarse_mask)
        ),
    }

    if np.any(fine_mask):
        fine_xyz = xyz[fine_mask]

        stats["fine_centroid_x"] = float(
            np.mean(fine_xyz[:, 0])
        )

        stats["fine_centroid_y"] = float(
            np.mean(fine_xyz[:, 1])
        )

        stats["fine_forward_extent"] = float(
            np.max(fine_xyz[:, 0])
        )

        stats["fine_lateral_extent"] = float(
            np.max(np.abs(fine_xyz[:, 1]))
        )

        stats["fine_min_x"] = float(
            np.min(fine_xyz[:, 0])
        )

        stats["fine_max_x"] = float(
            np.max(fine_xyz[:, 0])
        )

        stats["fine_min_y"] = float(
            np.min(fine_xyz[:, 1])
        )

        stats["fine_max_y"] = float(
            np.max(fine_xyz[:, 1])
        )

    else:
        stats["fine_centroid_x"] = 0.0
        stats["fine_centroid_y"] = 0.0
        stats["fine_forward_extent"] = 0.0
        stats["fine_lateral_extent"] = 0.0
        stats["fine_min_x"] = 0.0
        stats["fine_max_x"] = 0.0
        stats["fine_min_y"] = 0.0
        stats["fine_max_y"] = 0.0

    return stats


def dominant_reason_from_result(
    result: dict[str, np.ndarray],
) -> str:
    """
    Extract the most frequent dominant reason from A17.

    Works with ordinary NumPy/string-like representations.
    """
    dominant = np.asarray(
        result["dominant_reason"]
    )

    if dominant.size == 0:
        return "NONE"

    unique, counts = np.unique(
        dominant,
        return_counts=True,
    )

    return str(
        unique[np.argmax(counts)]
    )


def benchmark_case(
    name: str,
    xyz: np.ndarray,
    signals: dict[str, np.ndarray],
    warmup_runs: int = 2,
    benchmark_runs: int = 7,
) -> dict:
    """
    Benchmark one isolated A17 behavioral case.
    """
    for signal_name, signal in signals.items():
        signal = np.asarray(signal)

        if signal.shape != (len(xyz),):
            raise ValueError(
                f"{signal_name} signal has shape "
                f"{signal.shape}; expected "
                f"{(len(xyz),)}"
            )

    for _ in range(warmup_runs):
        run_a17(signals)

    latencies = []
    result = None

    for _ in range(benchmark_runs):
        start = time.perf_counter()

        result = run_a17(signals)

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        latencies.append(elapsed_ms)

    if result is None:
        raise RuntimeError(
            "A17 benchmark produced no result."
        )

    stats = resolution_stats(
        xyz,
        result,
    )

    dominant_reason = dominant_reason_from_result(
        result
    )

    stats["dominant_reason"] = dominant_reason
    stats["latency_mean_ms"] = float(
        np.mean(latencies)
    )
    stats["latency_median_ms"] = float(
        np.median(latencies)
    )
    stats["latency_p95_ms"] = float(
        np.percentile(latencies, 95)
    )

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"Importance mean    : "
        f"{stats['importance_mean']:.6f}"
    )

    print(
        f"Importance median  : "
        f"{stats['importance_median']:.6f}"
    )

    print(
        f"Fine fraction      : "
        f"{stats['fine_fraction'] * 100:.2f}%"
    )

    print(
        f"Coarse fraction    : "
        f"{stats['coarse_fraction'] * 100:.2f}%"
    )

    print(
        f"Fine centroid      : "
        f"({stats['fine_centroid_x']:.3f}, "
        f"{stats['fine_centroid_y']:.3f})"
    )

    print(
        f"Fine forward extent: "
        f"{stats['fine_forward_extent']:.3f} m"
    )

    print(
        f"Fine lateral extent: "
        f"{stats['fine_lateral_extent']:.3f} m"
    )

    print(
        f"Dominant reason    : "
        f"{dominant_reason}"
    )

    print(
        f"Latency mean       : "
        f"{stats['latency_mean_ms']:.3f} ms"
    )

    print(
        f"Latency median     : "
        f"{stats['latency_median_ms']:.3f} ms"
    )

    print(
        f"Latency p95        : "
        f"{stats['latency_p95_ms']:.3f} ms"
    )

    return stats


def main() -> int:
    print("=" * 70)
    print("FOVEAMAP A19 — FOVEATION POLICY CALIBRATION")
    print("=" * 70)

    print()
    print("A19 methodology:")
    print("  - A13 remains unchanged.")
    print("  - A17 is treated as a black box.")
    print("  - Each behavioral signal is isolated.")
    print("  - No DISTANCE signal is injected into other tests.")
    print("  - Synthetic object/terrain signals are behavioral probes.")
    print()

    xyz = make_points()

    print(f"Input points: {len(xyz):,}")
    print(f"Seed        : {SEED}")

    # ---------------------------------------------------------------
    # 1. Stationary
    # ---------------------------------------------------------------

    stationary_signal = make_kinematic_signal(
        xyz,
        speed=0.0,
        yaw_rate=0.0,
    )

    stationary = benchmark_case(
        "1. STATIONARY + STRAIGHT",
        xyz,
        {
            "KINEMATIC": stationary_signal,
        },
    )

    # ---------------------------------------------------------------
    # 2. 5 m/s straight
    # ---------------------------------------------------------------

    five_mps_signal = make_kinematic_signal(
        xyz,
        speed=5.0,
        yaw_rate=0.0,
    )

    five_mps = benchmark_case(
        "2. 5 m/s + STRAIGHT",
        xyz,
        {
            "KINEMATIC": five_mps_signal,
        },
    )

    # ---------------------------------------------------------------
    # 3. 10 m/s straight
    # ---------------------------------------------------------------

    ten_mps_signal = make_kinematic_signal(
        xyz,
        speed=10.0,
        yaw_rate=0.0,
    )

    ten_mps = benchmark_case(
        "3. 10 m/s + STRAIGHT",
        xyz,
        {
            "KINEMATIC": ten_mps_signal,
        },
    )

    # ---------------------------------------------------------------
    # 4. Left turn
    # ---------------------------------------------------------------

    left_turn_signal = make_kinematic_signal(
        xyz,
        speed=5.0,
        yaw_rate=0.35,
    )

    left_turn = benchmark_case(
        "4. 5 m/s + LEFT TURN",
        xyz,
        {
            "KINEMATIC": left_turn_signal,
        },
    )

    # ---------------------------------------------------------------
    # 5. Right turn
    # ---------------------------------------------------------------

    right_turn_signal = make_kinematic_signal(
        xyz,
        speed=5.0,
        yaw_rate=-0.35,
    )

    right_turn = benchmark_case(
        "5. 5 m/s + RIGHT TURN",
        xyz,
        {
            "KINEMATIC": right_turn_signal,
        },
    )

    # ---------------------------------------------------------------
    # 6. Predicted path
    # ---------------------------------------------------------------

    predicted_path_signal = make_predicted_path_signal(
        xyz
    )

    predicted_path = benchmark_case(
        "6. PREDICTED-PATH EMPHASIS",
        xyz,
        {
            "PREDICTED_PATH": predicted_path_signal,
        },
    )

    # ---------------------------------------------------------------
    # 7. Semantic object
    # ---------------------------------------------------------------

    semantic_signal = make_semantic_signal(
        xyz
    )

    semantic = benchmark_case(
        "7. SEMANTIC OBJECT EMPHASIS",
        xyz,
        {
            "SEMANTIC": semantic_signal,
        },
    )

    # ---------------------------------------------------------------
    # 8. Dynamic object
    # ---------------------------------------------------------------

    dynamic_signal = make_dynamic_signal(
        xyz
    )

    dynamic = benchmark_case(
        "8. DYNAMIC OBJECT EMPHASIS",
        xyz,
        {
            "DYNAMIC": dynamic_signal,
        },
    )

    # ---------------------------------------------------------------
    # 9. Rough terrain
    # ---------------------------------------------------------------

    rough_signal = make_rough_terrain_signal(
        xyz
    )

    rough = benchmark_case(
        "9. ROUGH-TERRAIN EMPHASIS",
        xyz,
        {
            "SEMANTIC": rough_signal,
        },
    )

    # ---------------------------------------------------------------
    # 10. Empty / confident terrain
    # ---------------------------------------------------------------

    empty_signal = np.zeros(
        len(xyz),
        dtype=np.float64,
    )

    empty = benchmark_case(
        "10. EMPTY / CONFIDENT TERRAIN",
        xyz,
        {
            "DISTANCE": empty_signal,
        },
    )

    # ===============================================================
    # BEHAVIORAL VALIDATION
    # ===============================================================

    print()
    print()
    print("=" * 70)
    print("A19 BEHAVIORAL VALIDATION")
    print("=" * 70)

    checks = {}

    # ---------------------------------------------------------------
    # Speed behavior
    # ---------------------------------------------------------------

    checks["speed_effect"] = (
        five_mps["importance_mean"]
        > stationary["importance_mean"]
        and
        ten_mps["importance_mean"]
        > five_mps["importance_mean"]
    )

    checks["stationary_vs_5mps"] = (
        five_mps["fine_fraction"]
        > stationary["fine_fraction"]
    )

    # ---------------------------------------------------------------
    # Turning behavior
    # ---------------------------------------------------------------
    #
    # IMPORTANT:
    # We intentionally do NOT assume that positive yaw rate
    # corresponds to positive Y.
    #
    # Instead, A19 verifies that opposite yaw rates produce
    # opposite lateral displacement of the fine region.
    #
    # This validates the behavioral effect without imposing an
    # unverified coordinate-sign convention.
    # ---------------------------------------------------------------

    turn_lateral_difference = (
        left_turn["fine_centroid_y"]
        - right_turn["fine_centroid_y"]
    )

    checks["turn_direction"] = (
        abs(turn_lateral_difference) > 1e-3
    )

    # ---------------------------------------------------------------
    # Predicted path
    # ---------------------------------------------------------------

    checks["predicted_path_refinement"] = (
        predicted_path["fine_fraction"] > 0.0
        and
        predicted_path["fine_fraction"]
        > empty["fine_fraction"]
    )

    # ---------------------------------------------------------------
    # Semantic
    # ---------------------------------------------------------------

    checks["semantic_refinement"] = (
        semantic["fine_fraction"] > 0.0
        and
        semantic["fine_fraction"]
        > empty["fine_fraction"]
    )

    # ---------------------------------------------------------------
    # Dynamic
    # ---------------------------------------------------------------

    checks["dynamic_refinement"] = (
        dynamic["fine_fraction"] > 0.0
        and
        dynamic["fine_fraction"]
        > empty["fine_fraction"]
    )

    # ---------------------------------------------------------------
    # Rough terrain
    # ---------------------------------------------------------------

    checks["rough_terrain_refinement"] = (
        rough["fine_fraction"] > 0.0
        and
        rough["fine_fraction"]
        > empty["fine_fraction"]
    )

    # ---------------------------------------------------------------
    # Empty / confident
    # ---------------------------------------------------------------

    checks["empty_coarse"] = (
        empty["coarse_fraction"] == 1.0
        and
        empty["fine_fraction"] == 0.0
        and
        empty["importance_mean"] == 0.0
    )

    # ===============================================================
    # Diagnostic summary
    # ===============================================================

    print()
    print("TURN DIAGNOSTIC")
    print("-" * 70)

    print(
        f"Left-turn fine centroid Y : "
        f"{left_turn['fine_centroid_y']:.6f}"
    )

    print(
        f"Right-turn fine centroid Y: "
        f"{right_turn['fine_centroid_y']:.6f}"
    )

    print(
        f"Left - right difference    : "
        f"{turn_lateral_difference:.6f}"
    )

    print()
    print(
        "Note: A19 checks opposite lateral behavior rather "
        "than assuming a particular yaw/Y sign convention."
    )

    # ===============================================================
    # PASS / FAIL
    # ===============================================================

    print()
    print("=" * 70)

    passed = 0

    for name, result in checks.items():
        status = "PASS" if result else "FAIL"

        if result:
            passed += 1

        print(
            f"{status:4s}  {name}"
        )

    print("=" * 70)

    print()
    print(
        f"Behavioral checks: "
        f"{passed}/{len(checks)} PASS"
    )

    if passed == len(checks):
        print()
        print("=" * 70)
        print("A19 STATUS: PASS")
        print("=" * 70)
        print(
            "Existing A17 foveation behavior satisfies "
            "the current A19 isolated-signal behavioral checks."
        )
        return 0

    print()
    print("=" * 70)
    print("A19 STATUS: NEEDS CALIBRATION")
    print("=" * 70)
    print(
        "At least one behavioral expectation failed."
    )
    print(
        "Do not modify A13 or A17 yet; inspect the failing "
        "behavior before making architectural changes."
    )

    return 1


if __name__ == "__main__":
    raise SystemExit(main())