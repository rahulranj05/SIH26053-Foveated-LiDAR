"""
A39.3 — Live FoveaMap Demonstration

Runs a live end-to-end FoveaMap demonstration:

    Synthetic LiDAR stream
        |
        v
    A21 SensorFrame
        |
        v
    A31 Temporal FoveaMap Pipeline
        |
        +--> A30 semantic + dynamic integration
        |
        +--> A20 FoveaMap
        |       |
        |       +--> A17 foveation controller
        |       +--> A18.3 hierarchical leaf mapper
        |
        v
    A38 safety assessment
        |
        v
    A39.1 DashboardFrame
        |
        v
    A39.2 Visual Dashboard

This module is strictly a demonstration/orchestration layer.
It does not modify the frozen FoveaMap mapping algorithms.

The synthetic scene contains:
    - drivable terrain
    - static obstacles
    - vehicles
    - vulnerable users
    - a moving vehicle
    - changing LiDAR density
    - changing safety conditions

The demonstration deliberately exercises:

    NORMAL -> DEGRADED -> SAFETY -> NORMAL

so that the final SIH dashboard visibly demonstrates the
robustness and safety behaviour of FoveaMap.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mapping.foveamap_dashboard import (
    DashboardConfig,
    DashboardFrame,
    build_dashboard_frame,
)
from mapping.foveamap_visual_dashboard import (
    FoveaMapVisualDashboard,
    VisualDashboardConfig,
)
from mapping.kinematic_foveation import (
    VehicleState,
    kinematic_importance,
)
from mapping.safety_controller import (
    SafetyConfig,
    assess_safety,
)
from mapping.sensor_frame import SensorFrame
from mapping.signal_integration import (
    SignalIntegrationConfig,
)
from mapping.temporal_foveamap_pipeline import (
    TemporalFoveaMapFrameResult,
    TemporalFoveaMapProcessor,
)


# ---------------------------------------------------------------------------
# Canonical FoveaMap semantic ontology
# ---------------------------------------------------------------------------

SEMANTIC_DRIVABLE = 1
SEMANTIC_NON_DRIVABLE = 2
SEMANTIC_STATIC_OBSTACLE = 3
SEMANTIC_VEHICLE = 4
SEMANTIC_VULNERABLE_USER = 5


# ---------------------------------------------------------------------------
# Demo configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DemoConfig:
    """Configuration for the A39.3 live demonstration."""

    total_frames: int = 60
    frame_rate_hz: float = 10.0
    random_seed: int = 39

    normal_points: int = 2_400
    degraded_points: int = 1_500
    safety_points: int = 400

    safety_normal_start: int = 0
    safety_degraded_start: int = 20
    safety_critical_start: int = 40
    safety_recovery_start: int = 50

    save_dir: Path | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(
                self.total_frames,
                int,
            )
            or self.total_frames < 1
        ):
            raise ValueError(
                "total_frames must be an integer >= 1"
            )

        if (
            not np.isfinite(
                self.frame_rate_hz
            )
            or self.frame_rate_hz <= 0.0
        ):
            raise ValueError(
                "frame_rate_hz must be finite and > 0"
            )

        for name in (
            "normal_points",
            "degraded_points",
            "safety_points",
        ):
            value = getattr(
                self,
                name,
            )

            if (
                not isinstance(
                    value,
                    int,
                )
                or value < 1
            ):
                raise ValueError(
                    f"{name} must be an integer >= 1"
                )

        if not (
            self.safety_normal_start
            <= self.safety_degraded_start
            <= self.safety_critical_start
            <= self.safety_recovery_start
        ):
            raise ValueError(
                "safety phase boundaries must be ordered"
            )


# ---------------------------------------------------------------------------
# Demonstration phase control
# ---------------------------------------------------------------------------

def _phase_for_frame(
    frame_index: int,
    config: DemoConfig,
) -> str:
    """Return the demonstration phase for a frame."""

    if frame_index < config.safety_degraded_start:
        return "NORMAL"

    if frame_index < config.safety_critical_start:
        return "DEGRADED"

    if frame_index < config.safety_recovery_start:
        return "SAFETY"

    return "NORMAL"


def _point_count_for_phase(
    phase: str,
    config: DemoConfig,
) -> int:
    """Return the synthetic LiDAR point count for a phase."""

    if phase == "NORMAL":
        return config.normal_points

    if phase == "DEGRADED":
        return config.degraded_points

    return config.safety_points


# ---------------------------------------------------------------------------
# Synthetic LiDAR scene generation
# ---------------------------------------------------------------------------

def _sample_ground(
    rng: np.random.Generator,
    count: int,
) -> np.ndarray:
    """Generate a road-like ground surface."""

    x = rng.uniform(
        2.0,
        55.0,
        size=count,
    )

    y = rng.uniform(
        -9.0,
        9.0,
        size=count,
    )

    z = (
        0.03
        * np.sin(x / 5.0)
        + 0.02
        * np.cos(y / 2.0)
        + rng.normal(
            0.0,
            0.025,
            size=count,
        )
    )

    return np.column_stack(
        (
            x,
            y,
            z,
        )
    )


def _sample_cluster(
    rng: np.random.Generator,
    *,
    center_x: float,
    center_y: float,
    center_z: float,
    count: int,
    scale_x: float,
    scale_y: float,
    scale_z: float,
) -> np.ndarray:
    """Generate a compact synthetic object cluster."""

    return np.column_stack(
        (
            rng.normal(
                center_x,
                scale_x,
                size=count,
            ),
            rng.normal(
                center_y,
                scale_y,
                size=count,
            ),
            rng.normal(
                center_z,
                scale_z,
                size=count,
            ),
        )
    )


def _generate_scene(
    *,
    rng: np.random.Generator,
    frame_index: int,
    point_count: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Generate one synthetic perception frame.

    Returns
    -------
    xyz:
        LiDAR points with shape (N, 3).

    semantic_labels:
        Canonical FoveaMap semantic labels.

    dynamic_probability:
        Explicit dynamic probability for each point.
    """

    ground_count = int(
        point_count * 0.78
    )

    obstacle_count = (
        point_count
        - ground_count
    )

    ground = _sample_ground(
        rng,
        ground_count,
    )

    semantic_ground = np.full(
        ground_count,
        SEMANTIC_DRIVABLE,
        dtype=np.int64,
    )

    dynamic_ground = np.zeros(
        ground_count,
        dtype=np.float64,
    )

    static_count = max(
        1,
        int(
            obstacle_count * 0.30
        ),
    )

    vehicle_count = max(
        1,
        int(
            obstacle_count * 0.45
        ),
    )

    vulnerable_count = max(
        1,
        obstacle_count
        - static_count
        - vehicle_count,
    )

    # Static obstacle.
    static_obstacle = _sample_cluster(
        rng,
        center_x=18.0,
        center_y=-3.5,
        center_z=0.9,
        count=static_count,
        scale_x=0.8,
        scale_y=0.7,
        scale_z=0.7,
    )

    # Moving vehicle.
    moving_x = (
        10.0
        + 0.35 * frame_index
    )

    moving_vehicle = _sample_cluster(
        rng,
        center_x=moving_x,
        center_y=2.5,
        center_z=0.8,
        count=vehicle_count,
        scale_x=0.9,
        scale_y=0.7,
        scale_z=0.6,
    )

    # Vulnerable road user.
    vulnerable_user = _sample_cluster(
        rng,
        center_x=7.0,
        center_y=-1.5,
        center_z=0.85,
        count=vulnerable_count,
        scale_x=0.35,
        scale_y=0.25,
        scale_z=0.45,
    )

    xyz = np.vstack(
        (
            ground,
            static_obstacle,
            moving_vehicle,
            vulnerable_user,
        )
    )

    semantic_labels = np.concatenate(
        (
            semantic_ground,
            np.full(
                static_count,
                SEMANTIC_STATIC_OBSTACLE,
                dtype=np.int64,
            ),
            np.full(
                vehicle_count,
                SEMANTIC_VEHICLE,
                dtype=np.int64,
            ),
            np.full(
                vulnerable_count,
                SEMANTIC_VULNERABLE_USER,
                dtype=np.int64,
            ),
        )
    )

    dynamic_probability = np.concatenate(
        (
            dynamic_ground,
            np.zeros(
                static_count,
                dtype=np.float64,
            ),
            np.full(
                vehicle_count,
                0.90,
                dtype=np.float64,
            ),
            np.full(
                vulnerable_count,
                0.65,
                dtype=np.float64,
            ),
        )
    )

    permutation = rng.permutation(
        xyz.shape[0]
    )

    return (
        xyz[permutation],
        semantic_labels[permutation],
        dynamic_probability[permutation],
    )


# ---------------------------------------------------------------------------
# FoveaMap signal provider
# ---------------------------------------------------------------------------

def _base_signal_provider(
    frame: SensorFrame,
) -> dict[str, np.ndarray]:
    """
    Generate independent geometric and kinematic signals.

    A30/A31 supply semantic and dynamic information separately.
    """

    distances = np.linalg.norm(
        frame.xyz[:, :2],
        axis=1,
    )

    distance_importance = np.exp(
        -distances / 45.0
    )

    kinematic = kinematic_importance(
        frame.xyz,
        frame.vehicle_state,
    )

    return {
        "DISTANCE": np.clip(
            distance_importance,
            0.0,
            1.0,
        ),
        "KINEMATIC": np.clip(
            kinematic,
            0.0,
            1.0,
        ),
    }


# ---------------------------------------------------------------------------
# SensorFrame construction
# ---------------------------------------------------------------------------

def _make_sensor_frame(
    *,
    frame_index: int,
    timestamp: float,
    rng: np.random.Generator,
    config: DemoConfig,
) -> SensorFrame:
    """Create one A21 SensorFrame."""

    phase = _phase_for_frame(
        frame_index,
        config,
    )

    point_count = _point_count_for_phase(
        phase,
        config,
    )

    (
        xyz,
        semantic_labels,
        dynamic_probability,
    ) = _generate_scene(
        rng=rng,
        frame_index=frame_index,
        point_count=point_count,
    )

    vehicle_state = VehicleState(
        speed=(
            4.0
            + 2.0
            * np.sin(
                frame_index / 8.0
            )
        ),
        yaw_rate=(
            0.03
            * np.sin(
                frame_index / 6.0
            )
        ),
        heading=(
            0.08
            * np.sin(
                frame_index / 10.0
            )
        ),
    )

    # First frame deliberately lacks an explicit dynamic signal.
    # This exercises the temporal dynamic path in A30/A31.
    if frame_index == 0:
        dynamic_probability_value = None
    else:
        dynamic_probability_value = (
            dynamic_probability
        )

    return SensorFrame(
        xyz=xyz,
        vehicle_state=vehicle_state,
        timestamp=timestamp,
        frame_id=frame_index,
        semantic_labels=semantic_labels,
        dynamic_probability=dynamic_probability_value,
    )


# ---------------------------------------------------------------------------
# Safety configuration
# ---------------------------------------------------------------------------

def _build_safety_config() -> SafetyConfig:
    """
    Configure safety transitions for the demonstration.

    Point-count thresholds deliberately produce:

        NORMAL
        DEGRADED
        SAFETY

    Latency thresholds are intentionally high so machine-dependent
    runtime does not accidentally dominate the demonstration.
    """

    return SafetyConfig(
        min_points_degraded=2_000,
        min_points_safety=500,
        max_latency_ms_degraded=10_000.0,
        max_latency_ms_safety=20_000.0,
    )


# ---------------------------------------------------------------------------
# Single-frame orchestration
# ---------------------------------------------------------------------------

def process_demo_frame(
    *,
    processor: TemporalFoveaMapProcessor,
    frame: SensorFrame,
    previous_timestamp: float | None,
    safety_config: SafetyConfig,
    dashboard_config: DashboardConfig,
) -> tuple[
    DashboardFrame,
    TemporalFoveaMapFrameResult,
]:
    """
    Execute the complete A39.3 processing path for one frame.
    """

    result = processor.process_frame(
        frame
    )

    latency_ms = float(
        result.pipeline_result.timings_ms[
            "total_ms"
        ]
    )

    frame_gap_s = None

    if previous_timestamp is not None:
        frame_gap_s = (
            float(frame.timestamp)
            - float(previous_timestamp)
        )

    safety = assess_safety(
        frame.xyz,
        latency_ms=latency_ms,
        frame_gap_s=frame_gap_s,
        semantic_available=(
            frame.semantic_labels is not None
        ),
        dynamic_available=(
            (
                frame.dynamic_probability
                is not None
            )
            or (
                "dynamic"
                in result.signals
            )
        ),
        config=safety_config,
    )

    dashboard_frame = build_dashboard_frame(
        frame_id=frame.frame_id,
        timestamp=frame.timestamp,
        xyz=frame.xyz,
        foveation=(
            result.pipeline_result.foveation
        ),
        leaf_map=(
            result.pipeline_result.leaf_map
        ),
        safety_assessment=safety,
        latency_ms=latency_ms,
        config=dashboard_config,
    )

    return (
        dashboard_frame,
        result,
    )


# ---------------------------------------------------------------------------
# Complete live demonstration
# ---------------------------------------------------------------------------

def run_demo(
    *,
    config: DemoConfig | None = None,
    show: bool = True,
) -> list[DashboardFrame]:
    """
    Run the complete A39.3 live demonstration.

    Returns
    -------
    list[DashboardFrame]
        All dashboard frames generated during the run.
    """

    if config is None:
        config = DemoConfig()

    rng = np.random.default_rng(
        config.random_seed
    )

    processor = TemporalFoveaMapProcessor(
        _base_signal_provider,
        signal_config=SignalIntegrationConfig(
            prefer_supplied_dynamic_probability=True,
            compensate_ego_motion=True,
        ),
    )

    safety_config = _build_safety_config()

    dashboard_config = DashboardConfig(
        max_points_for_preview=25_000,
        max_preview_range_m=75.0,
    )

    visual_dashboard = (
        FoveaMapVisualDashboard(
            config=VisualDashboardConfig(
                point_size=2.0,
                history_size=100,
            )
        )
    )

    if config.save_dir is not None:
        config.save_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    dashboard_frames: list[
        DashboardFrame
    ] = []

    previous_timestamp: float | None = None

    target_period = (
        1.0
        / config.frame_rate_hz
    )

    if show:
        import matplotlib.pyplot as plt

        plt.ion()

    for frame_index in range(
        config.total_frames
    ):
        loop_start = time.perf_counter()

        timestamp = (
            frame_index
            * target_period
        )

        frame = _make_sensor_frame(
            frame_index=frame_index,
            timestamp=timestamp,
            rng=rng,
            config=config,
        )

        (
            dashboard_frame,
            pipeline_result,
        ) = process_demo_frame(
            processor=processor,
            frame=frame,
            previous_timestamp=previous_timestamp,
            safety_config=safety_config,
            dashboard_config=dashboard_config,
        )

        dashboard_frames.append(
            dashboard_frame
        )

        visual_dashboard.update(
            dashboard_frame
        )

        if show:
            visual_dashboard.render()

            import matplotlib.pyplot as plt

            elapsed = (
                time.perf_counter()
                - loop_start
            )

            plt.pause(
                max(
                    0.001,
                    target_period
                    - elapsed,
                )
            )

        if config.save_dir is not None:
            output_path = (
                config.save_dir
                / (
                    f"frame_"
                    f"{frame_index:04d}.png"
                )
            )

            visual_dashboard.save(
                str(output_path)
            )

        phase = _phase_for_frame(
            frame_index,
            config,
        )

        print(
            (
                f"FRAME {frame_index:03d} | "
                f"PHASE={phase:<9} | "
                f"POINTS={frame.num_points:5d} | "
                f"MODE={dashboard_frame.safety_mode:<9} | "
                f"LEAVES={dashboard_frame.leaf_count:5d} | "
                f"LATENCY="
                f"{dashboard_frame.processing_latency_ms:8.2f} ms | "
                f"FPS={dashboard_frame.fps:7.2f}"
            )
        )

        previous_timestamp = timestamp

        del pipeline_result

    if show:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()

    return dashboard_frames


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the A39.3 live FoveaMap demonstration."
        )
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=60,
        help=(
            "number of demonstration frames"
        ),
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help=(
            "target demonstration rate"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=39,
        help=(
            "deterministic random seed"
        ),
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help=(
            "run without opening matplotlib"
        ),
    )

    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help=(
            "optional directory for dashboard PNG frames"
        ),
    )

    return parser


def main() -> int:
    """CLI entry point."""

    parser = build_argument_parser()

    args = parser.parse_args()

    config = DemoConfig(
        total_frames=args.frames,
        frame_rate_hz=args.fps,
        random_seed=args.seed,
        save_dir=args.save_dir,
    )

    run_demo(
        config=config,
        show=not args.no_show,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )