from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VehicleState:
    """
    Vehicle state used by the kinematic foveation controller.
    """

    speed: float
    yaw_rate: float
    heading: float

    def __post_init__(self):
        if not np.isfinite(self.speed):
            raise ValueError("speed must be finite.")

        if not np.isfinite(self.yaw_rate):
            raise ValueError("yaw_rate must be finite.")

        if not np.isfinite(self.heading):
            raise ValueError("heading must be finite.")

        if self.speed < 0.0:
            raise ValueError("speed must be non-negative.")


@dataclass(frozen=True)
class KinematicFoveationConfig:
    """
    Configuration for the motion-aware foveation controller.
    """

    minimum_forward_range: float = 10.0
    maximum_forward_range: float = 50.0

    reference_speed: float = 10.0

    lateral_half_width: float = 5.0

    turn_gain: float = 8.0

    minimum_importance: float = 0.0
    maximum_importance: float = 1.0

    def __post_init__(self):
        if self.minimum_forward_range < 0.0:
            raise ValueError(
                "minimum_forward_range must be non-negative."
            )

        if (
            self.maximum_forward_range
            <= self.minimum_forward_range
        ):
            raise ValueError(
                "maximum_forward_range must be greater "
                "than minimum_forward_range."
            )

        if self.reference_speed <= 0.0:
            raise ValueError(
                "reference_speed must be positive."
            )

        if self.lateral_half_width <= 0.0:
            raise ValueError(
                "lateral_half_width must be positive."
            )

        if self.turn_gain < 0.0:
            raise ValueError(
                "turn_gain must be non-negative."
            )


def normalize_speed(
    speed: float,
    reference_speed: float,
) -> float:
    """
    Convert vehicle speed into a bounded [0, 1] value.
    """

    if not np.isfinite(speed):
        raise ValueError("speed must be finite.")

    if speed < 0.0:
        raise ValueError("speed must be non-negative.")

    if reference_speed <= 0.0:
        raise ValueError(
            "reference_speed must be positive."
        )

    return float(
        np.clip(
            speed / reference_speed,
            0.0,
            1.0,
        )
    )


def forward_range_for_speed(
    speed: float,
    config: KinematicFoveationConfig,
) -> float:
    """
    Determine how far the forward fovea should extend.

    At zero speed, the controller preserves the minimum
    forward range.

    As speed increases, the fine region extends toward
    maximum_forward_range.
    """

    speed_factor = normalize_speed(
        speed,
        config.reference_speed,
    )

    return (
        config.minimum_forward_range
        + speed_factor
        * (
            config.maximum_forward_range
            - config.minimum_forward_range
        )
    )


def transform_to_vehicle_frame(
    x: np.ndarray,
    y: np.ndarray,
    heading: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform world XY coordinates into the vehicle frame.

    Vehicle-frame convention:

        forward = +X
        left    = +Y
    """

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    if x.shape != y.shape:
        raise ValueError(
            "x and y must have the same shape."
        )

    if not np.all(np.isfinite(x)):
        raise ValueError(
            "x must contain only finite values."
        )

    if not np.all(np.isfinite(y)):
        raise ValueError(
            "y must contain only finite values."
        )

    if not np.isfinite(heading):
        raise ValueError(
            "heading must be finite."
        )

    cos_heading = np.cos(heading)
    sin_heading = np.sin(heading)

    forward_x = (
        cos_heading * x
        + sin_heading * y
    )

    lateral_y = (
        -sin_heading * x
        + cos_heading * y
    )

    return (
        forward_x,
        lateral_y,
    )


def yaw_shift_for_distance(
    forward_distance: np.ndarray,
    yaw_rate: float,
    config: KinematicFoveationConfig,
) -> np.ndarray:
    """
    Estimate lateral foveation shift caused by turning.

    This is a lightweight geometric approximation.

    The farther ahead a point lies, the more the turning
    direction influences the desired foveation corridor.
    """

    forward_distance = np.asarray(
        forward_distance,
        dtype=np.float64,
    )

    if not np.all(
        np.isfinite(forward_distance)
    ):
        raise ValueError(
            "forward_distance must contain "
            "only finite values."
        )

    if not np.isfinite(yaw_rate):
        raise ValueError(
            "yaw_rate must be finite."
        )

    shift = (
        yaw_rate
        * forward_distance
        * config.turn_gain
    )

    return shift


def kinematic_importance(
    xyz: np.ndarray,
    vehicle_state: VehicleState,
    config: KinematicFoveationConfig | None = None,
) -> np.ndarray:
    """
    Compute motion-aware foveation importance.

    Importance is in [0, 1].

    High importance means that the location should receive
    finer spatial resolution.

    The controller considers:

        1. distance ahead of the vehicle
        2. vehicle speed
        3. vehicle heading
        4. yaw-rate-induced lateral shift
        5. lateral distance from the predicted corridor
    """

    if config is None:
        config = KinematicFoveationConfig()

    xyz = np.asarray(
        xyz,
        dtype=np.float64,
    )

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(
            "xyz must have shape (N, 3)."
        )

    if not np.all(np.isfinite(xyz)):
        raise ValueError(
            "xyz must contain only finite values."
        )

    if len(xyz) == 0:
        return np.empty(
            0,
            dtype=np.float64,
        )

    forward, lateral = transform_to_vehicle_frame(
        xyz[:, 0],
        xyz[:, 1],
        vehicle_state.heading,
    )

    forward_range = forward_range_for_speed(
        vehicle_state.speed,
        config,
    )

    positive_forward = np.maximum(
        forward,
        0.0,
    )

    turn_shift = yaw_shift_for_distance(
        positive_forward,
        vehicle_state.yaw_rate,
        config,
    )

    corridor_distance = np.abs(
        lateral - turn_shift
    )

    # Normalize forward distance.
    #
    # The epsilon prevents the exact outer boundary
    # from producing zero importance solely because:
    #
    #     1 - forward / forward_range
    #
    # becomes exactly zero there.
    #
    # The boundary remains active while points beyond
    # the range are still explicitly forced to zero.
    epsilon = 1e-6

    forward_factor = np.clip(
        forward / forward_range,
        0.0,
        1.0 - epsilon,
    )

    forward_importance = (
        1.0
        - forward_factor
    )

    lateral_importance = np.exp(
        -(
            corridor_distance
            / config.lateral_half_width
        ) ** 2
    )

    speed_factor = normalize_speed(
        vehicle_state.speed,
        config.reference_speed,
    )

    speed_importance = (
        0.5
        + 0.5 * speed_factor
    )

    importance = (
        forward_importance
        * lateral_importance
        * speed_importance
    )

    # Behind the vehicle is never part of the forward fovea.
    importance[
        forward <= 0.0
    ] = 0.0

    # Points strictly beyond the active forward range
    # are outside the foveation region.
    importance[
        forward > forward_range
    ] = 0.0

    importance = np.clip(
        importance,
        config.minimum_importance,
        config.maximum_importance,
    )

    return importance


def resolution_from_kinematic_importance(
    importance: np.ndarray,
) -> np.ndarray:
    """
    Convert kinematic importance into alignment-safe
    hierarchy levels.

    Mapping:

        importance >= 0.75 -> 5 cm
        importance >= 0.50 -> 10 cm
        importance >= 0.25 -> 20 cm
        otherwise           -> 40 cm
    """

    importance = np.asarray(
        importance,
        dtype=np.float64,
    )

    if not np.all(
        np.isfinite(importance)
    ):
        raise ValueError(
            "importance must contain only finite values."
        )

    if np.any(
        (importance < 0.0)
        | (importance > 1.0)
    ):
        raise ValueError(
            "importance must be in [0, 1]."
        )

    resolution = np.full(
        importance.shape,
        0.40,
        dtype=np.float64,
    )

    resolution[
        importance >= 0.25
    ] = 0.20

    resolution[
        importance >= 0.50
    ] = 0.10

    resolution[
        importance >= 0.75
    ] = 0.05

    return resolution


def kinematic_resolution(
    xyz: np.ndarray,
    vehicle_state: VehicleState,
    config: KinematicFoveationConfig | None = None,
) -> np.ndarray:
    """
    Compute the final kinematic resolution for each point.
    """

    importance = kinematic_importance(
        xyz,
        vehicle_state,
        config,
    )

    return resolution_from_kinematic_importance(
        importance
    )