"""
A22 — SemanticKITTI Sensor Frame Adapter

Converts one SemanticKITTI Velodyne .bin scan into the canonical
FoveaMap SensorFrame representation.

A22 is an ingestion boundary only.

It does not modify:
    - A13 kinematic foveation
    - A17 foveation controller
    - A18/A18.3 hierarchical mapping
    - A20 pipeline
    - A21 SensorFrame
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mapping.kinematic_foveation import VehicleState
from mapping.sensor_frame import SensorFrame


def load_semantickitti_scan(path: str | Path) -> np.ndarray:
    """
    Load one SemanticKITTI Velodyne scan.

    SemanticKITTI stores each point as four float32 values:

        x, y, z, intensity

    The returned array contains only XYZ coordinates.

    Parameters
    ----------
    path:
        Path to the SemanticKITTI .bin scan.

    Returns
    -------
    np.ndarray
        XYZ coordinates with shape (N, 3), dtype float64.

    Raises
    ------
    FileNotFoundError
        If the scan does not exist.

    ValueError
        If the file does not contain groups of four float32 values.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"SemanticKITTI scan not found: {path}")

    if not path.is_file():
        raise ValueError(f"SemanticKITTI scan path is not a file: {path}")

    raw = np.fromfile(path, dtype=np.float32)

    if raw.size % 4 != 0:
        raise ValueError(
            f"Invalid SemanticKITTI scan: {raw.size} float values "
            "is not divisible by 4."
        )

    points = raw.reshape(-1, 4)

    xyz = np.asarray(points[:, :3], dtype=np.float64)

    if not np.all(np.isfinite(xyz)):
        raise ValueError(
            "SemanticKITTI scan contains non-finite XYZ coordinates"
        )

    return xyz


def load_semantickitti_frame(
    path: str | Path,
    *,
    timestamp: float = 0.0,
    frame_id: object = "semantickitti",
    vehicle_state: VehicleState | None = None,
) -> SensorFrame:
    """
    Load one SemanticKITTI scan as a canonical SensorFrame.

    Semantic labels and dynamic probabilities are intentionally absent
    because this basic adapter only consumes the Velodyne .bin file.

    Parameters
    ----------
    path:
        Path to the SemanticKITTI .bin scan.

    timestamp:
        Timestamp assigned to this frame.

    frame_id:
        Source-specific frame identifier.

    vehicle_state:
        Optional ego-vehicle state. If omitted, a stationary default
        VehicleState is used.

    Returns
    -------
    SensorFrame
        Canonical FoveaMap sensor frame.
    """
    xyz = load_semantickitti_scan(path)

    if vehicle_state is None:
        vehicle_state = VehicleState(
            speed=0.0,
            yaw_rate=0.0,
            heading=0.0,
        )

    return SensorFrame(
        xyz=xyz,
        vehicle_state=vehicle_state,
        timestamp=timestamp,
        frame_id=frame_id,
    )