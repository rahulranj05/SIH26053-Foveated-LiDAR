from dataclasses import dataclass
from typing import Any


@dataclass
class BackendOutput:
    """
    BackendOutput schema v1.0.

    This follows the frozen top-level contract specified
    in the SIH26053 backend handover.
    """

    schema_version: str
    frame_id: str
    timestamp: float
    vehicle: dict[str, Any]
    map: dict[str, Any]
    objects: list[Any]
    metrics: dict[str, Any]
    diagnostics: dict[str, Any]

    def __post_init__(self):
        if self.schema_version != "1.0":
            raise ValueError(
                f"Unsupported BackendOutput schema version: "
                f"{self.schema_version}"
            )

        if not isinstance(self.frame_id, str) or not self.frame_id:
            raise ValueError("frame_id must be a non-empty string")

        if not isinstance(self.timestamp, (int, float)):
            raise TypeError("timestamp must be numeric")

        if not isinstance(self.vehicle, dict):
            raise TypeError("vehicle must be a dict")

        if not isinstance(self.map, dict):
            raise TypeError("map must be a dict")

        if not isinstance(self.objects, list):
            raise TypeError("objects must be a list")

        if not isinstance(self.metrics, dict):
            raise TypeError("metrics must be a dict")

        if not isinstance(self.diagnostics, dict):
            raise TypeError("diagnostics must be a dict")