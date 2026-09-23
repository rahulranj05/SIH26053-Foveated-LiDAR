import json
import math
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np

from backend.contracts.backend_output import BackendOutput


class FrontendOutputAdapter:
    """
    Converts BackendOutput into a frontend-safe JSON payload.

    Contract:
        BackendOutput -> JSON-safe Python dictionary -> JSON string
    """

    EXPECTED_SCHEMA_VERSION = "1.0"

    def serialize(self, output: BackendOutput) -> str:
        if not isinstance(output, BackendOutput):
            raise TypeError("output must be a BackendOutput")

        if output.schema_version != self.EXPECTED_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported BackendOutput schema version: "
                f"{output.schema_version}"
            )

        payload = asdict(output)

        payload = self._convert(payload)

        # Final validation: JSON must reject NaN/Infinity.
        return json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
        )

    def _convert(self, value: Any) -> Any:
        """
        Recursively convert NumPy/dataclass values into
        JSON-compatible Python values.
        """

        if is_dataclass(value) and not isinstance(value, type):
            return self._convert(asdict(value))

        if isinstance(value, np.ndarray):
            return self._convert(value.tolist())

        if isinstance(value, np.generic):
            return self._convert(value.item())

        if isinstance(value, dict):
            return {
                str(key): self._convert(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple)):
            return [self._convert(item) for item in value]

        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(
                    "BackendOutput contains NaN or Infinity"
                )
            return value

        if isinstance(value, int):
            return value

        if isinstance(value, (str, bool)) or value is None:
            return value

        raise TypeError(
            f"Unsupported value type in BackendOutput: "
            f"{type(value).__name__}"
        )