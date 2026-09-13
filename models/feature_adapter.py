from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import yaml


SUPPORTED_MODES = {
    "constant",
    "robust_intensity",
}


class FeatureAdapter:
    """
    Converts raw point attributes into model input features.

    Modes
    -----
    constant:
        Every point receives feature [1.0].

    robust_intensity:
        Dataset-specific TRAIN-derived robust min-max scaling:

            x = clip(x, lower, upper)
            x = (x - lower) / (upper - lower)

        Output lies in [0, 1].

    XYZ is NOT returned as a feature here.
    XYZ is used separately for sparse voxel coordinates.
    """

    def __init__(
        self,
        mode: str,
        normalization_config: str | Path | None = None,
    ) -> None:

        mode = str(mode)

        if mode not in SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported feature mode: {mode}. "
                f"Supported: {sorted(SUPPORTED_MODES)}"
            )

        self.mode = mode
        self.normalization: Dict[str, Dict[str, float]] = {}

        if self.mode == "robust_intensity":

            if normalization_config is None:
                raise ValueError(
                    "normalization_config is required "
                    "for robust_intensity mode."
                )

            path = Path(normalization_config)

            if not path.is_file():
                raise FileNotFoundError(
                    f"Normalization config not found: {path}"
                )

            with path.open(
                "r",
                encoding="utf-8",
            ) as f:
                cfg = yaml.safe_load(f)

            datasets = cfg.get("datasets")

            if not isinstance(datasets, dict):
                raise RuntimeError(
                    "Normalization config must contain "
                    "a 'datasets' mapping."
                )

            for dataset_name, values in datasets.items():

                lower = float(values["lower"])
                upper = float(values["upper"])

                if not np.isfinite(lower):
                    raise ValueError(
                        f"{dataset_name}: lower is not finite"
                    )

                if not np.isfinite(upper):
                    raise ValueError(
                        f"{dataset_name}: upper is not finite"
                    )

                if upper <= lower:
                    raise ValueError(
                        f"{dataset_name}: upper ({upper}) "
                        f"must be greater than lower ({lower})"
                    )

                self.normalization[str(dataset_name)] = {
                    "lower": lower,
                    "upper": upper,
                }

    def __call__(
        self,
        intensity: np.ndarray,
        dataset_name: str,
    ) -> np.ndarray:

        intensity = np.asarray(
            intensity,
            dtype=np.float32,
        )

        if intensity.ndim != 1:
            raise ValueError(
                f"Intensity must be 1-D, got shape "
                f"{intensity.shape}"
            )

        if not np.isfinite(intensity).all():
            raise ValueError(
                f"{dataset_name}: intensity contains NaN/Inf"
            )

        if self.mode == "constant":
            return np.ones(
                (intensity.shape[0], 1),
                dtype=np.float32,
            )

        if dataset_name not in self.normalization:
            raise KeyError(
                f"No normalization parameters for "
                f"dataset: {dataset_name}"
            )

        lower = self.normalization[
            dataset_name
        ]["lower"]

        upper = self.normalization[
            dataset_name
        ]["upper"]

        clipped = np.clip(
            intensity,
            lower,
            upper,
        )

        scaled = (
            clipped - lower
        ) / (
            upper - lower
        )

        scaled = scaled.astype(
            np.float32,
            copy=False,
        )

        if not np.isfinite(scaled).all():
            raise RuntimeError(
                f"{dataset_name}: normalized intensity "
                f"contains NaN/Inf"
            )

        if (
            scaled.min(initial=0.0) < -1e-6
            or scaled.max(initial=0.0) > 1.0 + 1e-6
        ):
            raise RuntimeError(
                f"{dataset_name}: normalized intensity "
                f"outside expected [0,1] range"
            )

        return scaled[:, None]