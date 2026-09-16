from __future__ import annotations

import numpy as np


RELLIS_P1 = 0.0002594033721834421
RELLIS_P99 = 0.00869764294475317


def normalize_intensity(dataset: str, intensity_raw: np.ndarray) -> np.ndarray:
    """
    Frozen S7-C intensity normalization.

    SemanticKITTI: identity
    RELLIS-3D: clip((x-p1)/(p99-p1), 0, 1)
    SemanticSTF: x / 255
    nuScenes: x / 255
    """
    x = np.asarray(intensity_raw, dtype=np.float32)

    if dataset == "SemanticKITTI":
        out = x

    elif dataset == "RELLIS-3D":
        out = (x - RELLIS_P1) / (RELLIS_P99 - RELLIS_P1)
        out = np.clip(out, 0.0, 1.0)

    elif dataset in ("SemanticSTF", "nuScenes"):
        out = x / 255.0

    else:
        raise KeyError(f"Unsupported dataset: {dataset}")

    return out.astype(np.float32, copy=False)
