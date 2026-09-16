from __future__ import annotations

import numpy as np


def compute_validity_mask(xyz: np.ndarray) -> np.ndarray:
    """
    S7-B frozen validity rule:
    valid = finite XYZ AND XYZ != (0, 0, 0)
    """
    xyz = np.asarray(xyz)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape (N, 3), got {xyz.shape}")

    finite = np.isfinite(xyz).all(axis=1)
    nonzero = np.any(xyz != 0.0, axis=1)

    return finite & nonzero


def compute_spherical_features(xyz: np.ndarray):
    """
    Compute geometric range, azimuth, elevation from XYZ.
    """
    xyz = np.asarray(xyz, dtype=np.float32)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape (N, 3), got {xyz.shape}")

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    range_m = np.sqrt(x * x + y * y + z * z)

    azimuth = np.arctan2(y, x)

    horizontal = np.sqrt(x * x + y * y)
    elevation = np.arctan2(z, horizontal)

    return (
        range_m.astype(np.float32),
        azimuth.astype(np.float32),
        elevation.astype(np.float32),
    )
