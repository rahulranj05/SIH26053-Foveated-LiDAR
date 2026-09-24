from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AugmentationConfig:
    rotation_min_rad: float = -np.pi
    rotation_max_rad: float = np.pi
    reflection_probability: float = 0.5
    scale_min: float = 0.95
    scale_max: float = 1.05
    translation_min_m: float = -0.2
    translation_max_m: float = 0.2


@dataclass(frozen=True)
class GeometricTransform:
    rotation_rad: float
    reflect_x: bool
    scale: float
    translation: np.ndarray


def sample_geometric_transform(
    rng: np.random.Generator,
    config: AugmentationConfig = AugmentationConfig(),
) -> GeometricTransform:
    rotation_rad = float(
        rng.uniform(
            config.rotation_min_rad,
            config.rotation_max_rad,
        )
    )

    reflect_x = bool(
        rng.random() < config.reflection_probability
    )

    scale = float(
        rng.uniform(
            config.scale_min,
            config.scale_max,
        )
    )

    translation = rng.uniform(
        config.translation_min_m,
        config.translation_max_m,
        size=3,
    ).astype(np.float32)

    return GeometricTransform(
        rotation_rad=rotation_rad,
        reflect_x=reflect_x,
        scale=scale,
        translation=translation,
    )


def apply_geometric_transform(
    xyz: np.ndarray,
    transform: GeometricTransform,
) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(
            f"xyz must have shape (N, 3), got {xyz.shape}"
        )

    if not np.all(np.isfinite(xyz)):
        raise ValueError(
            "Canonical XYZ must be finite before augmentation"
        )

    result = xyz.copy()

    # Frozen X-axis reflection.
    if transform.reflect_x:
        result[:, 0] *= -1.0

    # Rotation around sensor Z-axis.
    c = np.float32(np.cos(transform.rotation_rad))
    s = np.float32(np.sin(transform.rotation_rad))

    x = result[:, 0].copy()
    y = result[:, 1].copy()

    result[:, 0] = c * x - s * y
    result[:, 1] = s * x + c * y

    # Uniform geometric scaling.
    result *= np.float32(transform.scale)

    # XYZ translation.
    translation = np.asarray(
        transform.translation,
        dtype=np.float32,
    )

    if translation.shape != (3,):
        raise ValueError(
            "translation must have shape (3,)"
        )

    if not np.all(np.isfinite(translation)):
        raise ValueError(
            "translation must contain finite values"
        )

    result += translation

    return result.astype(np.float32, copy=False)


def augment_xyz(
    xyz: np.ndarray,
    *,
    training: bool,
    rng: np.random.Generator | None = None,
    config: AugmentationConfig = AugmentationConfig(),
) -> tuple[np.ndarray, GeometricTransform | None]:
    xyz = np.asarray(xyz, dtype=np.float32)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(
            f"xyz must have shape (N, 3), got {xyz.shape}"
        )

    if not training:
        # Validation/test/inference: absolutely no augmentation.
        return xyz.copy(), None

    if rng is None:
        raise ValueError(
            "Training augmentation requires an explicit RNG"
        )

    transform = sample_geometric_transform(
        rng,
        config,
    )

    augmented = apply_geometric_transform(
        xyz,
        transform,
    )

    return augmented, transform