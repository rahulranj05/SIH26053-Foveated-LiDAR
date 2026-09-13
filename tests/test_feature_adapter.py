from pathlib import Path

import numpy as np

from models.feature_adapter import FeatureAdapter


REPO_ROOT = Path(__file__).resolve().parents[1]

CONFIG = (
    REPO_ROOT
    / "configs"
    / "intensity_normalization.yaml"
)


def test_constant_mode():

    adapter = FeatureAdapter(
        mode="constant",
    )

    x = np.array(
        [0.0, 0.2, 0.9],
        dtype=np.float32,
    )

    out = adapter(
        x,
        "semantic_kitti",
    )

    assert out.shape == (3, 1)
    assert out.dtype == np.float32
    assert np.allclose(out, 1.0)


def test_robust_intensity_semantic_kitti():

    adapter = FeatureAdapter(
        mode="robust_intensity",
        normalization_config=CONFIG,
    )

    x = np.array(
        [
            0.0,
            0.32,
            0.64,
            1.0,
        ],
        dtype=np.float32,
    )

    out = adapter(
        x,
        "semantic_kitti",
    )

    assert out.shape == (4, 1)

    assert np.allclose(
        out[:, 0],
        [
            0.0,
            0.5,
            1.0,
            1.0,
        ],
        atol=1e-5,
    )


def test_robust_intensity_semantic_stf():

    adapter = FeatureAdapter(
        mode="robust_intensity",
        normalization_config=CONFIG,
    )

    lower = 11.40000057
    upper = 255.0
    middle = (
        lower
        + upper
    ) / 2.0

    x = np.array(
        [
            1.0,
            lower,
            middle,
            upper,
            300.0,
        ],
        dtype=np.float32,
    )

    out = adapter(
        x,
        "semantic_stf",
    )

    assert np.allclose(
        out[:, 0],
        [
            0.0,
            0.0,
            0.5,
            1.0,
            1.0,
        ],
        atol=1e-5,
    )


def test_unknown_dataset_raises():

    adapter = FeatureAdapter(
        mode="robust_intensity",
        normalization_config=CONFIG,
    )

    try:
        adapter(
            np.array(
                [1.0],
                dtype=np.float32,
            ),
            "unknown_dataset",
        )

    except KeyError:
        return

    raise AssertionError(
        "Expected KeyError for unknown dataset"
    )