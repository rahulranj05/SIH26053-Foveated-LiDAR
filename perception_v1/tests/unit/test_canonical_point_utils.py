import numpy as np

from perception_v1.src.data.canonical import build_canonical_point_frame
from perception_v1.src.data.normalization import normalize_intensity
from perception_v1.src.geometry.features import (
    compute_spherical_features,
    compute_validity_mask,
)


def test_validity_mask():
    xyz = np.array([
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0],
        [np.nan, 1.0, 2.0],
    ], dtype=np.float32)

    mask = compute_validity_mask(xyz)

    assert mask.tolist() == [True, False, False]


def test_spherical_features():
    xyz = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    r, az, el = compute_spherical_features(xyz)

    assert np.allclose(r, [1.0, 1.0, 1.0])
    assert np.isclose(az[0], 0.0)
    assert np.isclose(az[1], np.pi / 2)
    assert np.isclose(el[2], np.pi / 2)


def test_intensity_normalization():
    sk = normalize_intensity(
        "SemanticKITTI",
        np.array([0.0, 0.5, 1.0], dtype=np.float32),
    )
    assert np.allclose(sk, [0.0, 0.5, 1.0])

    stf = normalize_intensity(
        "SemanticSTF",
        np.array([0.0, 255.0], dtype=np.float32),
    )
    assert np.allclose(stf, [0.0, 1.0])

    nusc = normalize_intensity(
        "nuScenes",
        np.array([0.0, 255.0], dtype=np.float32),
    )
    assert np.allclose(nusc, [0.0, 1.0])


def test_canonical_frame():
    xyz = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ], dtype=np.float32)

    intensity = np.array([0.5, 0.2], dtype=np.float32)

    frame = build_canonical_point_frame(
        "SemanticKITTI",
        xyz,
        intensity,
    )

    assert frame.xyz.shape == (2, 3)
    assert frame.validity_mask.tolist() == [True, False]
    assert frame.source_point_id.tolist() == [0, 1]
    assert np.allclose(frame.intensity_normalized, intensity)


if __name__ == "__main__":
    test_validity_mask()
    test_spherical_features()
    test_intensity_normalization()
    test_canonical_frame()

    print("CANONICAL POINT UTILITIES TEST: PASS")
