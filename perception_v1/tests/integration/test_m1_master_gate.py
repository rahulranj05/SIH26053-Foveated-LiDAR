import numpy as np

from perception_v1.src.data.canonical import build_canonical_point_frame
from perception_v1.src.data.normalization import normalize_intensity
from perception_v1.src.ontology.classes import NUM_TRAINABLE_CLASSES
from perception_v1.src.ontology.mappings import NATIVE_TO_UNIFIED


DATASETS = [
    "SemanticKITTI",
    "RELLIS-3D",
    "SemanticSTF",
    "nuScenes",
]


def check_mapping_contract():
    assert set(NATIVE_TO_UNIFIED.keys()) == set(DATASETS)

    for dataset in DATASETS:
        mapping = NATIVE_TO_UNIFIED[dataset]
        assert len(mapping) > 0

        for native_id, unified_id in mapping.items():
            assert isinstance(native_id, int)
            assert isinstance(unified_id, int)
            assert 0 <= unified_id <= NUM_TRAINABLE_CLASSES


def check_canonical_contract():
    for dataset in DATASETS:
        mapping = NATIVE_TO_UNIFIED[dataset]
        native_label = next(iter(mapping.keys()))

        xyz = np.array([
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
            [np.nan, 1.0, 2.0],
            [-2.0, 1.0, 0.5],
        ], dtype=np.float32)

        if dataset in ("SemanticSTF", "nuScenes"):
            intensity = np.array(
                [255.0, 128.0, 64.0, 0.0],
                dtype=np.float32,
            )
        elif dataset == "RELLIS-3D":
            intensity = np.array(
                [0.001, 0.002, 0.003, 0.004],
                dtype=np.float32,
            )
        else:
            intensity = np.array(
                [0.2, 0.3, 0.4, 0.5],
                dtype=np.float32,
            )

        labels = np.full(
            4,
            native_label,
            dtype=np.int64,
        )

        frame = build_canonical_point_frame(
            dataset_id=dataset,
            xyz=xyz,
            intensity_raw=intensity,
            native_labels=labels,
        )

        assert len(frame.xyz) == 2
        assert frame.raw_source_index.tolist() == [0, 3]
        assert frame.source_point_id.tolist() == [0, 1]
        assert frame.validity_mask.tolist() == [True, True]

        assert np.all(np.isfinite(frame.xyz))
        assert not np.any(np.all(frame.xyz == 0.0, axis=1))

        n = len(frame.xyz)

        assert frame.intensity_raw.shape == (n,)
        assert frame.intensity_normalized.shape == (n,)
        assert frame.range.shape == (n,)
        assert frame.azimuth.shape == (n,)
        assert frame.elevation.shape == (n,)
        assert frame.unified_semantic_target.shape == (n,)

        assert np.all(np.isfinite(frame.range))
        assert np.all(np.isfinite(frame.azimuth))
        assert np.all(np.isfinite(frame.elevation))
        assert np.all(np.isfinite(frame.intensity_normalized))
        assert np.all(frame.range > 0)

        assert np.all(
            (frame.unified_semantic_target >= 0)
            & (frame.unified_semantic_target <= 24)
        )


def check_normalization_contract():
    x = np.array([0.0, 0.5, 0.99], dtype=np.float32)
    assert np.allclose(
        normalize_intensity("SemanticKITTI", x),
        x,
    )

    x = np.array([0.0, 127.5, 255.0], dtype=np.float32)

    assert np.allclose(
        normalize_intensity("SemanticSTF", x),
        [0.0, 0.5, 1.0],
    )

    assert np.allclose(
        normalize_intensity("nuScenes", x),
        [0.0, 0.5, 1.0],
    )

    x = np.array([
        0.0002594033721834421,
        0.00869764294475317,
    ], dtype=np.float32)

    assert np.allclose(
        normalize_intensity("RELLIS-3D", x),
        [0.0, 1.0],
        atol=1e-6,
    )


def main():
    print("M1 MASTER GATE")
    print("=" * 50)

    check_mapping_contract()
    print("[PASS] Frozen mapping contract")

    check_canonical_contract()
    print("[PASS] Canonical cross-dataset contract")

    check_normalization_contract()
    print("[PASS] Frozen intensity normalization contract")

    print("=" * 50)
    print("M1 MASTER GATE: PASS")


if __name__ == "__main__":
    main()
