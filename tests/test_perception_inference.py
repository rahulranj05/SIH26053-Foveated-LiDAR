import numpy as np
import torch

from perception.inference import F1InferenceEngine
from perception.sample_pipeline import build_unified_sample


def _make_sample():
    xyz = np.array(
        [
            [10.0, 0.0, 0.0],
            [10.0, 1.0, 0.0],
            [15.0, -2.0, 0.5],
            [20.0, 3.0, -1.0],
        ],
        dtype=np.float32,
    )

    intensity = np.array(
        [0.2, 0.4, 0.6, 0.8],
        dtype=np.float32,
    )

    source_point_id = np.arange(len(xyz), dtype=np.int64)

    return build_unified_sample(
        dataset_id="SemanticKITTI",
        split_role="inference",
        xyz=xyz,
        intensity=intensity,
        source_point_id=source_point_id,
    )


def test_build_sparse_batch_contract():
    sample = _make_sample()

    batch = F1InferenceEngine._build_sparse_batch(sample)

    batch.validate()

    assert batch.coordinates.dtype == torch.int32
    assert batch.coordinates.shape == (4, 4)

    # Single-frame inference uses batch index zero.
    assert torch.all(batch.coordinates[:, 0] == 0)

    assert batch.features.dtype == torch.float32
    assert batch.features.shape == (4, 7)

    assert batch.point_to_voxel_inverse.dtype == torch.int64
    assert batch.point_to_voxel_inverse.shape == (4,)


def test_sparse_coordinates_preserve_voxel_coordinates():
    sample = _make_sample()

    batch = F1InferenceEngine._build_sparse_batch(sample)

    expected = torch.from_numpy(
        sample.spvcnn.voxel_coordinates.astype(
            np.int32,
            copy=False,
        )
    )

    assert torch.equal(batch.coordinates[:, 1:], expected)


def test_sparse_features_preserve_preprocessed_features():
    sample = _make_sample()

    batch = F1InferenceEngine._build_sparse_batch(sample)

    expected = torch.from_numpy(
        sample.spvcnn.voxel_features.astype(
            np.float32,
            copy=False,
        )
    )

    assert torch.equal(batch.features, expected)


def test_point_to_voxel_inverse_is_preserved():
    sample = _make_sample()

    batch = F1InferenceEngine._build_sparse_batch(sample)

    expected = torch.from_numpy(
        sample.spvcnn.point_to_voxel_inverse.astype(
            np.int64,
            copy=False,
        )
    )

    assert torch.equal(
        batch.point_to_voxel_inverse,
        expected,
    )

def test_perception_result_accepts_valid_24_class_contract():
    from perception.inference import PerceptionResult

    n = 4

    result = PerceptionResult(
        xyz=np.zeros((n, 3), dtype=np.float32),
        unified_labels=np.array([1, 2, 23, 24], dtype=np.int64),
        confidence=np.array([0.25, 0.50, 0.75, 1.0], dtype=np.float32),
        model_predictions=np.array([0, 1, 22, 23], dtype=np.int64),
        point_logits=np.zeros((n, 24), dtype=np.float32),
    )

    result.validate()


def test_model_prediction_to_unified_label_contract():
    model_predictions = np.arange(24, dtype=np.int64)

    unified_labels = model_predictions + 1

    assert np.array_equal(
        unified_labels,
        np.arange(1, 25, dtype=np.int64),
    )


def test_perception_result_rejects_unified_label_zero():
    import pytest
    from perception.inference import PerceptionResult

    result = PerceptionResult(
        xyz=np.zeros((1, 3), dtype=np.float32),
        unified_labels=np.array([0], dtype=np.int64),
        confidence=np.array([0.9], dtype=np.float32),
        model_predictions=np.array([0], dtype=np.int64),
        point_logits=np.zeros((1, 24), dtype=np.float32),
    )

    with pytest.raises(
        ValueError,
        match="unified labels outside 1..24",
    ):
        result.validate()


def test_perception_result_rejects_model_prediction_24():
    import pytest
    from perception.inference import PerceptionResult

    result = PerceptionResult(
        xyz=np.zeros((1, 3), dtype=np.float32),
        unified_labels=np.array([24], dtype=np.int64),
        confidence=np.array([0.9], dtype=np.float32),
        model_predictions=np.array([24], dtype=np.int64),
        point_logits=np.zeros((1, 24), dtype=np.float32),
    )

    with pytest.raises(
        ValueError,
        match="model predictions outside 0..23",
    ):
        result.validate()
