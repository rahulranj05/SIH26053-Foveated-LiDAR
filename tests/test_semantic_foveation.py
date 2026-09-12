import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mapping.semantic_foveation import (
    DEFAULT_SEMANTIC_IMPORTANCE,
    SEMANTIC_CLASSES,
    SemanticFoveationConfig,
    fuse_semantic_importance,
    resolution_from_semantic_importance,
    semantic_importance,
    validate_semantic_labels,
)


def test_semantic_classes_exist():
    assert SEMANTIC_CLASSES["UNKNOWN"] == 0
    assert SEMANTIC_CLASSES["DRIVABLE"] == 1
    assert SEMANTIC_CLASSES["NON_DRIVABLE"] == 2
    assert SEMANTIC_CLASSES["STATIC_OBSTACLE"] == 3
    assert SEMANTIC_CLASSES["VEHICLE"] == 4
    assert SEMANTIC_CLASSES["VULNERABLE_USER"] == 5


def test_default_importance_order():
    importance = semantic_importance(
        np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
    )

    assert importance[1] == 0.0
    assert importance[0] > importance[1]
    assert importance[2] > importance[0]
    assert importance[3] > importance[2]
    assert importance[4] > importance[3]
    assert importance[5] > importance[4]


def test_drivable_has_zero_semantic_importance():
    labels = np.array([1, 1, 1], dtype=np.int32)

    result = semantic_importance(labels)

    np.testing.assert_array_equal(result, np.zeros(3))


def test_vulnerable_user_has_maximum_importance():
    labels = np.array([5], dtype=np.int32)

    result = semantic_importance(labels)

    assert result[0] == 1.0


def test_unknown_is_conservative():
    labels = np.array([0], dtype=np.int32)

    result = semantic_importance(labels)

    assert result[0] == DEFAULT_SEMANTIC_IMPORTANCE[0]
    assert result[0] > 0.0


def test_semantic_importance_preserves_shape():
    labels = np.array([1, 2, 3, 4, 5], dtype=np.int32)

    result = semantic_importance(labels)

    assert result.shape == labels.shape


def test_semantic_importance_empty_input():
    labels = np.array([], dtype=np.int32)

    result = semantic_importance(labels)

    assert result.shape == (0,)


def test_validate_valid_labels():
    labels = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)

    assert validate_semantic_labels(labels) is True


def test_validate_empty_labels():
    labels = np.array([], dtype=np.int32)

    assert validate_semantic_labels(labels) is True


def test_validate_rejects_negative_label():
    labels = np.array([-1, 1, 2], dtype=np.int32)

    with pytest.raises(ValueError):
        validate_semantic_labels(labels)


def test_validate_rejects_unknown_label():
    labels = np.array([1, 2, 99], dtype=np.int32)

    with pytest.raises(ValueError):
        validate_semantic_labels(labels)


def test_validate_rejects_non_integer_labels():
    labels = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        validate_semantic_labels(labels)


def test_fusion_preserves_high_base_importance():
    base = np.array([0.9], dtype=np.float64)
    semantic = np.array([0.0], dtype=np.float64)

    result = fuse_semantic_importance(base, semantic)

    assert result[0] >= base[0]


def test_vehicle_boosts_low_base_importance():
    base = np.array([0.1], dtype=np.float64)
    semantic = np.array([0.85], dtype=np.float64)

    result = fuse_semantic_importance(base, semantic)

    assert result[0] > base[0]


def test_vulnerable_user_boosts_low_base_importance():
    base = np.array([0.1], dtype=np.float64)
    semantic = np.array([1.0], dtype=np.float64)

    result = fuse_semantic_importance(base, semantic)

    assert result[0] == 1.0


def test_fusion_never_reduces_base_importance():
    base = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
    semantic = np.array([0.0, 0.1, 0.2, 0.3, 0.0])

    result = fuse_semantic_importance(base, semantic)

    assert np.all(result >= base)


def test_semantic_weight_zero_uses_base_importance():
    base = np.array([0.1, 0.5, 0.9])
    semantic = np.array([1.0, 0.0, 0.2])

    config = SemanticFoveationConfig(semantic_weight=0.0)

    result = fuse_semantic_importance(
        base,
        semantic,
        config=config,
    )

    np.testing.assert_allclose(result, base)


def test_semantic_weight_one_allows_full_semantic_refinement():
    base = np.array([0.1, 0.5, 0.9])
    semantic = np.array([1.0, 0.0, 0.2])

    config = SemanticFoveationConfig(semantic_weight=1.0)

    result = fuse_semantic_importance(
        base,
        semantic,
        config=config,
    )

    expected = np.maximum(base, semantic)

    np.testing.assert_allclose(result, expected)


def test_resolution_thresholds():
    importance = np.array(
        [0.80, 0.75, 0.60, 0.50, 0.40, 0.25, 0.10]
    )

    result = resolution_from_semantic_importance(importance)

    expected = np.array(
        [0.05, 0.05, 0.10, 0.10, 0.20, 0.20, 0.40]
    )

    np.testing.assert_allclose(result, expected)


def test_resolution_values_are_hierarchy_safe():
    importance = np.linspace(0.0, 1.0, 101)

    result = resolution_from_semantic_importance(importance)

    valid_resolutions = {0.05, 0.10, 0.20, 0.40}

    assert set(np.unique(result)).issubset(valid_resolutions)


def test_custom_semantic_importance():
    config = SemanticFoveationConfig(
        vulnerable_user_importance=0.95,
        vehicle_importance=0.80,
    )

    labels = np.array([4, 5], dtype=np.int32)

    result = semantic_importance(labels, config=config)

    np.testing.assert_allclose(
        result,
        np.array([0.80, 0.95]),
    )


def test_invalid_semantic_weight():
    with pytest.raises(ValueError):
        SemanticFoveationConfig(semantic_weight=-0.1)

    with pytest.raises(ValueError):
        SemanticFoveationConfig(semantic_weight=1.1)