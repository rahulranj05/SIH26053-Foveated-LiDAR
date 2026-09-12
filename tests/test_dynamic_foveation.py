from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

from mapping.dynamic_foveation import (
    DynamicFoveationConfig,
    dynamic_importance,
    dynamic_resolution,
    fuse_dynamic_importance,
    normalize_velocity,
    resolution_from_dynamic_importance,
    validate_dynamic_inputs,
)


def test_validate_valid_inputs():
    probability = np.array(
        [0.0, 0.25, 0.5, 1.0]
    )

    velocity = np.array(
        [0.0, 1.0, 5.0, 10.0]
    )

    assert validate_dynamic_inputs(
        probability,
        velocity,
    )


def test_probability_must_be_1d():
    probability = np.zeros((2, 2))
    velocity = np.zeros(4)

    with pytest.raises(ValueError):
        validate_dynamic_inputs(
            probability,
            velocity,
        )


def test_velocity_must_be_1d():
    probability = np.zeros(4)
    velocity = np.zeros((2, 2))

    with pytest.raises(ValueError):
        validate_dynamic_inputs(
            probability,
            velocity,
        )


def test_lengths_must_match():
    probability = np.zeros(4)
    velocity = np.zeros(3)

    with pytest.raises(ValueError):
        validate_dynamic_inputs(
            probability,
            velocity,
        )


def test_probability_range():
    probability = np.array(
        [0.0, 0.5, 1.1]
    )

    velocity = np.ones(3)

    with pytest.raises(ValueError):
        validate_dynamic_inputs(
            probability,
            velocity,
        )


def test_velocity_cannot_be_negative():
    probability = np.ones(3)
    velocity = np.array(
        [0.0, -1.0, 2.0]
    )

    with pytest.raises(ValueError):
        validate_dynamic_inputs(
            probability,
            velocity,
        )


def test_velocity_normalization():
    velocity = np.array(
        [0.0, 5.0, 10.0, 20.0]
    )

    normalized = normalize_velocity(
        velocity
    )

    assert np.allclose(
        normalized,
        [0.0, 0.5, 1.0, 1.0],
    )


def test_zero_dynamic_probability_gives_zero_importance():
    probability = np.zeros(4)
    velocity = np.array(
        [0.0, 2.0, 10.0, 100.0]
    )

    importance = dynamic_importance(
        probability,
        velocity,
    )

    assert np.allclose(
        importance,
        0.0,
    )


def test_higher_probability_increases_importance():
    probability = np.array(
        [0.1, 0.5, 1.0]
    )

    velocity = np.full(3, 5.0)

    importance = dynamic_importance(
        probability,
        velocity,
    )

    assert (
        importance[0]
        < importance[1]
        < importance[2]
    )


def test_higher_velocity_increases_importance():
    probability = np.full(
        4,
        0.8,
    )

    velocity = np.array(
        [0.0, 2.0, 5.0, 10.0]
    )

    importance = dynamic_importance(
        probability,
        velocity,
    )

    assert np.all(
        np.diff(importance) >= 0.0
    )


def test_importance_is_clipped():
    config = DynamicFoveationConfig(
        dynamic_weight=2.0,
        velocity_weight=2.0,
    )

    probability = np.ones(3)
    velocity = np.full(
        3,
        100.0,
    )

    importance = dynamic_importance(
        probability,
        velocity,
        config,
    )

    assert np.all(
        importance <= 1.0
    )

    assert np.all(
        importance >= 0.0
    )


def test_dynamic_importance_resolution_boundaries():
    importance = np.array(
        [
            0.0,
            0.249999,
            0.25,
            0.499999,
            0.50,
            0.749999,
            0.75,
            1.0,
        ]
    )

    resolution = resolution_from_dynamic_importance(
        importance
    )

    expected = np.array(
        [
            0.40,
            0.40,
            0.20,
            0.20,
            0.10,
            0.10,
            0.05,
            0.05,
        ]
    )

    assert np.allclose(
        resolution,
        expected,
    )


def test_fusion_never_reduces_base_importance():
    base = np.array(
        [0.2, 0.5, 0.8, 1.0]
    )

    dynamic = np.array(
        [0.1, 0.2, 0.9, 0.3]
    )

    fused = fuse_dynamic_importance(
        base,
        dynamic,
    )

    assert np.all(
        fused >= base
    )


def test_fusion_uses_maximum():
    base = np.array(
        [0.2, 0.5, 0.8]
    )

    dynamic = np.array(
        [0.7, 0.3, 0.9]
    )

    fused = fuse_dynamic_importance(
        base,
        dynamic,
    )

    expected = np.array(
        [0.7, 0.5, 0.9]
    )

    assert np.allclose(
        fused,
        expected,
    )


def test_dynamic_weight_zero_preserves_base():
    config = DynamicFoveationConfig(
        dynamic_weight=0.0
    )

    base = np.array(
        [0.1, 0.4, 0.9]
    )

    dynamic = np.array(
        [1.0, 1.0, 1.0]
    )

    fused = fuse_dynamic_importance(
        base,
        dynamic,
        config,
    )

    assert np.allclose(
        fused,
        base,
    )


def test_dynamic_resolution_convenience_function():
    probability = np.array(
        [0.0, 0.3, 0.6, 1.0]
    )

    velocity = np.zeros(4)

    resolution = dynamic_resolution(
        probability,
        velocity,
    )

    expected = np.array(
        [0.40, 0.20, 0.10, 0.05]
    )

    assert np.allclose(
        resolution,
        expected,
    )


def test_output_length_matches_input():
    probability = np.linspace(
        0.0,
        1.0,
        100,
    )

    velocity = np.linspace(
        0.0,
        10.0,
        100,
    )

    resolution = dynamic_resolution(
        probability,
        velocity,
    )

    assert len(resolution) == 100


def test_output_uses_only_hierarchical_resolutions():
    probability = np.linspace(
        0.0,
        1.0,
        100,
    )

    velocity = np.linspace(
        0.0,
        20.0,
        100,
    )

    resolution = dynamic_resolution(
        probability,
        velocity,
    )

    allowed = {
        0.05,
        0.10,
        0.20,
        0.40,
    }

    assert set(
        np.unique(resolution)
    ).issubset(allowed)


def test_fast_dynamic_object_gets_finer_resolution():
    probability = np.array(
        [0.8, 0.8]
    )

    velocity = np.array(
        [0.0, 10.0]
    )

    resolution = dynamic_resolution(
        probability,
        velocity,
    )

    assert resolution[1] <= resolution[0]


def test_stationary_zero_probability_is_coarse():
    probability = np.array(
        [0.0]
    )

    velocity = np.array(
        [0.0]
    )

    resolution = dynamic_resolution(
        probability,
        velocity,
    )

    assert resolution[0] == 0.40