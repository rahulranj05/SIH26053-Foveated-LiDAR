from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

from mapping.foveation_controller import (
    FOVEATION_REASONS,
    FoveationControllerConfig,
    compute_foveation,
    count_dominant_reasons,
    dominant_reason,
    fuse_importance,
    resolution_from_importance,
    validate_importance,
)


def test_valid_importance():
    importance = np.array(
        [0.0, 0.25, 0.5, 0.75, 1.0]
    )

    assert validate_importance(
        importance
    )


def test_importance_must_be_1d():
    importance = np.zeros(
        (2, 2)
    )

    with pytest.raises(ValueError):
        validate_importance(
            importance
        )


def test_importance_range():
    importance = np.array(
        [0.0, 0.5, 1.1]
    )

    with pytest.raises(ValueError):
        validate_importance(
            importance
        )


def test_importance_must_be_finite():
    importance = np.array(
        [0.0, np.nan, 1.0]
    )

    with pytest.raises(ValueError):
        validate_importance(
            importance
        )


def test_fusion_uses_elementwise_maximum():
    signals = {
        "DISTANCE": np.array(
            [0.1, 0.8, 0.2]
        ),
        "SEMANTIC": np.array(
            [0.5, 0.3, 0.9]
        ),
        "DYNAMIC": np.array(
            [0.2, 0.95, 0.4]
        ),
    }

    fused = fuse_importance(
        signals
    )

    expected = np.array(
        [0.5, 0.95, 0.9]
    )

    assert np.allclose(
        fused,
        expected,
    )


def test_fusion_never_reduces_importance():
    distance = np.array(
        [0.2, 0.5, 0.8]
    )

    semantic = np.array(
        [0.7, 0.1, 0.3]
    )

    dynamic = np.array(
        [0.1, 0.9, 0.2]
    )

    fused = fuse_importance(
        {
            "DISTANCE": distance,
            "SEMANTIC": semantic,
            "DYNAMIC": dynamic,
        }
    )

    assert np.all(
        fused >= distance
    )

    assert np.all(
        fused >= semantic
    )

    assert np.all(
        fused >= dynamic
    )


def test_empty_signal_dictionary():
    with pytest.raises(ValueError):
        fuse_importance({})


def test_mismatched_signal_lengths():
    signals = {
        "DISTANCE": np.zeros(5),
        "SEMANTIC": np.zeros(4),
    }

    with pytest.raises(ValueError):
        fuse_importance(
            signals
        )


def test_resolution_boundaries():
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

    resolution = resolution_from_importance(
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


def test_only_hierarchical_resolutions():
    importance = np.linspace(
        0.0,
        1.0,
        1000,
    )

    resolution = resolution_from_importance(
        importance
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


def test_dominant_reason():
    signals = {
        "DISTANCE": np.array(
            [0.2, 0.9, 0.1]
        ),
        "KINEMATIC": np.array(
            [0.5, 0.4, 0.1]
        ),
        "PREDICTED_PATH": np.array(
            [0.3, 0.8, 0.2]
        ),
        "SEMANTIC": np.array(
            [0.7, 0.3, 0.4]
        ),
        "DYNAMIC": np.array(
            [0.4, 0.6, 0.9]
        ),
    }

    reason = dominant_reason(
        signals
    )

    expected = np.array(
        [
            "SEMANTIC",
            "DISTANCE",
            "DYNAMIC",
        ],
        dtype=object,
    )

    assert np.array_equal(
        reason,
        expected,
    )


def test_tie_is_deterministic():
    signals = {
        "DISTANCE": np.array(
            [0.8]
        ),
        "SEMANTIC": np.array(
            [0.8]
        ),
    }

    reason = dominant_reason(
        signals
    )

    assert reason[0] == "DISTANCE"


def test_compute_foveation():
    signals = {
        "DISTANCE": np.array(
            [0.1, 0.3, 0.6, 0.9]
        ),
        "SEMANTIC": np.array(
            [0.0, 0.7, 0.2, 0.1]
        ),
        "DYNAMIC": np.array(
            [0.0, 0.1, 0.8, 1.0]
        ),
    }

    result = compute_foveation(
        signals
    )

    assert set(
        result.keys()
    ) == {
        "importance",
        "resolution",
        "dominant_reason",
    }

    assert np.allclose(
        result["importance"],
        [0.1, 0.7, 0.8, 1.0],
    )

    assert np.allclose(
        result["resolution"],
        [0.40, 0.10, 0.05, 0.05],
    )


def test_compute_foveation_preserves_all_signals():
    rng = np.random.default_rng(
        42
    )

    signals = {
        name: rng.uniform(
            0.0,
            1.0,
            size=1000,
        )
        for name in FOVEATION_REASONS
    }

    result = compute_foveation(
        signals
    )

    expected = np.maximum.reduce(
        list(signals.values())
    )

    assert np.allclose(
        result["importance"],
        expected,
    )


def test_count_dominant_reasons():
    reasons = np.array(
        [
            "DISTANCE",
            "SEMANTIC",
            "SEMANTIC",
            "DYNAMIC",
            "KINEMATIC",
            "PREDICTED_PATH",
            "DYNAMIC",
        ],
        dtype=object,
    )

    counts = count_dominant_reasons(
        reasons
    )

    assert counts["DISTANCE"] == 1
    assert counts["KINEMATIC"] == 1
    assert counts["PREDICTED_PATH"] == 1
    assert counts["SEMANTIC"] == 2
    assert counts["DYNAMIC"] == 2


def test_count_includes_zero_categories():
    reasons = np.array(
        [
            "SEMANTIC",
            "SEMANTIC",
        ],
        dtype=object,
    )

    counts = count_dominant_reasons(
        reasons
    )

    for reason in FOVEATION_REASONS:
        assert reason in counts

    assert counts["SEMANTIC"] == 2
    assert counts["DISTANCE"] == 0
    assert counts["KINEMATIC"] == 0
    assert counts["PREDICTED_PATH"] == 0
    assert counts["DYNAMIC"] == 0


def test_custom_thresholds():
    config = FoveationControllerConfig(
        fine_threshold=0.80,
        medium_threshold=0.60,
        coarse_threshold=0.30,
    )

    importance = np.array(
        [0.29, 0.30, 0.59, 0.60, 0.79, 0.80]
    )

    resolution = resolution_from_importance(
        importance,
        config,
    )

    expected = np.array(
        [
            0.40,
            0.20,
            0.20,
            0.10,
            0.10,
            0.05,
        ]
    )

    assert np.allclose(
        resolution,
        expected,
    )


def test_custom_threshold_validation():
    with pytest.raises(ValueError):
        FoveationControllerConfig(
            fine_threshold=0.4,
            medium_threshold=0.6,
            coarse_threshold=0.2,
        )


def test_output_length():
    rng = np.random.default_rng(
        123
    )

    signals = {
        "DISTANCE": rng.uniform(
            0.0,
            1.0,
            500,
        ),
        "SEMANTIC": rng.uniform(
            0.0,
            1.0,
            500,
        ),
    }

    result = compute_foveation(
        signals
    )

    assert len(
        result["importance"]
    ) == 500

    assert len(
        result["resolution"]
    ) == 500

    assert len(
        result["dominant_reason"]
    ) == 500