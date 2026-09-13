from __future__ import annotations

import numpy as np
import pytest

from evaluation.ablation import (
    ABLATION_SIGNALS,
    ABLATION_VARIANTS,
    AblationComparisonResult,
    AblationVariantResult,
    run_ablation_study,
    select_ablation_signals,
    validate_ablation_signals,
)


def _signals(point_count: int = 8) -> dict[str, np.ndarray]:
    return {
        "DISTANCE": np.linspace(0.0, 1.0, point_count),
        "KINEMATIC": np.linspace(0.1, 0.9, point_count),
        "PREDICTED_PATH": np.linspace(0.2, 0.8, point_count),
        "SEMANTIC": np.linspace(0.3, 0.7, point_count),
        "DYNAMIC": np.linspace(0.4, 0.6, point_count),
    }


def _xyz(point_count: int = 8) -> np.ndarray:
    return np.column_stack(
        (
            np.arange(point_count, dtype=np.float64),
            np.zeros(point_count),
            np.ones(point_count),
        )
    )


class _FakeLeafMap:
    def __init__(self, count: int = 5) -> None:
        self.levels = np.zeros(count, dtype=np.int8)
        self.resolutions = np.full(
            count,
            0.05,
            dtype=np.float64,
        )


class _FakeResult:
    def __init__(self, count: int = 5) -> None:
        self.leaf_map = _FakeLeafMap(count)
        self.foveation = {
            "resolution": np.full(
                8,
                0.05,
                dtype=np.float64,
            ),
            "dominant_reason": np.array(
                ["DISTANCE"] * 8,
                dtype=object,
            ),
        }


def _fake_pipeline(
    xyz: np.ndarray,
    signals: dict[str, np.ndarray],
) -> _FakeResult:
    return _FakeResult()


def test_variant_matrix_is_complete():
    assert ABLATION_VARIANTS == (
        "DISTANCE_ONLY",
        "DISTANCE_KINEMATIC",
        "DISTANCE_PATH",
        "DISTANCE_SEMANTIC",
        "DISTANCE_DYNAMIC",
        "FULL_FOVEAMAP",
    )


def test_each_variant_has_expected_signals():
    assert ABLATION_SIGNALS["DISTANCE_ONLY"] == (
        "DISTANCE",
    )

    assert ABLATION_SIGNALS["DISTANCE_KINEMATIC"] == (
        "DISTANCE",
        "KINEMATIC",
    )

    assert ABLATION_SIGNALS["DISTANCE_PATH"] == (
        "DISTANCE",
        "PREDICTED_PATH",
    )

    assert ABLATION_SIGNALS["DISTANCE_SEMANTIC"] == (
        "DISTANCE",
        "SEMANTIC",
    )

    assert ABLATION_SIGNALS["DISTANCE_DYNAMIC"] == (
        "DISTANCE",
        "DYNAMIC",
    )


def test_full_variant_contains_all_foveation_signals():
    assert ABLATION_SIGNALS["FULL_FOVEAMAP"] == (
        "DISTANCE",
        "KINEMATIC",
        "PREDICTED_PATH",
        "SEMANTIC",
        "DYNAMIC",
    )


def test_signal_validation_normalizes_names():
    signals = {
        "distance": np.ones(4),
        "semantic": np.zeros(4),
    }

    result = validate_ablation_signals(signals)

    assert set(result) == {"DISTANCE", "SEMANTIC"}


def test_signal_validation_rejects_non_1d_signal():
    with pytest.raises(ValueError, match="shape"):
        validate_ablation_signals(
            {
                "DISTANCE": np.ones((4, 1)),
            }
        )


def test_signal_validation_rejects_non_finite_signal():
    values = np.ones(4)
    values[2] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        validate_ablation_signals(
            {
                "DISTANCE": values,
            }
        )


def test_select_distance_only():
    result = select_ablation_signals(
        _signals(),
        "DISTANCE_ONLY",
    )

    assert tuple(result) == ("DISTANCE",)


def test_select_semantic_variant():
    result = select_ablation_signals(
        _signals(),
        "DISTANCE_SEMANTIC",
    )

    assert tuple(result) == (
        "DISTANCE",
        "SEMANTIC",
    )


def test_select_dynamic_variant():
    result = select_ablation_signals(
        _signals(),
        "DISTANCE_DYNAMIC",
    )

    assert tuple(result) == (
        "DISTANCE",
        "DYNAMIC",
    )


def test_select_full_variant():
    result = select_ablation_signals(
        _signals(),
        "FULL_FOVEAMAP",
    )

    assert tuple(result) == (
        "DISTANCE",
        "KINEMATIC",
        "PREDICTED_PATH",
        "SEMANTIC",
        "DYNAMIC",
    )


def test_missing_required_signal_is_rejected():
    signals = _signals()
    del signals["SEMANTIC"]

    with pytest.raises(ValueError, match="Missing required signals"):
        select_ablation_signals(
            signals,
            "DISTANCE_SEMANTIC",
        )


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError, match="Unknown ablation variant"):
        select_ablation_signals(
            _signals(),
            "INVALID",
        )


def test_ablation_study_runs_all_variants():
    result = run_ablation_study(
        _xyz(),
        _signals(),
        pipeline=_fake_pipeline,
    )

    assert isinstance(result, AblationComparisonResult)
    assert len(result.variants) == 6

    assert [
        variant.name
        for variant in result.variants
    ] == list(ABLATION_VARIANTS)


def test_ablation_result_contains_metrics():
    result = run_ablation_study(
        _xyz(),
        _signals(),
        pipeline=_fake_pipeline,
    )

    variant = result.by_name("DISTANCE_ONLY")

    assert isinstance(variant, AblationVariantResult)
    assert variant.point_count == 8
    assert variant.active_cells == 5
    assert variant.memory_bytes == 5 * 64
    assert variant.resolution_counts == {0.05: 5}
    assert variant.dominant_reason_counts == {
        "DISTANCE": 8
    }


def test_ablation_variant_memory_is_fixed_width():
    result = run_ablation_study(
        _xyz(),
        _signals(),
        pipeline=_fake_pipeline,
    )

    for variant in result.variants:
        assert variant.memory_bytes == variant.active_cells * 64


def test_ablation_fps_is_positive():
    result = run_ablation_study(
        _xyz(),
        _signals(),
        pipeline=_fake_pipeline,
    )

    for variant in result.variants:
        assert variant.latency_ms >= 0.0
        assert variant.latency_fps > 0.0


def test_ablation_checks_signal_length():
    signals = _signals()
    signals["DYNAMIC"] = np.ones(4)

    with pytest.raises(ValueError, match="expected 8"):
        run_ablation_study(
            _xyz(),
            signals,
            pipeline=_fake_pipeline,
        )


def test_ablation_checks_xyz_shape():
    with pytest.raises(ValueError, match="shape"):
        run_ablation_study(
            np.zeros((8, 2)),
            _signals(),
            pipeline=_fake_pipeline,
        )


def test_ablation_checks_xyz_finiteness():
    xyz = _xyz()
    xyz[0, 0] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        run_ablation_study(
            xyz,
            _signals(),
            pipeline=_fake_pipeline,
        )


def test_as_dict_is_serializable():
    result = run_ablation_study(
        _xyz(),
        _signals(),
        pipeline=_fake_pipeline,
    )

    data = result.as_dict()

    assert "variants" in data
    assert len(data["variants"]) == 6
    assert data["variants"][0]["name"] == "DISTANCE_ONLY"


def test_full_foveamap_accessor():
    result = run_ablation_study(
        _xyz(),
        _signals(),
        pipeline=_fake_pipeline,
    )

    assert result.full_foveamap.name == "FULL_FOVEAMAP"