"""
A37 — Temporal Motion Optimization Tests

Validates the optimized temporal correspondence implementation while
preserving the A25 public API and numerical behaviour.
"""

from __future__ import annotations

import time

import numpy as np

from mapping.temporal_motion import (
    TemporalMotionConfig,
    _nearest_distances,
    _nearest_distances_bruteforce,
    _nearest_distances_within_radius,
    _unique_voxel_centres,
    temporal_motion_evidence,
)


def test_bruteforce_backend_matches_nearest_distance_helper():
    rng = np.random.default_rng(26053)

    source = rng.uniform(
        -10.0,
        10.0,
        size=(30, 3),
    )

    target = rng.uniform(
        -10.0,
        10.0,
        size=(40, 3),
    )

    expected = _nearest_distances_bruteforce(
        source,
        target,
    )

    actual = _nearest_distances(
        source,
        target,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=1e-12,
    )


def test_bruteforce_empty_source():
    source = np.empty(
        (0, 3),
        dtype=np.float64,
    )

    target = np.array(
        [[0.0, 0.0, 0.0]],
        dtype=np.float64,
    )

    distances = _nearest_distances_bruteforce(
        source,
        target,
    )

    assert distances.shape == (0,)
    assert distances.dtype == np.float64


def test_bruteforce_empty_target():
    source = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    target = np.empty(
        (0, 3),
        dtype=np.float64,
    )

    distances = _nearest_distances_bruteforce(
        source,
        target,
    )

    assert distances.shape == (2,)
    assert np.all(np.isinf(distances))


def test_spatial_hash_empty_source():
    source = np.empty(
        (0, 3),
        dtype=np.float64,
    )

    target = np.array(
        [[0.0, 0.0, 0.0]],
        dtype=np.float64,
    )

    distances = _nearest_distances_within_radius(
        source,
        target,
        radius=1.5,
    )

    assert distances.shape == (0,)
    assert distances.dtype == np.float64


def test_spatial_hash_empty_target():
    source = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    target = np.empty(
        (0, 3),
        dtype=np.float64,
    )

    distances = _nearest_distances_within_radius(
        source,
        target,
        radius=1.5,
    )

    assert distances.shape == (2,)
    assert np.all(np.isinf(distances))


def test_spatial_hash_respects_radius():
    source = np.array(
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    target = np.array(
        [
            [0.5, 0.0, 0.0],
            [7.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    distances = _nearest_distances_within_radius(
        source,
        target,
        radius=1.0,
    )

    assert np.isclose(
        distances[0],
        0.5,
    )

    assert np.isinf(distances[1])


def test_spatial_hash_matches_bruteforce_within_radius():
    rng = np.random.default_rng(26053)

    source = rng.uniform(
        -10.0,
        10.0,
        size=(80, 3),
    )

    target = rng.uniform(
        -10.0,
        10.0,
        size=(100, 3),
    )

    radius = 1.5

    brute_force = _nearest_distances_bruteforce(
        source,
        target,
    )

    expected = np.where(
        brute_force <= radius,
        brute_force,
        np.inf,
    )

    actual = _nearest_distances_within_radius(
        source,
        target,
        radius,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=1e-12,
    )


def test_optimized_motion_evidence_matches_bruteforce_reference():
    rng = np.random.default_rng(26053)

    current = rng.uniform(
        -8.0,
        8.0,
        size=(80, 3),
    )

    previous = rng.uniform(
        -8.0,
        8.0,
        size=(100, 3),
    )

    config = TemporalMotionConfig(
        voxel_size=0.20,
        motion_threshold=0.50,
        max_correspondence_distance=1.50,
        min_probability=0.0,
    )

    # IMPORTANT:
    # _unique_voxel_centres returns:
    #
    #     (unique_keys, centres)
    #
    # The previous broken test assigned the entire tuple to
    # previous_centres, causing the AttributeError.
    _, previous_centres = _unique_voxel_centres(
        previous,
        config.voxel_size,
    )

    distances = _nearest_distances_bruteforce(
        current,
        previous_centres,
    )

    threshold = config.motion_threshold
    maximum = config.max_correspondence_distance

    expected = np.clip(
        (
            distances - threshold
        )
        / (
            maximum - threshold
        ),
        0.0,
        1.0,
    )

    expected = np.maximum(
        expected,
        config.min_probability,
    )

    expected = np.clip(
        expected,
        0.0,
        1.0,
    )

    actual = temporal_motion_evidence(
        previous,
        current,
        config,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=1e-12,
    )


def test_optimized_motion_evidence_preserves_current_point_count():
    rng = np.random.default_rng(26053)

    previous = rng.uniform(
        -20.0,
        20.0,
        size=(500, 3),
    )

    current = rng.uniform(
        -20.0,
        20.0,
        size=(700, 3),
    )

    evidence = temporal_motion_evidence(
        previous,
        current,
    )

    assert evidence.shape == (700,)
    assert np.all(np.isfinite(evidence))
    assert np.all(
        (evidence >= 0.0)
        & (evidence <= 1.0)
    )


def test_optimized_motion_evidence_empty_current():
    previous = np.array(
        [[0.0, 0.0, 0.0]],
        dtype=np.float64,
    )

    current = np.empty(
        (0, 3),
        dtype=np.float64,
    )

    evidence = temporal_motion_evidence(
        previous,
        current,
    )

    assert evidence.shape == (0,)


def test_optimized_motion_evidence_empty_previous():
    previous = np.empty(
        (0, 3),
        dtype=np.float64,
    )

    current = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
        ],
        dtype=np.float64,
    )

    evidence = temporal_motion_evidence(
        previous,
        current,
    )

    np.testing.assert_array_equal(
        evidence,
        np.ones(2),
    )


def test_optimized_backend_handles_large_point_cloud():
    rng = np.random.default_rng(26053)

    points = rng.uniform(
        -50.0,
        50.0,
        size=(10_000, 3),
    )

    start = time.perf_counter()

    distances = _nearest_distances_within_radius(
        points,
        points,
        radius=1.5,
    )

    elapsed = time.perf_counter() - start

    assert distances.shape == (10_000,)
    assert np.all(np.isfinite(distances))

    # Loose machine-independent sanity bound.
    assert elapsed < 2.0