from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mapping.hierarchical_foveated_mapper import HierarchicalLeafMap
from mapping.map_visualizer import (
    MapVisualizationConfig,
    leaf_cell_rectangles,
    plot_foveamap_result,
    plot_leaf_map,
    plot_lidar_points,
    save_foveamap_figure,
)
from mapping.foveamap_pipeline import FoveaMapPipelineResult


@pytest.fixture
def sample_leaf_map():
    return HierarchicalLeafMap(
        levels=np.array([0, 1, 2, 3], dtype=np.int8),
        resolutions=np.array([0.05, 0.10, 0.20, 0.40], dtype=np.float64),
        ix=np.array([0, 2, 5, 3], dtype=np.int64),
        iy=np.array([0, 1, -2, 4], dtype=np.int64),
        z_min=np.array([0.0, 0.1, 0.2, 0.3]),
        z_max=np.array([0.5, 0.6, 0.7, 0.8]),
        z_mean=np.array([0.25, 0.35, 0.45, 0.55]),
        z_variance=np.array([0.01, 0.02, 0.03, 0.04]),
        point_count=np.array([1, 1, 1, 2], dtype=np.int64),
        dominant_reason=np.array(
            [
                "DISTANCE",
                "KINEMATIC",
                "SEMANTIC",
                "DYNAMIC",
            ],
            dtype=object,
        ),
        input_points=5,
    )


@pytest.fixture
def sample_xyz():
    return np.array(
        [
            [0.10, 0.10, 0.20],
            [0.20, 0.10, 0.30],
            [0.30, 0.20, 0.40],
            [0.40, 0.30, 0.50],
            [0.50, 0.40, 0.60],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def sample_result(sample_leaf_map):
    return FoveaMapPipelineResult(
        foveation={
            "importance": np.array(
                [0.90, 0.70, 0.50, 0.30, 0.10],
                dtype=np.float64,
            ),
            "resolution": np.array(
                [0.05, 0.10, 0.20, 0.40, 0.40],
                dtype=np.float64,
            ),
            "dominant_reason": np.array(
                [
                    "DISTANCE",
                    "KINEMATIC",
                    "SEMANTIC",
                    "DYNAMIC",
                    "DISTANCE",
                ],
                dtype=object,
            ),
        },
        leaf_map=sample_leaf_map,
        timings_ms={
            "controller": 1.0,
            "mapper": 2.0,
            "total": 3.0,
        },
    )


def test_config_defaults():
    config = MapVisualizationConfig()

    assert config.point_size == 0.5
    assert config.point_alpha == 0.20
    assert config.cell_alpha == 0.60
    assert config.show_points is True
    assert config.show_cell_edges is True
    assert config.show_reason_labels is True
    assert config.show_resolution_legend is True
    assert config.equal_aspect is True
    assert config.max_points == 100_000


@pytest.mark.parametrize(
    "field,value",
    [
        ("point_size", -1.0),
        ("point_alpha", -0.1),
        ("point_alpha", 1.1),
        ("cell_alpha", -0.1),
        ("cell_alpha", 1.1),
        ("max_points", 0),
    ],
)
def test_config_rejects_invalid_values(field, value):
    with pytest.raises(ValueError):
        MapVisualizationConfig(**{field: value})


def test_leaf_cell_rectangles(sample_leaf_map):
    rectangles = leaf_cell_rectangles(sample_leaf_map)

    assert len(rectangles) == 4

    assert rectangles[0].get_x() == pytest.approx(0.0)
    assert rectangles[0].get_y() == pytest.approx(0.0)
    assert rectangles[0].get_width() == pytest.approx(0.05)
    assert rectangles[0].get_height() == pytest.approx(0.05)

    assert rectangles[2].get_x() == pytest.approx(1.0)
    assert rectangles[2].get_y() == pytest.approx(-0.4)
    assert rectangles[2].get_width() == pytest.approx(0.20)
    assert rectangles[2].get_height() == pytest.approx(0.20)


def test_leaf_cell_rectangles_rejects_invalid_leaf_map():
    with pytest.raises(TypeError):
        leaf_cell_rectangles(object())


def test_plot_lidar_points(sample_xyz):
    fig, ax = plt.subplots()

    collection = plot_lidar_points(ax, sample_xyz)

    assert collection is not None
    assert len(collection.get_offsets()) == 5

    plt.close(fig)


def test_plot_lidar_points_respects_point_limit(sample_xyz):
    fig, ax = plt.subplots()

    config = MapVisualizationConfig(max_points=3)
    collection = plot_lidar_points(ax, sample_xyz, config)

    assert len(collection.get_offsets()) == 3

    plt.close(fig)


def test_plot_lidar_points_rejects_invalid_xyz():
    fig, ax = plt.subplots()

    invalid_xyz = np.zeros((10, 2))

    with pytest.raises(ValueError):
        plot_lidar_points(ax, invalid_xyz)

    plt.close(fig)


def test_plot_leaf_map(sample_leaf_map):
    fig, ax = plt.subplots()

    collection = plot_leaf_map(ax, sample_leaf_map)

    assert collection is not None
    assert len(collection.get_paths()) == 4

    plt.close(fig)


def test_plot_leaf_map_handles_empty_map():
    empty_map = HierarchicalLeafMap(
        levels=np.array([], dtype=np.int8),
        resolutions=np.array([], dtype=np.float64),
        ix=np.array([], dtype=np.int64),
        iy=np.array([], dtype=np.int64),
        z_min=np.array([], dtype=np.float64),
        z_max=np.array([], dtype=np.float64),
        z_mean=np.array([], dtype=np.float64),
        z_variance=np.array([], dtype=np.float64),
        point_count=np.array([], dtype=np.int64),
        dominant_reason=np.array([], dtype=object),
        input_points=0,
    )

    fig, ax = plt.subplots()

    collection = plot_leaf_map(ax, empty_map)

    assert collection is not None
    assert len(collection.get_paths()) == 0

    plt.close(fig)


def test_plot_foveamap_result(sample_xyz, sample_result):
    figure = plot_foveamap_result(sample_xyz, sample_result)

    assert figure is not None
    assert len(figure.axes) >= 1

    plt.close(figure)


def test_plot_foveamap_result_rejects_mismatched_input_count(
    sample_result,
):
    xyz = np.zeros((4, 3), dtype=np.float64)

    with pytest.raises(ValueError):
        plot_foveamap_result(xyz, sample_result)


def test_plot_foveamap_result_rejects_invalid_result(sample_xyz):
    with pytest.raises(TypeError):
        plot_foveamap_result(sample_xyz, object())


def test_plot_foveamap_result_can_hide_points(
    sample_xyz,
    sample_result,
):
    config = MapVisualizationConfig(show_points=False)

    figure = plot_foveamap_result(
        sample_xyz,
        sample_result,
        config,
    )

    assert figure is not None
    assert len(figure.axes) >= 1

    plt.close(figure)


def test_save_foveamap_figure(tmp_path, sample_xyz, sample_result):
    figure = plot_foveamap_result(sample_xyz, sample_result)

    output_path = tmp_path / "foveamap_test.png"

    save_foveamap_figure(
        figure,
        output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    plt.close(figure)