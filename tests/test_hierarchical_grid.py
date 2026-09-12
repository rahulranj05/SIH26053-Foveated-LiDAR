import numpy as np
import pytest

from mapping.hierarchical_grid import (
    LEVEL_RESOLUTIONS,
    HierarchicalCell,
    resolution_for_level,
    level_for_resolution,
    cell_from_world,
    parent_cell,
    child_cells,
    is_child_of,
    cells_are_aligned,
    hierarchy_cells_for_points,
)


def test_hierarchy_has_expected_resolutions():
    assert LEVEL_RESOLUTIONS == {
        0: 0.05,
        1: 0.10,
        2: 0.20,
        3: 0.40,
    }


def test_resolution_for_level():
    assert resolution_for_level(0) == 0.05
    assert resolution_for_level(1) == 0.10
    assert resolution_for_level(2) == 0.20
    assert resolution_for_level(3) == 0.40


def test_invalid_level_raises():
    with pytest.raises(ValueError):
        resolution_for_level(4)


def test_level_for_resolution():
    assert level_for_resolution(0.05) == 0
    assert level_for_resolution(0.10) == 1
    assert level_for_resolution(0.20) == 2
    assert level_for_resolution(0.40) == 3


def test_invalid_resolution_raises():
    with pytest.raises(ValueError):
        level_for_resolution(0.50)


def test_world_to_cell_positive_coordinates():
    cell = cell_from_world(1.23, 2.34, level=0)

    assert cell.level == 0
    assert cell.resolution == 0.05
    assert cell.x_index == 24
    assert cell.y_index == 46


def test_world_to_cell_negative_coordinates():
    cell = cell_from_world(-0.01, -0.01, level=0)

    assert cell.x_index == -1
    assert cell.y_index == -1
    assert np.isclose(cell.x_min, -0.05)
    assert np.isclose(cell.x_max, 0.0)
    assert np.isclose(cell.y_min, -0.05)
    assert np.isclose(cell.y_max, 0.0)


def test_cell_geometry():
    cell = HierarchicalCell(
        level=1,
        x_index=10,
        y_index=-5,
    )

    assert np.isclose(cell.resolution, 0.10)
    assert np.isclose(cell.x_min, 1.0)
    assert np.isclose(cell.x_max, 1.1)
    assert np.isclose(cell.y_min, -0.5)
    assert np.isclose(cell.y_max, -0.4)
    assert np.isclose(cell.x_center, 1.05)
    assert np.isclose(cell.y_center, -0.45)


def test_parent_child_relationship():
    parent = HierarchicalCell(
        level=1,
        x_index=10,
        y_index=20,
    )

    children = child_cells(parent)

    assert len(children) == 4

    for child in children:
        assert is_child_of(child, parent)
        assert parent_cell(child) == parent


def test_parent_of_coarsest_level_is_none():
    cell = HierarchicalCell(
        level=3,
        x_index=5,
        y_index=5,
    )

    assert parent_cell(cell) is None


def test_finest_level_has_no_children():
    cell = HierarchicalCell(
        level=0,
        x_index=5,
        y_index=5,
    )

    assert child_cells(cell) == ()


def test_alignment_across_multiple_levels():
    coarse = HierarchicalCell(
        level=3,
        x_index=0,
        y_index=0,
    )

    fine = HierarchicalCell(
        level=0,
        x_index=7,
        y_index=6,
    )

    assert cells_are_aligned(coarse, fine)


def test_non_aligned_cells_are_rejected():
    coarse = HierarchicalCell(
        level=3,
        x_index=0,
        y_index=0,
    )

    fine = HierarchicalCell(
        level=0,
        x_index=8,
        y_index=6,
    )

    assert not cells_are_aligned(coarse, fine)


def test_four_children_tile_parent_exactly():
    parent = HierarchicalCell(
        level=1,
        x_index=4,
        y_index=-2,
    )

    children = child_cells(parent)

    assert len(children) == 4

    parent_x_min = parent.x_min
    parent_x_max = parent.x_max
    parent_y_min = parent.y_min
    parent_y_max = parent.y_max

    for child in children:
        assert child.x_min >= parent_x_min
        assert child.x_max <= parent_x_max
        assert child.y_min >= parent_y_min
        assert child.y_max <= parent_y_max


def test_children_have_correct_resolution():
    parent = HierarchicalCell(
        level=2,
        x_index=4,
        y_index=3,
    )

    children = child_cells(parent)

    assert all(child.level == 1 for child in children)
    assert all(child.resolution == 0.10 for child in children)


def test_points_produce_unique_cells():
    xyz = np.array([
        [0.01, 0.01, 1.0],
        [0.02, 0.02, 2.0],
        [0.06, 0.01, 3.0],
    ])

    cells = hierarchy_cells_for_points(
        xyz,
        level=0,
    )

    assert len(cells) == 2


def test_empty_point_cloud():
    xyz = np.empty((0, 3), dtype=np.float64)

    cells = hierarchy_cells_for_points(
        xyz,
        level=0,
    )

    assert cells == []


def test_invalid_point_cloud_shape():
    xyz = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    with pytest.raises(ValueError):
        hierarchy_cells_for_points(
            xyz,
            level=0,
        )


def test_nonfinite_points_are_rejected():
    xyz = np.array([
        [1.0, 2.0, 3.0],
        [np.nan, 1.0, 2.0],
    ])

    with pytest.raises(ValueError):
        hierarchy_cells_for_points(
            xyz,
            level=0,
        )