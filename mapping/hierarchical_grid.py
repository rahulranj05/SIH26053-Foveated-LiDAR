from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# A9 HIERARCHY
#
# Level 0 = 0.05 m
# Level 1 = 0.10 m
# Level 2 = 0.20 m
# Level 3 = 0.40 m
#
# Level 0 is the finest level.
# Level 3 is the coarsest level.
#
# Every coarser level is exactly 2x the previous level.
# Therefore every parent cell contains exactly 4 children in 2D.
# ---------------------------------------------------------------------------

LEVEL_RESOLUTIONS = {
    0: 0.05,
    1: 0.10,
    2: 0.20,
    3: 0.40,
}


@dataclass(frozen=True)
class HierarchicalCell:
    """
    One alignment-safe cell in the hierarchical map.

    Level 0 is the finest resolution.
    Higher levels are progressively coarser.
    """

    level: int
    x_index: int
    y_index: int

    @property
    def resolution(self) -> float:
        return LEVEL_RESOLUTIONS[self.level]

    @property
    def x_min(self) -> float:
        return self.x_index * self.resolution

    @property
    def y_min(self) -> float:
        return self.y_index * self.resolution

    @property
    def x_max(self) -> float:
        return (self.x_index + 1) * self.resolution

    @property
    def y_max(self) -> float:
        return (self.y_index + 1) * self.resolution

    @property
    def x_center(self) -> float:
        return self.x_min + self.resolution / 2.0

    @property
    def y_center(self) -> float:
        return self.y_min + self.resolution / 2.0


def resolution_for_level(level: int) -> float:
    """
    Return the resolution associated with a hierarchy level.
    """

    if level not in LEVEL_RESOLUTIONS:
        raise ValueError(
            f"Invalid hierarchy level: {level}. "
            f"Valid levels are {sorted(LEVEL_RESOLUTIONS)}."
        )

    return LEVEL_RESOLUTIONS[level]


def level_for_resolution(resolution: float) -> int:
    """
    Return the hierarchy level corresponding to a resolution.
    """

    resolution = float(resolution)

    for level, level_resolution in LEVEL_RESOLUTIONS.items():
        if np.isclose(resolution, level_resolution):
            return level

    raise ValueError(
        f"Resolution {resolution} m is not part of the hierarchy."
    )


def cell_from_world(
    x: float,
    y: float,
    level: int,
) -> HierarchicalCell:
    """
    Convert world coordinates into an integer-indexed hierarchical cell.

    np.floor is deliberately used so negative coordinates behave correctly.
    """

    if not np.isfinite(x) or not np.isfinite(y):
        raise ValueError(
            "x and y must be finite."
        )

    resolution = resolution_for_level(level)

    x_index = int(np.floor(x / resolution))
    y_index = int(np.floor(y / resolution))

    return HierarchicalCell(
        level=level,
        x_index=x_index,
        y_index=y_index,
    )


def parent_cell(
    cell: HierarchicalCell,
) -> Optional[HierarchicalCell]:
    """
    Return the direct parent of a cell.

    Level 3 is the coarsest level and therefore has no parent.

    Example:

        Level 0 (5 cm)
              ↓
        Level 1 (10 cm)
              ↓
        Level 2 (20 cm)
              ↓
        Level 3 (40 cm)
    """

    if cell.level == max(LEVEL_RESOLUTIONS):
        return None

    parent_level = cell.level + 1

    parent_x_index = cell.x_index // 2
    parent_y_index = cell.y_index // 2

    return HierarchicalCell(
        level=parent_level,
        x_index=parent_x_index,
        y_index=parent_y_index,
    )


def child_cells(
    cell: HierarchicalCell,
) -> tuple[HierarchicalCell, ...]:
    """
    Return the four direct children of a cell.

    A child is one level FINER than its parent.

    Child order is deterministic:

        (0,0), (1,0), (0,1), (1,1)
    """

    if cell.level == min(LEVEL_RESOLUTIONS):
        return ()

    child_level = cell.level - 1

    base_x = cell.x_index * 2
    base_y = cell.y_index * 2

    return (
        HierarchicalCell(
            level=child_level,
            x_index=base_x,
            y_index=base_y,
        ),
        HierarchicalCell(
            level=child_level,
            x_index=base_x + 1,
            y_index=base_y,
        ),
        HierarchicalCell(
            level=child_level,
            x_index=base_x,
            y_index=base_y + 1,
        ),
        HierarchicalCell(
            level=child_level,
            x_index=base_x + 1,
            y_index=base_y + 1,
        ),
    )


def is_child_of(
    child: HierarchicalCell,
    parent: HierarchicalCell,
) -> bool:
    """
    Check whether `child` is the direct child of `parent`.
    """

    if child.level != parent.level - 1:
        return False

    return (
        child.x_index // 2 == parent.x_index
        and child.y_index // 2 == parent.y_index
    )


def cells_are_aligned(
    coarse: HierarchicalCell,
    fine: HierarchicalCell,
) -> bool:
    """
    Check whether a finer cell is spatially contained inside
    a coarser cell.

    The two cells may be separated by multiple hierarchy levels.
    """

    if fine.level >= coarse.level:
        return False

    current = fine

    while current.level < coarse.level:
        parent = parent_cell(current)

        if parent is None:
            return False

        current = parent

    return current == coarse


def hierarchy_cells_for_points(
    xyz: np.ndarray,
    level: int,
) -> list[HierarchicalCell]:
    """
    Convert an Nx3 point cloud into unique hierarchical cells.

    Z is ignored here because A9 is concerned with spatial indexing
    and alignment only.
    """

    xyz = np.asarray(xyz, dtype=np.float64)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(
            "xyz must have shape (N, 3)."
        )

    if not np.all(np.isfinite(xyz)):
        raise ValueError(
            "xyz must contain only finite values."
        )

    resolution_for_level(level)

    if len(xyz) == 0:
        return []

    cells = {
        cell_from_world(
            x=float(point[0]),
            y=float(point[1]),
            level=level,
        )
        for point in xyz
    }

    return sorted(
        cells,
        key=lambda cell: (
            cell.level,
            cell.x_index,
            cell.y_index,
        ),
    )