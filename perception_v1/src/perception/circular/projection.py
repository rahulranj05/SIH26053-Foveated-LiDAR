from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PROJECTION_WIDTH = 2048

PROJECTION_CONFIGS = {
    "SemanticKITTI": {
        "height": 64,
        "fov_min_deg": -24.553390,
        "fov_max_deg": 4.550787,
    },
    "RELLIS-3D": {
        "height": 64,
        "fov_min_deg": -16.439002,
        "fov_max_deg": 17.024001,
    },
    "SemanticSTF": {
        "height": 64,
        "fov_min_deg": -22.832110,
        "fov_max_deg": 9.656203,
    },
    "nuScenes": {
        "height": 32,
        "fov_min_deg": -71.348909,
        "fov_max_deg": 28.693044,
    },
}


@dataclass(frozen=True)
class RangeProjection:
    height: int
    width: int
    point_to_pixel: np.ndarray
    pixel_to_source_point: np.ndarray
    in_fov_mask: np.ndarray
    winner_mask: np.ndarray


def get_projection_config(
    dataset_id: str,
) -> tuple[int, int, float, float]:
    if dataset_id not in PROJECTION_CONFIGS:
        raise ValueError(
            f"Unknown projection dataset: {dataset_id}"
        )

    config = PROJECTION_CONFIGS[dataset_id]

    return (
        int(config["height"]),
        PROJECTION_WIDTH,
        float(config["fov_min_deg"]),
        float(config["fov_max_deg"]),
    )


def project_range_view(
    xyz: np.ndarray,
    *,
    dataset_id: str,
    source_point_id: np.ndarray | None = None,
) -> RangeProjection:
    """
    Frozen S7-D geometric circular range projection.

    Rules:
    - projection is derived geometrically from XYZ
    - horizontal axis is circular over [-pi, pi)
    - dataset-specific frozen vertical FOV
    - nearest-range point wins each pixel
    - exact range tie -> lowest source point ID
    - point_to_pixel retained for all in-FOV points
    - pixel_to_source_point stores only z-buffer winners
    """

    xyz = np.asarray(xyz, dtype=np.float32)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(
            f"xyz must have shape (N, 3), got {xyz.shape}"
        )

    if not np.all(np.isfinite(xyz)):
        raise ValueError(
            "Range projection requires finite canonical XYZ"
        )

    n = len(xyz)

    if source_point_id is None:
        source_point_id = np.arange(
            n,
            dtype=np.int64,
        )
    else:
        source_point_id = np.asarray(
            source_point_id,
            dtype=np.int64,
        )

        if source_point_id.ndim != 1:
            raise ValueError(
                "source_point_id must have shape (N,)"
            )

        if len(source_point_id) != n:
            raise ValueError(
                "XYZ/source_point_id length mismatch"
            )

        if len(np.unique(source_point_id)) != n:
            raise ValueError(
                "source_point_id must be unique"
            )

    height, width, fov_min_deg, fov_max_deg = (
        get_projection_config(dataset_id)
    )

    point_to_pixel = np.full(
        (n, 2),
        -1,
        dtype=np.int64,
    )

    in_fov_mask = np.zeros(
        n,
        dtype=bool,
    )

    winner_mask = np.zeros(
        n,
        dtype=bool,
    )

    pixel_to_source_point = np.full(
        (height, width),
        -1,
        dtype=np.int64,
    )

    if n == 0:
        return RangeProjection(
            height=height,
            width=width,
            point_to_pixel=point_to_pixel,
            pixel_to_source_point=pixel_to_source_point,
            in_fov_mask=in_fov_mask,
            winner_mask=winner_mask,
        )

    xyz64 = xyz.astype(
        np.float64,
        copy=False,
    )

    x = xyz64[:, 0]
    y = xyz64[:, 1]
    z = xyz64[:, 2]

    ranges = np.sqrt(
        x * x + y * y + z * z
    )

    if np.any(ranges <= 0.0):
        raise ValueError(
            "Range projection requires non-zero canonical XYZ"
        )

    azimuth = np.arctan2(y, x)

    horizontal_range = np.sqrt(
        x * x + y * y
    )

    elevation = np.arctan2(
        z,
        horizontal_range,
    )

    fov_min = np.deg2rad(fov_min_deg)
    fov_max = np.deg2rad(fov_max_deg)

    in_fov_mask = (
        (elevation >= fov_min)
        & (elevation <= fov_max)
    )

    # Horizontal circular mapping:
    # [-pi, pi) -> [0, width)
    u_float = (
        (azimuth + np.pi)
        / (2.0 * np.pi)
        * width
    )

    u = np.floor(u_float).astype(
        np.int64
    )

    # Explicit modulo makes horizontal circularity robust
    # at the -pi/+pi seam.
    u = np.mod(u, width)

    # Vertical mapping:
    # maximum elevation -> row 0
    # minimum elevation -> row height-1
    v_float = (
        (fov_max - elevation)
        / (fov_max - fov_min)
        * height
    )

    v = np.floor(v_float).astype(
        np.int64
    )

    # Exact lower FOV boundary maps to the last row.
    v = np.clip(
        v,
        0,
        height - 1,
    )

    valid_indices = np.flatnonzero(
        in_fov_mask
    )

    point_to_pixel[
        valid_indices, 0
    ] = v[valid_indices]

    point_to_pixel[
        valid_indices, 1
    ] = u[valid_indices]

    # Z-buffer state is indexed by flattened pixel.
    best_point_index = np.full(
        height * width,
        -1,
        dtype=np.int64,
    )

    best_range = np.full(
        height * width,
        np.inf,
        dtype=np.float64,
    )

    best_source_id = np.full(
        height * width,
        np.iinfo(np.int64).max,
        dtype=np.int64,
    )

    # Deterministic point-order-independent comparison:
    # 1. minimum range
    # 2. exact tie -> minimum source_point_id
    for point_index in valid_indices:
        row = v[point_index]
        col = u[point_index]
        pixel = row * width + col

        point_range = ranges[point_index]
        point_source_id = source_point_id[
            point_index
        ]

        replace = False

        if point_range < best_range[pixel]:
            replace = True
        elif (
            point_range == best_range[pixel]
            and point_source_id
            < best_source_id[pixel]
        ):
            replace = True

        if replace:
            best_point_index[pixel] = point_index
            best_range[pixel] = point_range
            best_source_id[pixel] = point_source_id

    occupied = np.flatnonzero(
        best_point_index >= 0
    )

    for pixel in occupied:
        point_index = best_point_index[pixel]

        row = pixel // width
        col = pixel % width

        pixel_to_source_point[
            row, col
        ] = source_point_id[point_index]

        winner_mask[point_index] = True

    return RangeProjection(
        height=height,
        width=width,
        point_to_pixel=point_to_pixel,
        pixel_to_source_point=pixel_to_source_point,
        in_fov_mask=in_fov_mask,
        winner_mask=winner_mask,
    )