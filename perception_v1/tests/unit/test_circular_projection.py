import numpy as np

from perception_v1.src.perception.circular.projection import (
    PROJECTION_WIDTH,
    get_projection_config,
    project_range_view,
)


def xyz_from_angles(
    azimuth_deg,
    elevation_deg,
    radius=10.0,
):
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)

    horizontal = radius * np.cos(el)

    return np.array([
        horizontal * np.cos(az),
        horizontal * np.sin(az),
        radius * np.sin(el),
    ], dtype=np.float32)


def check_frozen_dimensions():
    expected = {
        "SemanticKITTI": (
            64,
            2048,
            -24.553390,
            4.550787,
        ),
        "RELLIS-3D": (
            64,
            2048,
            -16.439002,
            17.024001,
        ),
        "SemanticSTF": (
            64,
            2048,
            -22.832110,
            9.656203,
        ),
        "nuScenes": (
            32,
            2048,
            -71.348909,
            28.693044,
        ),
    }

    assert PROJECTION_WIDTH == 2048

    for dataset, values in expected.items():
        actual = get_projection_config(
            dataset
        )

        assert actual == values


def check_geometric_projection():
    xyz = np.stack([
        xyz_from_angles(0.0, 0.0),
        xyz_from_angles(90.0, 0.0),
        xyz_from_angles(-90.0, 0.0),
    ])

    result = project_range_view(
        xyz,
        dataset_id="SemanticKITTI",
    )

    assert np.all(result.in_fov_mask)

    # 0 degrees -> middle horizontal column.
    assert result.point_to_pixel[0, 1] == 1024

    # +90 -> 3/4 width.
    assert result.point_to_pixel[1, 1] == 1536

    # -90 -> 1/4 width.
    assert result.point_to_pixel[2, 1] == 512


def check_horizontal_circular_seam():
    xyz = np.stack([
        xyz_from_angles(-179.999, 0.0),
        xyz_from_angles(179.999, 0.0),
    ])

    result = project_range_view(
        xyz,
        dataset_id="SemanticKITTI",
    )

    left = result.point_to_pixel[0, 1]
    right = result.point_to_pixel[1, 1]

    assert left in (0, 1)
    assert right in (
        PROJECTION_WIDTH - 2,
        PROJECTION_WIDTH - 1,
    )


def check_vertical_fov_subset():
    xyz = np.stack([
        xyz_from_angles(0.0, 0.0),
        xyz_from_angles(0.0, 20.0),
        xyz_from_angles(0.0, -40.0),
    ])

    result = project_range_view(
        xyz,
        dataset_id="SemanticKITTI",
    )

    assert np.array_equal(
        result.in_fov_mask,
        np.array(
            [True, False, False]
        ),
    )

    assert np.all(
        result.point_to_pixel[1:] == -1
    )


def check_nearest_range_zbuffer():
    xyz = np.stack([
        xyz_from_angles(
            10.0,
            0.0,
            radius=20.0,
        ),
        xyz_from_angles(
            10.0,
            0.0,
            radius=5.0,
        ),
    ])

    source_ids = np.array(
        [100, 200],
        dtype=np.int64,
    )

    result = project_range_view(
        xyz,
        dataset_id="SemanticKITTI",
        source_point_id=source_ids,
    )

    assert np.array_equal(
        result.point_to_pixel[0],
        result.point_to_pixel[1],
    )

    row, col = result.point_to_pixel[0]

    assert (
        result.pixel_to_source_point[
            row, col
        ]
        == 200
    )

    assert np.array_equal(
        result.winner_mask,
        np.array([False, True]),
    )


def check_exact_tie_lowest_source_id():
    point = xyz_from_angles(
        20.0,
        0.0,
        radius=10.0,
    )

    # Identical geometry -> exact range/pixel tie.
    xyz = np.stack([
        point,
        point,
    ])

    source_ids = np.array(
        [50, 10],
        dtype=np.int64,
    )

    result = project_range_view(
        xyz,
        dataset_id="SemanticKITTI",
        source_point_id=source_ids,
    )

    row, col = result.point_to_pixel[0]

    assert (
        result.pixel_to_source_point[
            row, col
        ]
        == 10
    )

    assert np.array_equal(
        result.winner_mask,
        np.array([False, True]),
    )


def check_point_to_pixel_for_zbuffer_loser():
    point = xyz_from_angles(
        30.0,
        0.0,
    )

    xyz = np.stack([
        point * 2.0,
        point,
    ])

    result = project_range_view(
        xyz,
        dataset_id="SemanticKITTI",
    )

    # Both in-FOV points retain geometric pixel mapping.
    assert np.array_equal(
        result.point_to_pixel[0],
        result.point_to_pixel[1],
    )

    # But only nearest point owns the pixel.
    assert not result.winner_mask[0]
    assert result.winner_mask[1]


def check_deterministic_replay():
    rng = np.random.default_rng(26053)

    xyz = rng.normal(
        size=(2000, 3)
    ).astype(np.float32)

    # Keep points non-zero.
    xyz[:, 0] += 5.0

    ids = np.arange(
        len(xyz),
        dtype=np.int64,
    )

    a = project_range_view(
        xyz,
        dataset_id="SemanticKITTI",
        source_point_id=ids,
    )

    b = project_range_view(
        xyz,
        dataset_id="SemanticKITTI",
        source_point_id=ids,
    )

    assert np.array_equal(
        a.point_to_pixel,
        b.point_to_pixel,
    )
    assert np.array_equal(
        a.pixel_to_source_point,
        b.pixel_to_source_point,
    )
    assert np.array_equal(
        a.in_fov_mask,
        b.in_fov_mask,
    )
    assert np.array_equal(
        a.winner_mask,
        b.winner_mask,
    )


def check_empty_frame():
    result = project_range_view(
        np.empty(
            (0, 3),
            dtype=np.float32,
        ),
        dataset_id="nuScenes",
    )

    assert result.height == 32
    assert result.width == 2048
    assert result.point_to_pixel.shape == (0, 2)
    assert result.in_fov_mask.shape == (0,)
    assert result.winner_mask.shape == (0,)
    assert result.pixel_to_source_point.shape == (
        32,
        2048,
    )
    assert np.all(
        result.pixel_to_source_point == -1
    )


def check_invalid_inputs():
    try:
        project_range_view(
            np.zeros(
                (1, 3),
                dtype=np.float32,
            ),
            dataset_id="SemanticKITTI",
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Zero XYZ was accepted"
        )

    try:
        project_range_view(
            np.array(
                [[1.0, np.nan, 2.0]],
                dtype=np.float32,
            ),
            dataset_id="SemanticKITTI",
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Non-finite XYZ was accepted"
        )

    try:
        project_range_view(
            np.array(
                [[1.0, 0.0, 0.0]],
                dtype=np.float32,
            ),
            dataset_id="UNKNOWN",
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Unknown dataset was accepted"
        )


def main():
    print("M5 CIRCULAR PROJECTION UNIT GATE")
    print("=" * 50)

    check_frozen_dimensions()
    print("[PASS] Frozen dataset projection configs")

    check_geometric_projection()
    print("[PASS] Geometric azimuth projection")

    check_horizontal_circular_seam()
    print("[PASS] Horizontal circular seam")

    check_vertical_fov_subset()
    print("[PASS] Dataset vertical-FOV subset")

    check_nearest_range_zbuffer()
    print("[PASS] Nearest-range z-buffer")

    check_exact_tie_lowest_source_id()
    print("[PASS] Lowest-source-ID exact tie-break")

    check_point_to_pixel_for_zbuffer_loser()
    print("[PASS] Z-buffer loser mapping retained")

    check_deterministic_replay()
    print("[PASS] Deterministic projection replay")

    check_empty_frame()
    print("[PASS] Empty frame handling")

    check_invalid_inputs()
    print("[PASS] Invalid projection inputs rejected")

    print("=" * 50)
    print("M5 CIRCULAR PROJECTION UNIT GATE: PASS")


if __name__ == "__main__":
    main()