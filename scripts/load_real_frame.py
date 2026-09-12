import sys
from pathlib import Path

import numpy as np

# Add project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mapping.basic_grid import build_uniform_grid


def load_semantickitti_scan(path):
    """
    Load one SemanticKITTI Velodyne scan.

    Each point is stored as:
        x, y, z, intensity

    We only need XYZ for the current mapper.
    """
    data = np.fromfile(path, dtype=np.float32)

    if data.size % 4 != 0:
        raise ValueError(
            f"Invalid SemanticKITTI scan: {data.size} float values "
            "is not divisible by 4."
        )

    points = data.reshape(-1, 4)

    xyz = points[:, :3]

    return xyz


def main():
    scan_path = (
        PROJECT_ROOT
        / "datasets"
        / "test"
        / "000000.bin"
    )

    print("=" * 70)
    print("FOVEAMAP — REAL LIDAR FRAME")
    print("=" * 70)

    print(f"Scan: {scan_path}")

    if not scan_path.exists():
        raise FileNotFoundError(f"Scan not found: {scan_path}")

    xyz = load_semantickitti_scan(scan_path)

    print(f"Points loaded: {len(xyz):,}")
    print(f"XYZ shape: {xyz.shape}")

    print("\nXYZ ranges:")
    print(f"X: {xyz[:, 0].min():.2f} to {xyz[:, 0].max():.2f} m")
    print(f"Y: {xyz[:, 1].min():.2f} to {xyz[:, 1].max():.2f} m")
    print(f"Z: {xyz[:, 2].min():.2f} to {xyz[:, 2].max():.2f} m")

    grid = build_uniform_grid(
        xyz,
        resolution=0.10,
        x_range=(-50.0, 50.0),
        y_range=(-50.0, 50.0),
    )

    populated = grid.count > 0

    print("\nGrid:")
    print(f"Resolution: {grid.resolution:.2f} m")
    print(f"Grid size: {grid.width} x {grid.height}")
    print(f"Populated cells: {populated.sum():,}")

    if populated.any():
        print("\nElevation statistics:")
        print(
            f"Z min: {np.nanmin(grid.z_min):.2f} m"
        )
        print(
            f"Z max: {np.nanmax(grid.z_max):.2f} m"
        )
        print(
            f"Z mean: {np.nanmean(grid.z_mean):.2f} m"
        )

    print("\nA5 REAL FRAME INTEGRATION: PASS")


if __name__ == "__main__":
    main()