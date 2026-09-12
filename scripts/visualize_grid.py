import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mapping.basic_grid import build_uniform_grid

def create_synthetic_terrain():
    """Create a small synthetic LiDAR point cloud."""

    rng = np.random.default_rng(42)

    # Random XY points
    n_points = 12000

    x = rng.uniform(-10.0, 10.0, n_points)
    y = rng.uniform(-10.0, 10.0, n_points)

    # Base terrain
    z = np.zeros(n_points)

    # Gentle slope on the right side
    slope_region = x > 2.0
    z[slope_region] = 0.12 * (x[slope_region] - 2.0)

    # Raised obstacle
    obstacle = (
        (x > -4.0)
        & (x < -1.0)
        & (y > 2.0)
        & (y < 5.0)
    )
    z[obstacle] += 1.5

    # Small amount of measurement noise
    z += rng.normal(0.0, 0.02, n_points)

    return np.column_stack((x, y, z))


def main():
    # ---------------------------------------------------------
    # 1. Generate synthetic LiDAR
    # ---------------------------------------------------------
    xyz = create_synthetic_terrain()

    print(f"Generated points: {len(xyz):,}")

    # ---------------------------------------------------------
    # 2. Build our actual uniform 2.5D grid
    # ---------------------------------------------------------
    grid = build_uniform_grid(
        xyz,
        resolution=0.10,
        x_range=(-10.0, 10.0),
        y_range=(-10.0, 10.0),
    )

    print(f"Grid size: {grid.width} x {grid.height}")
    print(f"Resolution: {grid.resolution:.2f} m")
    print(f"Populated cells: {(grid.count > 0).sum():,}")

    # ---------------------------------------------------------
    # 3. Plot elevation
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 8))

    plt.imshow(
        grid.z_mean,
        origin="lower",
        extent=[
            grid.x_range[0],
            grid.x_range[1],
            grid.y_range[0],
            grid.y_range[1],
        ],
        interpolation="nearest",
        aspect="equal",
    )

    plt.colorbar(label="Mean elevation (m)")

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("FoveaMap — Uniform 2.5D Elevation Grid")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()