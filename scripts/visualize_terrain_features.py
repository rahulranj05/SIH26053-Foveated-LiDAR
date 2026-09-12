import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mapping.basic_grid import build_uniform_grid
from mapping.terrain_features import (
    compute_height_span,
    compute_height_discontinuity,
    compute_roughness,
    compute_slope,
)


# ============================================================
# CONFIG
# ============================================================

SCAN_PATH = PROJECT_ROOT / "datasets" / "test" / "000000.bin"

RESOLUTION = 0.10

# Same mapping window used during A5.
X_RANGE = (-50.0, 50.0)
Y_RANGE = (-50.0, 50.0)


# ============================================================
# LOAD SEMANTICKITTI FRAME
# ============================================================

print("=" * 70)
print("FOVEAMAP — REAL TERRAIN FEATURE INSPECTION")
print("=" * 70)

print(f"\nScan: {SCAN_PATH}")

raw = np.fromfile(SCAN_PATH, dtype=np.float32)

if raw.size % 4 != 0:
    raise ValueError(
        f"Invalid SemanticKITTI file: {raw.size} float values "
        "cannot be reshaped into XYZ + intensity."
    )

points = raw.reshape(-1, 4)

xyz = points[:, :3].astype(np.float64)

print(f"Points loaded: {len(xyz):,}")
print(f"XYZ shape: {xyz.shape}")

print("\nXYZ ranges:")
print(f"X: {xyz[:, 0].min():.2f} to {xyz[:, 0].max():.2f} m")
print(f"Y: {xyz[:, 1].min():.2f} to {xyz[:, 1].max():.2f} m")
print(f"Z: {xyz[:, 2].min():.2f} to {xyz[:, 2].max():.2f} m")


# ============================================================
# BUILD UNIFORM 2.5D GRID
# ============================================================

grid = build_uniform_grid(
    xyz,
    resolution=RESOLUTION,
    x_range=X_RANGE,
    y_range=Y_RANGE,
)

print("\nGrid:")
print(f"Resolution: {grid.resolution:.2f} m")
print(f"Grid size: {grid.height} x {grid.width}")
print(f"Populated cells: {(grid.count > 0).sum():,}")


# ============================================================
# COMPUTE TERRAIN FEATURES
# ============================================================

height_span = compute_height_span(grid)
slope = compute_slope(grid)
roughness = compute_roughness(grid)
discontinuity = compute_height_discontinuity(grid)


# ============================================================
# PRINT STATISTICS
# ============================================================

def print_feature_stats(name, values):
    valid = np.isfinite(values)

    print(f"\n{name}:")
    print(f"  Valid cells: {valid.sum():,}")
    print(f"  Minimum: {np.nanmin(values):.4f}")
    print(f"  Maximum: {np.nanmax(values):.4f}")
    print(f"  Mean:    {np.nanmean(values):.4f}")


print_feature_stats("Height span (m)", height_span)
print_feature_stats("Slope (degrees)", slope)
print_feature_stats("Roughness (m)", roughness)
print_feature_stats("Height discontinuity (m)", discontinuity)


# ============================================================
# VISUALIZATION
# ============================================================

features = [
    ("Elevation", grid.z_mean, "Elevation (m)"),
    ("Height Span", height_span, "Height Span (m)"),
    ("Slope", slope, "Slope (degrees)"),
    ("Roughness", roughness, "Roughness (m)"),
    ("Height Discontinuity", discontinuity, "Discontinuity (m)"),
]

fig, axes = plt.subplots(
    2,
    3,
    figsize=(18, 10),
)

axes = axes.flatten()

extent = [
    X_RANGE[0],
    X_RANGE[1],
    Y_RANGE[0],
    Y_RANGE[1],
]

for ax, (title, data, colorbar_label) in zip(axes, features):

    image = ax.imshow(
        data,
        origin="lower",
        extent=extent,
        aspect="equal",
    )

    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)


# Remove unused sixth subplot.
axes[-1].axis("off")

fig.suptitle(
    "FoveaMap — SemanticKITTI 000000.bin Terrain Features",
    fontsize=16,
)

plt.tight_layout()

output_path = PROJECT_ROOT / "terrain_features_semantickitti.png"

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight",
)

print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE")
print("=" * 70)
print(f"Saved to: {output_path}")

plt.show()