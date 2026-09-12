import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------
# Add project root to Python path
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mapping.basic_grid import build_uniform_grid
from mapping.traversability import compute_traversability


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

BIN_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "test"
    / "000000.bin"
)

RESOLUTION = 0.10

X_RANGE = (-50.0, 50.0)
Y_RANGE = (-50.0, 50.0)


# ------------------------------------------------------------
# Load SemanticKITTI LiDAR frame
# ------------------------------------------------------------

print("=" * 70)
print("FOVEAMAP — REAL LIDAR TRAVERSABILITY")
print("=" * 70)

print(f"Scan: {BIN_FILE}")

if not BIN_FILE.exists():
    raise FileNotFoundError(
        f"LiDAR file not found:\n{BIN_FILE}"
    )

raw = np.fromfile(
    BIN_FILE,
    dtype=np.float32,
)

if raw.size % 4 != 0:
    raise ValueError(
        "LiDAR file does not contain a multiple of 4 float32 values."
    )

points = raw.reshape(-1, 4)

xyz = points[:, :3]

print(f"Points loaded: {len(xyz):,}")
print(f"XYZ shape: {xyz.shape}")

# ------------------------------------------------------------
# Build uniform 2.5D grid
# ------------------------------------------------------------

grid = build_uniform_grid(
    xyz,
    resolution=RESOLUTION,
    x_range=X_RANGE,
    y_range=Y_RANGE,
)

print()
print("Grid:")
print(f"  Resolution: {grid.resolution:.2f} m")
print(f"  Grid size: {grid.height} x {grid.width}")

populated = grid.count > 0

print(
    f"  Populated cells: "
    f"{np.count_nonzero(populated):,}"
)

print(
    f"  Empty cells: "
    f"{np.count_nonzero(~populated):,}"
)

# ------------------------------------------------------------
# Compute geometry-only traversability
# ------------------------------------------------------------

traversability = compute_traversability(
    grid
)

valid = np.isfinite(traversability)

if not np.any(valid):
    raise RuntimeError(
        "No valid traversability cells were produced."
    )

valid_values = traversability[valid]

print()
print("Traversability:")
print(
    f"  Valid cells: "
    f"{len(valid_values):,}"
)

print(
    f"  Minimum: "
    f"{np.min(valid_values):.4f}"
)

print(
    f"  Maximum: "
    f"{np.max(valid_values):.4f}"
)

print(
    f"  Mean: "
    f"{np.mean(valid_values):.4f}"
)

print(
    f"  Median: "
    f"{np.median(valid_values):.4f}"
)

print(
    f"  10th percentile: "
    f"{np.percentile(valid_values, 10):.4f}"
)

print(
    f"  90th percentile: "
    f"{np.percentile(valid_values, 90):.4f}"
)

# ------------------------------------------------------------
# Categorize traversability
# ------------------------------------------------------------

high = valid_values >= 0.75
medium = (
    (valid_values >= 0.40)
    & (valid_values < 0.75)
)
low = valid_values < 0.40

print()
print("Traversability distribution:")

print(
    f"  High   (>= 0.75): "
    f"{np.count_nonzero(high):,} "
    f"({100.0 * np.mean(high):.2f}%)"
)

print(
    f"  Medium (0.40-0.75): "
    f"{np.count_nonzero(medium):,} "
    f"({100.0 * np.mean(medium):.2f}%)"
)

print(
    f"  Low    (< 0.40): "
    f"{np.count_nonzero(low):,} "
    f"({100.0 * np.mean(low):.2f}%)"
)

# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------

display_map = np.ma.masked_invalid(
    traversability
)

plt.figure(
    figsize=(10, 8)
)

image = plt.imshow(
    display_map,
    origin="lower",
    extent=[
        X_RANGE[0],
        X_RANGE[1],
        Y_RANGE[0],
        Y_RANGE[1],
    ],
    vmin=0.0,
    vmax=1.0,
    cmap="RdYlGn",
)

plt.colorbar(
    image,
    label="Traversability (0 = unsafe, 1 = safe)",
)

plt.xlabel("X (m)")
plt.ylabel("Y (m)")

plt.title(
    "FoveaMap — SemanticKITTI 000000\n"
    "Geometry-Only Traversability"
)

plt.tight_layout()

output_file = (
    PROJECT_ROOT
    / "traversability_semantickitti.png"
)

plt.savefig(
    output_file,
    dpi=200,
    bbox_inches="tight",
)

plt.show()

print()
print(f"Saved visualization: {output_file}")

print()
print("=" * 70)
print("A7 REAL-FRAME TRAVERSABILITY: COMPLETE")
print("=" * 70)