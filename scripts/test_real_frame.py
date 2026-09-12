import sys
from pathlib import Path

import numpy as np

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mapping.basic_grid import build_uniform_grid


def main():
    # Temporary test path.
    # We will replace this with an actual dataset frame.
    scan_path = Path(
        "/content/drive/MyDrive/SIH26053 (1)/datasets/SemanticKITTI"
    )

    print("=" * 70)
    print("FOVEAMAP — REAL LIDAR FRAME INTEGRATION")
    print("=" * 70)

    print(f"Dataset path: {scan_path}")

    if not scan_path.exists():
        print("\n[ERROR] Dataset path does not exist on this machine.")
        print("This path is the Google Colab/Drive path.")
        print("We need a local LiDAR frame for this Mac test.")
        return

    print("\nDataset path found.")


if __name__ == "__main__":
    main()