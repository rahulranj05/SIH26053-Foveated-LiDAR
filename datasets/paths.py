from pathlib import Path
import os


# ============================================================
# Exact Google Drive folder names
# ============================================================

DRIVE_DATASET_FOLDER_NAMES = {
    "semantic_stf": "SemanticSTF",
    "semantic_kitti": "SemanticKITTI",
    "rellis_3d": "RELLIS_3D",
    "nuscenes_mini": "nuScenes",
    "iith": "IITH",
}


def get_dataset_base() -> Path:
    """
    Return the top-level SIH26053 datasets directory.

    Priority:
    1. SIH26053_DATASETS environment variable
    2. Google Colab canonical path
    3. Windows Google Drive path
    """

    env_path = os.getenv("SIH26053_DATASETS")

    if env_path:
        root = Path(env_path)

        if root.exists():
            return root

        raise FileNotFoundError(
            f"SIH26053_DATASETS points to a missing path: {root}"
        )

    candidates = [
        Path(
            "/content/drive/MyDrive/SIH26053/datasets"
        ),
        Path(
            r"G:\My Drive\SIH26053\datasets"
        ),
    ]

    for root in candidates:
        if root.exists():
            return root

    raise FileNotFoundError(
        "Could not locate the SIH26053 datasets directory.\n"
        "Expected one of:\n"
        "  /content/drive/MyDrive/SIH26053/datasets\n"
        r"  G:\My Drive\SIH26053\datasets"
        "\n\n"
        "You can override this with the environment variable:\n"
        "SIH26053_DATASETS"
    )


def get_dataset_roots():
    """
    Return canonical loader roots.

    These are not always the top-level folders because some
    datasets were extracted into known subdirectories.
    """

    base = get_dataset_base()

    roots = {
        "semantic_stf": (
            base
            / DRIVE_DATASET_FOLDER_NAMES["semantic_stf"]
        ),

        "semantic_kitti": (
            base
            / DRIVE_DATASET_FOLDER_NAMES["semantic_kitti"]
            / "extracted"
            / "dataset"
        ),

        "rellis_3d": (
            base
            / DRIVE_DATASET_FOLDER_NAMES["rellis_3d"]
            / "extracted"
            / "Rellis-3D"
        ),

        "nuscenes_mini": (
            base
            / DRIVE_DATASET_FOLDER_NAMES["nuscenes_mini"]
        ),

        "iith": (
            base
            / DRIVE_DATASET_FOLDER_NAMES["iith"]
            / "extracted"
            / "IITH_LiDAR_ground_dataset_labelled_raw"
        ),
    }

    return roots


def validate_dataset_roots():
    """
    Verify that all expected project dataset roots exist.
    """

    roots = get_dataset_roots()

    missing = {
        name: path
        for name, path in roots.items()
        if not path.exists()
    }

    if missing:
        message = "\n".join(
            f"  {name}: {path}"
            for name, path in missing.items()
        )

        raise FileNotFoundError(
            "Missing dataset roots:\n"
            f"{message}"
        )

    return roots


if __name__ == "__main__":
    roots = validate_dataset_roots()

    print("=" * 80)
    print("SIH26053 DATASET ROOTS")
    print("=" * 80)

    for name, path in roots.items():
        print(f"{name:20s}: {path}")

    print()
    print("PASS: ALL DATASET ROOTS FOUND")