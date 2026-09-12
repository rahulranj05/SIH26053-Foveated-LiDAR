from pathlib import Path

import yaml


ONTOLOGY_FILE = Path(
    "configs/ontology.yaml"
)

MAPPING_FILES = [
    Path("datasets/mappings/semantic_kitti.yaml"),
    Path("datasets/mappings/rellis.yaml"),
    Path("datasets/mappings/semantic_stf.yaml"),
    Path("datasets/mappings/nuscenes.yaml"),
]


def load_yaml(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def main():
    print("=" * 70)
    print("UNIFIED ONTOLOGY / DATASET MAPPING VALIDATION")
    print("=" * 70)

    ontology = load_yaml(
        ONTOLOGY_FILE
    )

    classes = ontology.get(
        "classes",
        {}
    )

    ontology_ids = {
        int(class_id)
        for class_id in classes.keys()
    }

    if not ontology_ids:
        raise RuntimeError(
            "Ontology contains no classes."
        )

    print(
        f"Ontology classes: {len(ontology_ids)}"
    )

    print(
        f"Ontology IDs: {sorted(ontology_ids)}"
    )

    print()

    failures = 0

    for mapping_file in MAPPING_FILES:
        print("-" * 70)
        print(mapping_file)

        config = load_yaml(
            mapping_file
        )

        mapping = config.get(
            "mapping",
            {}
        )

        if not mapping:
            print("FAIL: mapping is empty")
            failures += 1
            continue

        native_ids = {
            int(native)
            for native in mapping.keys()
        }

        unified_ids = {
            int(unified)
            for unified in mapping.values()
        }

        invalid_targets = (
            unified_ids
            - ontology_ids
        )

        print(
            f"Native IDs mapped:  {len(native_ids)}"
        )

        print(
            f"Unified IDs used:   {sorted(unified_ids)}"
        )

        if invalid_targets:
            print(
                "FAIL: invalid ontology IDs:",
                sorted(invalid_targets),
            )

            failures += 1

        else:
            print(
                "PASS: all targets exist "
                "in unified ontology"
            )

    print()
    print("=" * 70)

    if failures:
        raise RuntimeError(
            f"Mapping validation failed: "
            f"{failures} problem(s)"
        )

    print(
        "PASS: ALL DATASET MAPPINGS ARE VALID"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()