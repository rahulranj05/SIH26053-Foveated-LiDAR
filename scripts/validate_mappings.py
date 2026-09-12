from pathlib import Path

import yaml


ONTOLOGY_FILE = Path(
    "configs/ontology.yaml"
)

MAPPING_FILES = [
    Path(
        "datasets/mappings/"
        "semantic_kitti.yaml"
    ),
    Path(
        "datasets/mappings/"
        "rellis.yaml"
    ),
    Path(
        "datasets/mappings/"
        "semantic_stf.yaml"
    ),
    Path(
        "datasets/mappings/"
        "nuscenes.yaml"
    ),
]


def load_yaml(
    path: Path,
):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid YAML document: {path}"
        )

    return data


def main():
    print("=" * 80)
    print(
        "UNIFIED ONTOLOGY / DATASET "
        "MAPPING VALIDATION"
    )
    print("=" * 80)

    ontology = load_yaml(
        ONTOLOGY_FILE
    )

    classes = ontology.get(
        "classes",
        {}
    )

    if not classes:
        raise RuntimeError(
            "Ontology contains no classes."
        )

    ontology_ids = {
        int(class_id)
        for class_id in classes
    }

    expected_ids = set(
        range(23)
    )

    if ontology_ids != expected_ids:
        raise RuntimeError(
            "Ontology IDs must be exactly 0–22.\n"
            f"Observed: {sorted(ontology_ids)}"
        )

    if (
        classes[0]["name"]
        != "ignore"
    ):
        raise RuntimeError(
            "Ontology class 0 must be 'ignore'."
        )

    print(
        "Ontology IDs: 0–22"
    )
    print(
        "Ontology classes: 23"
    )
    print()

    failures = []

    for mapping_file in MAPPING_FILES:
        print("-" * 80)
        print(mapping_file)

        config = load_yaml(
            mapping_file
        )

        mapping = config.get(
            "mapping"
        )

        if not mapping:
            failures.append(
                f"{mapping_file}: empty mapping"
            )
            continue

        ignore_label = config.get(
            "ignore_label"
        )

        if ignore_label != 0:
            failures.append(
                f"{mapping_file}: "
                "ignore_label must be 0"
            )

        native_ids = {
            int(native)
            for native in mapping
        }

        unified_ids = {
            int(unified)
            for unified in mapping.values()
        }

        invalid_targets = (
            unified_ids
            - ontology_ids
        )

        if invalid_targets:
            failures.append(
                f"{mapping_file}: invalid targets "
                f"{sorted(invalid_targets)}"
            )

        print(
            f"Native IDs mapped : {len(native_ids)}"
        )

        print(
            "Unified IDs used  : "
            f"{sorted(unified_ids)}"
        )

        if not invalid_targets:
            print(
                "PASS: mapping targets are valid"
            )

    print()
    print("=" * 80)

    if failures:
        print("FAILURES:")

        for failure in failures:
            print(
                f"  - {failure}"
            )

        raise RuntimeError(
            f"{len(failures)} mapping "
            "validation problem(s)"
        )

    print(
        "PASS: ALL ONTOLOGY AND "
        "MAPPING CHECKS PASSED"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()