"""
F1 24-class semantic ontology.

This module is intentionally isolated from the live CARLA/FoveaMap pipeline.

The authoritative names of F1 unified classes 1..24 and their mapping to
FoveaMap classes 0..5 have not yet been recovered. Do NOT invent values here.

Once the authoritative taxonomy is recovered, populate:
    F1_CLASS_NAMES
    F1_TO_FOVEAMAP
"""

from __future__ import annotations

from typing import Final


# F1 unified semantic IDs.
# Model predictions are 0..23; unified IDs are 1..24.
F1_CLASS_IDS: Final[tuple[int, ...]] = tuple(range(1, 25))


# FoveaMap semantic IDs.
FOVEAMAP_UNKNOWN: Final[int] = 0
FOVEAMAP_DRIVABLE: Final[int] = 1
FOVEAMAP_NON_DRIVABLE: Final[int] = 2
FOVEAMAP_STATIC_OBSTACLE: Final[int] = 3
FOVEAMAP_VEHICLE: Final[int] = 4
FOVEAMAP_VULNERABLE_USER: Final[int] = 5


FOVEAMAP_CLASS_IDS: Final[tuple[int, ...]] = (
    FOVEAMAP_UNKNOWN,
    FOVEAMAP_DRIVABLE,
    FOVEAMAP_NON_DRIVABLE,
    FOVEAMAP_STATIC_OBSTACLE,
    FOVEAMAP_VEHICLE,
    FOVEAMAP_VULNERABLE_USER,
)


# Authoritative F1 class names are still unresolved.
# Keep every ID explicit so missing entries cannot be overlooked.
F1_CLASS_NAMES: dict[int, str | None] = {
    1: "paved_drivable",
    2: "sidewalk",
    3: "parking",
    4: "other_ground",
    5: "natural_terrain",
    6: "mud",
    7: "rubble",
    8: "water",
    9: "low_vegetation",
    10: "vegetation",
    11: "trunk",
    12: "building",
    13: "fence",
    14: "other_structure",
    15: "barrier",
    16: "pole",
    17: "traffic_sign",
    18: "car",
    19: "heavy_vehicle",
    20: "other_vehicle",
    21: "bicycle",
    22: "motorcycle",
    23: "person",
    24: "rider",
}


# PROPOSED engineering abstraction only.
#
# This is NOT an authoritative historical mapping recovered from the
# original F1 training specification. It is kept separate from the live
# F1_TO_FOVEAMAP mapping until reviewed and approved.
PROPOSED_F1_TO_FOVEAMAP: dict[int, int] = {
    # Ground / terrain
    1: FOVEAMAP_DRIVABLE,          # paved_drivable
    2: FOVEAMAP_NON_DRIVABLE,      # sidewalk
    3: FOVEAMAP_NON_DRIVABLE,      # parking
    4: FOVEAMAP_NON_DRIVABLE,      # other_ground
    5: FOVEAMAP_NON_DRIVABLE,      # natural_terrain
    6: FOVEAMAP_NON_DRIVABLE,      # mud
    7: FOVEAMAP_STATIC_OBSTACLE,   # rubble
    8: FOVEAMAP_STATIC_OBSTACLE,   # water

    # Vegetation
    9: FOVEAMAP_STATIC_OBSTACLE,   # low_vegetation
    10: FOVEAMAP_STATIC_OBSTACLE,  # vegetation
    11: FOVEAMAP_STATIC_OBSTACLE,  # trunk

    # Structures / obstacles
    12: FOVEAMAP_STATIC_OBSTACLE,  # building
    13: FOVEAMAP_STATIC_OBSTACLE,  # fence
    14: FOVEAMAP_STATIC_OBSTACLE,  # other_structure
    15: FOVEAMAP_STATIC_OBSTACLE,  # barrier
    16: FOVEAMAP_STATIC_OBSTACLE,  # pole
    17: FOVEAMAP_STATIC_OBSTACLE,  # traffic_sign

    # Vehicles
    18: FOVEAMAP_VEHICLE,           # car
    19: FOVEAMAP_VEHICLE,           # heavy_vehicle
    20: FOVEAMAP_VEHICLE,           # other_vehicle
    21: FOVEAMAP_VEHICLE,           # bicycle
    22: FOVEAMAP_VEHICLE,           # motorcycle

    # Humans
    23: FOVEAMAP_VULNERABLE_USER,   # person
    24: FOVEAMAP_VULNERABLE_USER,   # rider
}


# Live mapping remains empty until the proposed mapping is formally reviewed.
F1_TO_FOVEAMAP: dict[int, int] = {}


def validate_ontology() -> None:
    """Validate the ontology structure without requiring resolved names."""
    if tuple(F1_CLASS_NAMES.keys()) != F1_CLASS_IDS:
        raise ValueError("F1_CLASS_NAMES must contain exactly IDs 1..24.")

    if not set(F1_TO_FOVEAMAP).issubset(F1_CLASS_IDS):
        raise ValueError("F1_TO_FOVEAMAP contains an invalid F1 class ID.")

    invalid_targets = set(F1_TO_FOVEAMAP.values()) - set(FOVEAMAP_CLASS_IDS)
    if invalid_targets:
        raise ValueError(
            f"F1_TO_FOVEAMAP contains invalid FoveaMap IDs: {sorted(invalid_targets)}"
        )


def unresolved_f1_classes() -> tuple[int, ...]:
    """Return F1 IDs whose authoritative names have not been recovered."""
    return tuple(
        class_id
        for class_id in F1_CLASS_IDS
        if F1_CLASS_NAMES[class_id] is None
    )


def unresolved_mappings() -> tuple[int, ...]:
    """Return F1 IDs whose FoveaMap mapping has not been verified."""
    return tuple(
        class_id
        for class_id in F1_CLASS_IDS
        if class_id not in F1_TO_FOVEAMAP
    )


validate_ontology()
