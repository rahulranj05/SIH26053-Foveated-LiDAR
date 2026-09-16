CLASS_NAMES = {
    0: "ignore",
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

IGNORE_LABEL = 0
NUM_TRAINABLE_CLASSES = 24
TRAINABLE_LABELS = tuple(range(1, 25))