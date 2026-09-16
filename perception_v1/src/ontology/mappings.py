import json
from pathlib import Path

_MAPPING_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "SIH26053_NATIVE_TO_UNIFIED_MAPPING_v1.0.json"
)

with open(_MAPPING_PATH, "r", encoding="utf-8") as f:
    _DATA = json.load(f)

MAPPING_RECORDS = _DATA["mappings"]

NATIVE_TO_UNIFIED = {}

for record in MAPPING_RECORDS:
    dataset = record["dataset"]
    native_id = int(record["native_id"])
    unified_id = int(record["unified_id"])

    NATIVE_TO_UNIFIED.setdefault(dataset, {})
    NATIVE_TO_UNIFIED[dataset][native_id] = unified_id


def map_native_label(dataset: str, native_id: int) -> int:
    return NATIVE_TO_UNIFIED[dataset][int(native_id)]
