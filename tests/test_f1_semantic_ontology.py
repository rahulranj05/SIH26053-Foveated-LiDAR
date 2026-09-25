from perception.f1_semantic_ontology import (
    F1_CLASS_IDS,
    F1_CLASS_NAMES,
    F1_TO_FOVEAMAP,
    PROPOSED_F1_TO_FOVEAMAP,
    FOVEAMAP_CLASS_IDS,
    unresolved_f1_classes,
    unresolved_mappings,
)


def test_f1_has_exactly_24_classes():
    assert F1_CLASS_IDS == tuple(range(1, 25))


def test_f1_class_names_are_complete():
    assert len(F1_CLASS_NAMES) == 24
    assert all(F1_CLASS_NAMES[class_id] for class_id in F1_CLASS_IDS)
    assert unresolved_f1_classes() == ()


def test_proposed_mapping_covers_all_f1_classes():
    assert set(PROPOSED_F1_TO_FOVEAMAP) == set(F1_CLASS_IDS)
    assert set(PROPOSED_F1_TO_FOVEAMAP.values()).issubset(
        set(FOVEAMAP_CLASS_IDS)
    )


def test_live_mapping_is_not_activated():
    assert F1_TO_FOVEAMAP == {}
    assert len(unresolved_mappings()) == 24
