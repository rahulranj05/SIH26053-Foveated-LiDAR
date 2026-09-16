import copy

from perception_v1.src.data.sampling import (
    DatasetBalancedSampler,
)


def make_frames():
    return {
        "SemanticKITTI": tuple(
            f"sk-{i}"
            for i in range(100)
        ),
        "RELLIS-3D": tuple(
            f"rellis-{i}"
            for i in range(5)
        ),
        "SemanticSTF": tuple(
            f"stf-{i}"
            for i in range(20)
        ),
        "nuScenes": tuple(
            f"nusc-{i}"
            for i in range(2)
        ),
    }


def check_seeded_replay():
    frames = make_frames()

    a = DatasetBalancedSampler(
        frames,
        seed=26053,
    )

    b = DatasetBalancedSampler(
        frames,
        seed=26053,
    )

    assert (
        a.sample_many(500)
        == b.sample_many(500)
    )


def check_different_seed():
    frames = make_frames()

    a = DatasetBalancedSampler(
        frames,
        seed=26053,
    )

    b = DatasetBalancedSampler(
        frames,
        seed=26054,
    )

    assert (
        a.sample_many(100)
        != b.sample_many(100)
    )


def check_dictionary_order_independence():
    frames = make_frames()

    reversed_frames = dict(
        reversed(
            list(frames.items())
        )
    )

    a = DatasetBalancedSampler(
        frames,
        seed=26053,
    )

    b = DatasetBalancedSampler(
        reversed_frames,
        seed=26053,
    )

    assert (
        a.sample_many(500)
        == b.sample_many(500)
    )


def check_dataset_balance():
    sampler = DatasetBalancedSampler(
        make_frames(),
        seed=26053,
    )

    draws = sampler.sample_many(
        40000
    )

    counts = {
        dataset_id: 0
        for dataset_id
        in sampler.dataset_ids
    }

    for sample in draws:
        counts[
            sample.dataset_id
        ] += 1

    expected = 10000

    # Wide deterministic falsification bound.
    # This is NOT a statistical training guarantee.
    for count in counts.values():
        assert abs(
            count - expected
        ) < 700


def check_small_dataset_not_size_suppressed():
    sampler = DatasetBalancedSampler(
        make_frames(),
        seed=26053,
    )

    draws = sampler.sample_many(
        20000
    )

    counts = {}

    for sample in draws:
        counts[sample.dataset_id] = (
            counts.get(
                sample.dataset_id,
                0,
            )
            + 1
        )

    # nuScenes has only 2 synthetic frames while
    # SemanticKITTI has 100. Dataset selection must
    # nevertheless remain approximately equal.
    difference = abs(
        counts["nuScenes"]
        - counts["SemanticKITTI"]
    )

    assert difference < 600


def check_frame_membership():
    frames = make_frames()

    sampler = DatasetBalancedSampler(
        frames,
        seed=26053,
    )

    for sample in sampler.sample_many(
        5000
    ):
        assert (
            sample.frame_id
            in frames[
                sample.dataset_id
            ]
        )


def check_resume_exact_replay():
    frames = make_frames()

    original = DatasetBalancedSampler(
        frames,
        seed=26053,
    )

    original.sample_many(137)

    state = copy.deepcopy(
        original.state_dict()
    )

    expected_future = (
        original.sample_many(300)
    )

    resumed = DatasetBalancedSampler(
        frames,
        seed=26053,
    )

    resumed.load_state_dict(
        state
    )

    actual_future = (
        resumed.sample_many(300)
    )

    assert (
        actual_future
        == expected_future
    )

    assert resumed.draw_count == (
        137 + 300
    )


def check_resume_mismatch_rejected():
    frames = make_frames()

    original = DatasetBalancedSampler(
        frames,
        seed=26053,
    )

    state = copy.deepcopy(
        original.state_dict()
    )

    changed = make_frames()
    changed["nuScenes"] = (
        "different-frame",
    )

    resumed = DatasetBalancedSampler(
        changed,
        seed=26053,
    )

    try:
        resumed.load_state_dict(
            state
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Frame configuration mismatch accepted"
        )


def check_invalid_inputs():
    try:
        DatasetBalancedSampler(
            {},
            seed=26053,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Empty dataset collection accepted"
        )

    try:
        DatasetBalancedSampler(
            {
                "SemanticKITTI": (),
            },
            seed=26053,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Empty dataset accepted"
        )

    sampler = DatasetBalancedSampler(
        make_frames(),
        seed=26053,
    )

    try:
        sampler.sample_many(-1)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Negative sample count accepted"
        )


def main():
    print("M6 DATASET-BALANCED SAMPLER UNIT GATE")
    print("=" * 50)

    check_seeded_replay()
    print("[PASS] Seeded sampling replay deterministic")

    check_different_seed()
    print("[PASS] Different seed changes sampling sequence")

    check_dictionary_order_independence()
    print("[PASS] Dataset insertion order irrelevant")

    check_dataset_balance()
    print("[PASS] Dataset-level sampling approximately uniform")

    check_small_dataset_not_size_suppressed()
    print("[PASS] Dataset size does not control dataset probability")

    check_frame_membership()
    print("[PASS] Frames sampled only within selected dataset")

    check_resume_exact_replay()
    print("[PASS] Sampler RNG state resumes exactly")

    check_resume_mismatch_rejected()
    print("[PASS] Resume configuration mismatch rejected")

    check_invalid_inputs()
    print("[PASS] Invalid sampler inputs rejected")

    print("=" * 50)
    print(
        "M6 DATASET-BALANCED SAMPLER UNIT GATE: PASS"
    )


if __name__ == "__main__":
    main()