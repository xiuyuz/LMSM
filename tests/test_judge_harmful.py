from collections import Counter

from reproduction.judge_harmful import historical_batches


REPORTED_ARMS = ("matched_disabled", "lmsm_checkpoint", "lmsm_anytime")


def sample_arms(row_count=3):
    return {
        arm: [
            {
                "request_id": str(index),
                "prompt": f"prompt-{index}",
                "released_output": f"{arm}-{index}",
            }
            for index in range(row_count)
        ]
        for arm in REPORTED_ARMS
    }


def test_harmbench_blind_mix_contains_only_reported_arms():
    batches = historical_batches(sample_arms(), "harmbench")

    assert len(batches) == 1
    assert Counter(row["arm"] for row in batches[0]) == {
        "matched_disabled": 3,
        "lmsm_checkpoint": 3,
        "lmsm_anytime": 3,
    }


def test_wildjailbreak_scores_guarded_batch_before_disabled():
    guarded, disabled = historical_batches(sample_arms(), "wildjailbreak")

    assert {row["arm"] for row in guarded} == {
        "lmsm_checkpoint",
        "lmsm_anytime",
    }
    assert {row["arm"] for row in disabled} == {"matched_disabled"}
