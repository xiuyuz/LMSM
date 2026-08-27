import json
from pathlib import Path

import pytest

from lmsm import load_profile
from reproduction.run_hf_fixed_policy import (
    ActionCapture,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    read_jsonl,
    select_profile,
)


ROOT = Path(__file__).resolve().parents[1]


def test_selected_mode_uses_only_requested_provided_rules():
    profile = load_profile(ROOT / "profiles/lmsm_transcoder.yaml")

    selected = select_profile(
        profile,
        "selected",
        ["cybercrime_intrusion", "illegal_activity"],
    )

    assert selected.policy.active_rule_ids == (
        "cybercrime_intrusion",
        "illegal_activity",
    )
    assert len(profile.policy.active_rule_ids) == 6
    assert DEFAULT_MAX_NEW_TOKENS == 8192
    assert DEFAULT_TEMPERATURE == 0.6
    assert DEFAULT_TOP_P == 0.95


def test_selected_mode_rejects_an_empty_or_unknown_rule_list():
    profile = load_profile(ROOT / "profiles/lmsm_sae.yaml")

    with pytest.raises(ValueError, match="at least one"):
        select_profile(profile, "selected", [])
    with pytest.raises(KeyError, match="unknown rule"):
        select_profile(profile, "selected", ["not_a_rule"])


def test_jsonl_limit_and_action_capture(tmp_path):
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        "\n".join(json.dumps({"id": str(i), "prompt": f"p{i}"}) for i in range(3))
        + "\n",
        encoding="utf-8",
    )
    capture = ActionCapture()
    capture.begin("1")
    capture.log_step(
        {
            "step": 7,
            "decision": {
                "action": "refuse",
                "params": {"rule_id": "illegal_activity"},
                "reason": "threshold crossed",
            },
        }
    )

    assert [row["id"] for row in read_jsonl(input_path, limit=2)] == ["0", "1"]
    assert [row["id"] for row in read_jsonl(input_path, num_shards=2, shard_index=1)] == ["1"]
    assert capture.action == {
        "request_id": "1",
        "step": 7,
        "action": "refuse",
        "reason": "threshold crossed",
        "rule_id": "illegal_activity",
    }
