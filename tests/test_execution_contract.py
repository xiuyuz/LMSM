"""CPU tests for the LMSM execution contract (state, release, startup checks)."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import lmsm.execution_contract as contract
from lmsm.backends.dense_probe import CheckpointRule, DenseProbe
from lmsm.execution_contract import (
    ALLOW_WITH_RECORD,
    COMPLETED,
    DENY,
    EXECUTION_FAILED,
    NOT_REACHED,
    RELEASE_NONE,
    RELEASE_ORDINARY,
    RELEASE_REFUSAL,
    ExecutionRequirements,
    ObservationWindow,
    RequestAccount,
    UnsupportedEngineError,
    assign_execution_state,
    check_engine_arguments,
    check_vllm_config,
    derive_execution_requirements,
    parse_drop_observation_specs,
    release_outcome,
)
from lmsm.integrations import vllm as runtime
from lmsm.loaders.profile_loader import load_profile
from lmsm.policy_evaluator import (
    AnytimePolicyEvaluator,
    CheckpointPolicyEvaluator,
    SharedFeaturePolicyEvaluator,
)
from lmsm.state import Decision


ROOT = Path(__file__).resolve().parents[1]
GOOD_ENV = {"VLLM_ENABLE_V1_MULTIPROCESSING": "0"}


# ---------------------------------------------------------------------------
# Helpers


def _profile(name: str, *, hidden_width: int = 1, disposition: str | None = None):
    profile = load_profile(ROOT / f"profiles/{name}.yaml")
    binding = replace(
        profile.backend_bindings[0],
        config={**profile.backend_bindings[0].config, "hidden_width": hidden_width},
    )
    profile = replace(profile, backend_bindings=(binding,))
    if disposition is not None:
        profile = replace(
            profile, policy=replace(profile.policy, not_reached_disposition=disposition)
        )
    return profile


def _probe(target_id: str, *, width: int = 1, threshold: float = 0.5):
    return DenseProbe(
        target_id=target_id,
        mean=np.zeros(width, dtype=np.float64),
        scale=np.ones(width, dtype=np.float64),
        coef=np.ones(width, dtype=np.float64),
        intercept=np.float64(0.0),
        threshold=np.float64(threshold),
    )


def _checkpoint_rule(target_id: str, *, fusion_threshold: float = 2.0):
    return CheckpointRule(
        target_id=target_id,
        prompt=_probe(target_id, threshold=0.0),
        steps_1_32=_probe(target_id, threshold=0.0),
        steps_33_64=_probe(target_id, threshold=0.0),
        fusion=_probe(target_id, width=3, threshold=fusion_threshold),
    )


def _checkpoint_requirements(disposition: str = ALLOW_WITH_RECORD):
    profile = _profile("lmsm_checkpoint", disposition=disposition)
    evaluator = CheckpointPolicyEvaluator([_checkpoint_rule("chemical_biological")], profile)
    return evaluator.execution_requirements()


def _anytime_requirements(disposition: str = ALLOW_WITH_RECORD):
    profile = _profile("lmsm_anytime", disposition=disposition)
    evaluator = AnytimePolicyEvaluator([_probe("chemical_biological")], profile)
    return evaluator.execution_requirements()


def _account(observed_steps, *, decision_step=None, decision=None):
    account = RequestAccount("request", prompt_length=5)
    for step in observed_steps:
        account.mark_processed(step)
        account.mark_evaluated(step)
    if decision_step is not None:
        account.record_decision(decision_step, decision)
    return account


def _params(request_id, **overrides):
    values = {
        "extra_args": {runtime.REQUEST_ID_EXTRA_ARG: request_id},
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "n": 1,
        "max_tokens": 512,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _update(*, batch_size, removed=(), added=(), moved=()):
    return SimpleNamespace(
        batch_size=batch_size,
        removed=list(removed),
        added=list(added),
        moved=list(moved),
    )


class RowScoreEvaluator:
    """Anytime-like evaluator: refuse when column 0 of the row is positive."""

    observes_prompt = False
    observes_decode = True
    max_decode_step = None

    def __init__(self, *, raise_on_step=None):
        self.profile = _profile("lmsm_anytime", hidden_width=2)
        self.raise_on_step = raise_on_step
        self.seen = {}
        self.active = set()

    def add(self, key):
        self.active.add(key)
        self.seen[key] = []

    def remove(self, key):
        self.active.discard(key)

    def needs_prompt(self, _key):
        return False

    def evaluate(self, keys, activations, steps):
        if self.raise_on_step is not None and self.raise_on_step in steps:
            raise RuntimeError("synthetic evaluator fault")
        decisions = []
        for row, (key, step) in enumerate(zip(keys, steps, strict=True)):
            score = float(activations[row, 0])
            self.seen.setdefault(key, []).append((step, float(activations[row, 1])))
            if score > 0:
                decisions.append(
                    Decision(
                        "refuse",
                        {"rule_id": "chemical_biological", "target_id": "chemical_biological",
                         "score": score, "threshold": 0.0},
                        "crossed",
                    )
                )
            else:
                decisions.append(Decision("allow"))
        return decisions

    def feature_maxima_for(self, _key):
        return {}

    def scores_for(self, key):
        return {"trace": list(self.seen.get(key, []))}

    def score_steps_for(self, _key):
        return {}

    def actions_for(self, _key):
        return {}


def _configure(evaluator, **knobs):
    state = runtime.STATE
    state.configure(evaluator, [91, 92], **knobs)
    state.eos_token_id = 3
    return state


def _step(state, rows_hidden, *, capture=True, counts=None):
    """Run one forward: begin_forward, the hook (optional), and the gate."""

    counts = counts or [1] * len(rows_hidden)
    state.begin_forward(counts)
    if capture:
        # Packed layout: each request contributes `count` positions.
        packed = [vector for vector, count in zip(rows_hidden, counts, strict=True)
                  for _ in range(count)]
        state.capture(torch.tensor(packed, dtype=torch.float32))
    logits = torch.zeros(len(rows_hidden), 4)
    state.apply_actions(logits)
    return logits


def _grow(outputs, *tokens):
    for output, token in zip(outputs, tokens, strict=True):
        output.append(token)


@pytest.fixture(autouse=True)
def _reset_state():
    yield
    runtime.STATE.reset()


# ---------------------------------------------------------------------------
# Declarations


def test_builtin_policies_declare_observations_and_evaluation_points():
    checkpoint = _checkpoint_requirements()
    assert [(w.name, w.first_step, w.last_step) for w in checkpoint.windows] == [
        ("prompt", 0, 0),
        ("steps_1_32", 1, 32),
        ("steps_33_64", 33, 64),
    ]
    assert checkpoint.evaluation_steps == (64,)
    assert checkpoint.not_reached_disposition == ALLOW_WITH_RECORD
    assert checkpoint.requires_observation(64, None)
    assert not checkpoint.requires_observation(65, None)

    anytime = _anytime_requirements()
    assert [(w.name, w.first_step, w.last_step) for w in anytime.windows] == [
        ("decode", 1, None)
    ]
    assert anytime.evaluation_steps is None
    assert not anytime.requires_observation(0, None)
    assert anytime.is_evaluation_point(400, None)
    assert not anytime.is_evaluation_point(9, 8)  # nothing after the decision

    shared = SharedFeaturePolicyEvaluator(
        None, [], None, None, _profile("lmsm_transcoder")
    ).execution_requirements()
    assert shared.is_evaluation_point(0, None)
    assert shared.first_evaluation_step() == 0

    for name in ("lmsm_checkpoint", "lmsm_anytime", "lmsm_sae", "lmsm_transcoder"):
        assert load_profile(ROOT / f"profiles/{name}.yaml").policy.not_reached_disposition == (
            ALLOW_WITH_RECORD
        )


def test_custom_evaluators_get_a_derived_running_declaration():
    evaluator = RowScoreEvaluator()
    requirements = derive_execution_requirements(evaluator)
    assert [(w.name, w.first_step, w.last_step) for w in requirements.windows] == [
        ("decode", 1, None)
    ]
    assert requirements.evaluation_steps is None


def test_profile_loader_reads_and_validates_the_not_reached_disposition(tmp_path):
    source = (ROOT / "profiles/lmsm_checkpoint.yaml").read_text(encoding="utf-8")
    deny = tmp_path / "deny.yaml"
    deny.write_text(
        source.replace("execution: {not_reached: allow_with_record}", "execution: {not_reached: deny}"),
        encoding="utf-8",
    )
    assert load_profile(deny).policy.not_reached_disposition == DENY

    default = tmp_path / "default.yaml"
    default.write_text(
        source.replace("  execution: {not_reached: allow_with_record}\n", ""),
        encoding="utf-8",
    )
    assert load_profile(default).policy.not_reached_disposition == ALLOW_WITH_RECORD

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        source.replace("not_reached: allow_with_record", "not_reached: fall_back"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not-reached disposition"):
        load_profile(invalid)


# ---------------------------------------------------------------------------
# State assignment


def test_checkpoint_completed_allow_and_refuse():
    requirements = _checkpoint_requirements()
    allow = assign_execution_state(requirements, _account(range(65)), 300)
    assert (allow.state, allow.decision, allow.release) == (COMPLETED, "allow", RELEASE_ORDINARY)
    assert allow.evaluation_points_reached == [64]
    assert [w["required"] for w in allow.windows] == [1, 32, 32]
    assert [w["observed"] for w in allow.windows] == [1, 32, 32]

    refuse = assign_execution_state(
        requirements, _account(range(65), decision_step=64, decision="refuse"), 65
    )
    assert (refuse.state, refuse.decision, refuse.release) == (COMPLETED, "refuse", RELEASE_REFUSAL)


def test_checkpoint_not_reached_follows_the_declared_disposition():
    for disposition, release in ((ALLOW_WITH_RECORD, RELEASE_ORDINARY), (DENY, RELEASE_REFUSAL)):
        outcome = assign_execution_state(
            _checkpoint_requirements(disposition), _account(range(40)), 40
        )
        assert outcome.state == NOT_REACHED
        assert outcome.disposition == disposition
        assert outcome.release == release
        assert outcome.decision is None
    # Generated token 64 is the final token, so it is never processed: not reached.
    assert assign_execution_state(_checkpoint_requirements(), _account(range(64)), 64).state == (
        NOT_REACHED
    )


def test_missing_observation_takes_precedence_over_not_reached():
    observed = [step for step in range(40) if step != 17]
    outcome = assign_execution_state(_checkpoint_requirements(DENY), _account(observed), 40)
    assert outcome.state == EXECUTION_FAILED
    assert outcome.release == RELEASE_REFUSAL
    assert outcome.missing_observation_steps == [17]
    assert "missing observation" in outcome.reason


def test_missing_evaluation_point_observation_fails():
    outcome = assign_execution_state(_checkpoint_requirements(), _account(range(64)), 200)
    assert outcome.state == EXECUTION_FAILED
    assert outcome.missing_observation_steps == [64]
    assert outcome.missing_evaluation_steps == [64]


def test_positions_after_the_declared_horizon_are_not_required():
    outcome = assign_execution_state(_checkpoint_requirements(), _account(range(65)), 512)
    assert outcome.state == COMPLETED


def test_anytime_states():
    requirements = _anytime_requirements()
    assert assign_execution_state(requirements, _account([]), 1).state == NOT_REACHED
    assert assign_execution_state(requirements, _account([1]), 2).state == COMPLETED
    gap = assign_execution_state(requirements, _account([1, 2, 3, 4, 6, 7, 8, 9]), 10)
    assert (gap.state, gap.missing_observation_steps) == (EXECUTION_FAILED, [5])
    # After a decision (e.g. min_tokens kept generating) nothing more is required.
    acted = assign_execution_state(
        requirements, _account([1, 2, 3], decision_step=3, decision="refuse"), 10
    )
    assert (acted.state, acted.decision, acted.release) == (COMPLETED, "refuse", RELEASE_REFUSAL)
    terminated = assign_execution_state(
        requirements, _account([1, 2], decision_step=2, decision="terminate"), 3
    )
    assert (terminated.state, terminated.release) == (COMPLETED, RELEASE_NONE)


def test_shared_feature_policy_is_never_not_reached():
    requirements = SharedFeaturePolicyEvaluator(
        None, [], None, None, _profile("lmsm_transcoder")
    ).execution_requirements()
    assert assign_execution_state(requirements, _account([0]), 1).state == COMPLETED
    assert assign_execution_state(requirements, _account([]), 1).state == EXECUTION_FAILED


def test_runtime_failures_and_missing_accounts_fail():
    requirements = _anytime_requirements()
    account = _account(range(1, 10))
    account.fail(4, "re-admitted to the batch after 4 generated token(s)")
    outcome = assign_execution_state(requirements, account, 10)
    assert outcome.state == EXECUTION_FAILED
    assert "re-admitted" in outcome.reason

    missing = assign_execution_state(requirements, None, 10)
    assert (missing.state, missing.release) == (EXECUTION_FAILED, RELEASE_REFUSAL)
    assert "no execution account" in missing.reason

    # A decision at a step the policy does not declare is inconsistent.
    checkpoint = assign_execution_state(
        _checkpoint_requirements(), _account(range(65), decision_step=10, decision="refuse"), 65
    )
    assert checkpoint.state == EXECUTION_FAILED


def test_release_rules_table():
    assert release_outcome(COMPLETED, "allow", None) == RELEASE_ORDINARY
    assert release_outcome(COMPLETED, "refuse", None) == RELEASE_REFUSAL
    assert release_outcome(COMPLETED, "terminate", None) == RELEASE_NONE
    assert release_outcome(NOT_REACHED, None, ALLOW_WITH_RECORD) == RELEASE_ORDINARY
    assert release_outcome(NOT_REACHED, None, DENY) == RELEASE_REFUSAL
    assert release_outcome(EXECUTION_FAILED, None, DENY) == RELEASE_REFUSAL
    assert release_outcome(EXECUTION_FAILED, "allow", None) == RELEASE_REFUSAL


# ---------------------------------------------------------------------------
# Runtime accounting through BatchPolicyState


def test_completed_requests_keep_the_plain_release_view():
    state = _configure(RowScoreEvaluator())
    out_a, out_b = [], []
    state.update_batch(
        _update(batch_size=2, added=[(0, _params("a"), [10, 11], out_a), (1, _params("b"), [12], out_b)])
    )
    _step(state, [[0.0, 0.0], [0.0, 0.0]], counts=[2, 1])  # prompt pass
    _grow((out_a, out_b), 5, 6)
    _step(state, [[0.0, 1.0], [1.0, 1.0]])  # b crosses at step 1
    _grow((out_a, out_b), 7, 3)
    state.update_batch(_update(batch_size=1, removed=[1]))
    _step(state, [[0.0, 2.0]])
    _grow((out_a,), 8)

    view_a = runtime.buffered_output("a", list(out_a))
    record_a = view_a.pop("execution_record")
    assert view_a == {
        "request_id": "a",
        "ordinary_model_prefix_tokens": [5, 7, 8],
        "forced_termination_token": None,
        "fixed_refusal_tokens": [],
        "admitted_prefix_tokens": [5, 7, 8],
        "released_output_tokens": [5, 7, 8],
        "action_record": None,
    }
    assert (record_a["state"], record_a["decision"], record_a["release_outcome"]) == (
        COMPLETED,
        "allow",
        RELEASE_ORDINARY,
    )
    assert record_a["observed_positions"] == [
        {"window": "decode", "first_step": 1, "last_step": None, "required": 2, "observed": 2}
    ]

    view_b = runtime.buffered_output("b", list(out_b))
    record_b = view_b.pop("execution_record")
    action = state.record_for("b").to_dict()
    assert view_b == {**action, "action_record": action}
    assert view_b["released_output_tokens"] == [91, 92]
    assert (record_b["state"], record_b["decision"], record_b["decision_step"]) == (
        COMPLETED,
        "refuse",
        1,
    )


def test_uncaptured_forward_fails_closed_instead_of_crashing():
    state = _configure(RowScoreEvaluator())
    out = []
    state.update_batch(_update(batch_size=1, added=[(0, _params("a"), [10], out)]))
    _step(state, [[0.0, 0.0]])
    out.append(5)
    # Compiled/CUDA-graph execution: the hook never runs, the gate still does.
    logits = _step(state, [[1.0, 0.0]], capture=False)
    assert torch.equal(logits, torch.zeros(1, 4))  # nothing acted on
    out.append(6)
    _step(state, [[1.0, 0.0]], capture=False)  # failed request is not observed again
    out.append(7)

    view = runtime.buffered_output("a", list(out))
    record = view["execution_record"]
    assert record["state"] == EXECUTION_FAILED
    assert view["released_output_tokens"] == [91, 92]
    assert view["admitted_prefix_tokens"] == []
    assert view["action_record"] is None
    assert "activation hook did not run" in record["reason"]
    assert record["missing_observation_steps"] == [1, 2]
    assert runtime.runtime_counters()["uncaptured_forwards"] == 2


def test_injected_drop_fails_only_the_selected_request():
    evaluator = RowScoreEvaluator()
    state = _configure(evaluator, inject_drop_observations={"a": {2}})
    out_a, out_b = [], []
    state.update_batch(
        _update(batch_size=2, added=[(0, _params("a"), [10], out_a), (1, _params("b"), [11], out_b)])
    )
    for step in range(4):
        _step(state, [[0.0, step], [0.0, step]])
        _grow((out_a, out_b), 5, 5)

    record_a = runtime.buffered_output("a", list(out_a))["execution_record"]
    record_b = runtime.buffered_output("b", list(out_b))["execution_record"]
    assert record_a["state"] == EXECUTION_FAILED
    assert record_a["injected_faults"] == [{"kind": "drop_observation", "step": 2}]
    assert record_a["missing_observation_steps"] == [2, 3]
    assert record_b["state"] == COMPLETED
    assert [step for step, _ in evaluator.seen["b"]] == [1, 2, 3]
    assert [step for step, _ in evaluator.seen["a"]] == [1]


def test_checkpoint_missing_window_observation_fails_without_crashing():
    profile = _profile("lmsm_checkpoint")
    rule = _checkpoint_rule("chemical_biological", fusion_threshold=-1.0)
    evaluator = CheckpointPolicyEvaluator([rule], profile.with_active_rules(("chemical_biological",)))
    state = _configure(evaluator, inject_drop_observations={"a": {20}})
    out_a, out_b = [], []
    state.update_batch(
        _update(batch_size=2, added=[(0, _params("a"), [10, 11], out_a), (1, _params("b"), [12, 13], out_b)])
    )
    for step in range(66):
        counts = [2, 2] if step == 0 else [1, 1]
        if out_b and out_b[-1] == 3:
            break
        _step(state, [[1.0], [1.0]], counts=counts)
        _grow((out_a, out_b), 5, 3 if state.record_for("b") else 5)

    record_a = runtime.buffered_output("a", list(out_a))["execution_record"]
    view_b = runtime.buffered_output("b", list(out_b))
    assert record_a["state"] == EXECUTION_FAILED
    assert 20 in record_a["missing_observation_steps"]
    assert state.record_for("a") is None  # never evaluated at step 64
    assert view_b["execution_record"]["state"] == COMPLETED
    assert view_b["execution_record"]["decision_step"] == 64
    assert view_b["released_output_tokens"] == [91, 92]


def test_partial_prefill_fails_a_prompt_observing_policy():
    profile = _profile("lmsm_checkpoint")
    evaluator = CheckpointPolicyEvaluator(
        [_checkpoint_rule("chemical_biological")],
        profile.with_active_rules(("chemical_biological",)),
    )
    state = _configure(evaluator)
    out = []
    state.update_batch(_update(batch_size=1, added=[(0, _params("a"), [10, 11, 12], out)]))
    _step(state, [[1.0]], counts=[2])  # only 2 of 3 prompt tokens scheduled
    out.append(5)
    record = runtime.buffered_output("a", list(out))["execution_record"]
    assert record["state"] == EXECUTION_FAILED
    assert "chunked prefill or prefix caching" in record["reason"]


def test_readmission_after_removal_fails():
    state = _configure(RowScoreEvaluator())
    out = []
    state.update_batch(_update(batch_size=1, added=[(0, _params("a"), [10], out)]))
    _step(state, [[0.0, 0.0]])
    out.append(5)
    _step(state, [[0.0, 0.0]])
    out.append(6)
    state.update_batch(_update(batch_size=0, removed=[0]))
    state.update_batch(_update(batch_size=1, added=[(0, _params("a"), [10], out)]))
    _step(state, [[0.0, 0.0]], counts=[3])
    out.append(7)
    record = runtime.buffered_output("a", list(out))["execution_record"]
    assert record["state"] == EXECUTION_FAILED
    assert record["admissions"] == 2
    assert "re-admitted" in record["reason"]


def test_evaluator_errors_fail_the_batch_rows_and_keep_serving():
    state = _configure(RowScoreEvaluator(raise_on_step=2))
    out = []
    state.update_batch(_update(batch_size=1, added=[(0, _params("a"), [10], out)]))
    for _ in range(4):
        _step(state, [[0.0, 0.0]])
        out.append(5)
    record = runtime.buffered_output("a", list(out))["execution_record"]
    assert record["state"] == EXECUTION_FAILED
    assert "synthetic evaluator fault" in record["reason"]
    assert runtime.runtime_counters()["evaluation_errors"] == 1


def test_not_reached_deny_releases_the_fixed_refusal():
    evaluator = RowScoreEvaluator()
    evaluator.profile = replace(
        evaluator.profile,
        policy=replace(evaluator.profile.policy, not_reached_disposition=DENY),
    )
    state = _configure(evaluator)
    out = []
    state.update_batch(_update(batch_size=1, added=[(0, _params("a"), [10], out)]))
    _step(state, [[0.0, 0.0]])
    out.append(3)  # EOS as the first generated token
    view = runtime.buffered_output("a", list(out))
    assert view["execution_record"]["state"] == NOT_REACHED
    assert view["execution_record"]["disposition"] == DENY
    assert view["released_output_tokens"] == [91, 92]


def test_unknown_request_is_never_silently_allowed():
    _configure(RowScoreEvaluator())
    view = runtime.buffered_output("never-admitted", [5, 6, 7])
    assert view["execution_record"]["state"] == EXECUTION_FAILED
    assert view["released_output_tokens"] == [91, 92]


def test_row_keyed_ablation_mixes_state_across_requests():
    evaluator = RowScoreEvaluator()
    state = _configure(evaluator, inject_row_keyed_state=True)
    out_a, out_b = [], []
    state.update_batch(
        _update(batch_size=2, added=[(0, _params("a"), [10], out_a), (1, _params("b"), [11], out_b)])
    )
    _step(state, [[0.0, 0.0], [0.0, 0.0]])
    _grow((out_a, out_b), 5, 5)
    _step(state, [[0.0, 1.0], [0.0, 2.0]])
    _grow((out_a, out_b), 5, 5)
    state.update_batch(_update(batch_size=2, moved=[(0, 1, SimpleNamespace(name="SWAP"))]))
    _step(state, [[0.0, 3.0], [0.0, 4.0]])  # row 0 is now b, row 1 is a
    _grow((out_b, out_a), 5, 5)

    record_a = runtime.buffered_output("a", list(out_a))["execution_record"]
    assert record_a["state"] == COMPLETED  # the contract does not see association errors
    assert record_a["association_ablation"]["foreign_evidence_request_ids"] == ["b"]
    assert record_a["association_ablation"]["decision_used_foreign_evidence"] is True
    # Request-keyed state (the default) keeps each request's evidence separate.
    runtime.STATE.reset()
    evaluator = RowScoreEvaluator()
    state = _configure(evaluator)
    out_a, out_b = [], []
    state.update_batch(
        _update(batch_size=2, added=[(0, _params("a"), [10], out_a), (1, _params("b"), [11], out_b)])
    )
    _step(state, [[0.0, 0.0], [0.0, 0.0]])
    _grow((out_a, out_b), 5, 5)
    _step(state, [[0.0, 1.0], [0.0, 2.0]])
    _grow((out_a, out_b), 5, 5)
    state.update_batch(_update(batch_size=2, moved=[(0, 1, SimpleNamespace(name="SWAP"))]))
    _step(state, [[0.0, 3.0], [0.0, 4.0]])
    assert evaluator.seen["a"] == [(1, 1.0), (2, 4.0)]
    assert evaluator.seen["b"] == [(1, 2.0), (2, 3.0)]
    assert "association_ablation" not in runtime.buffered_output("a", [5, 5, 5])["execution_record"]


def test_row_keyed_ablation_gives_an_admitted_request_a_fresh_slot():
    # Anytime keeps a running activation sum that only `remove` clears.  The
    # ablation must reset a reused slot completely, or the new occupant's first
    # running mean would include the previous occupant's evidence.
    profile = _profile("lmsm_anytime").with_active_rules(("chemical_biological",))
    evaluator = AnytimePolicyEvaluator([_probe("chemical_biological", threshold=5.0)], profile)
    state = _configure(evaluator, inject_row_keyed_state=True)
    out_a = []
    state.update_batch(_update(batch_size=1, added=[(0, _params("a"), [10], out_a)]))
    _step(state, [[0.0]])
    out_a.append(5)
    for _ in range(3):
        _step(state, [[4.0]])  # running mean 4.0 < 5.0
        out_a.append(5)
    out_c = []
    state.update_batch(_update(batch_size=1, added=[(0, _params("c"), [11], out_c)]))
    _step(state, [[0.0]])
    out_c.append(5)
    _step(state, [[1.0]])  # fresh slot: mean 1.0, where a stale sum would give 13.0
    out_c.append(5)
    assert state.record_for("c") is None
    assert runtime.policy_scores("c") == {"chemical_biological": 1.0}
    assert runtime.policy_scores("a") == {"chemical_biological": 4.0}


def test_row_keyed_ablation_without_reset_inherits_the_slot():
    profile = _profile("lmsm_anytime").with_active_rules(("chemical_biological",))
    evaluator = AnytimePolicyEvaluator([_probe("chemical_biological", threshold=5.0)], profile)
    state = _configure(evaluator, inject_row_keyed_state="no_reset_on_admission")
    out_a = []
    state.update_batch(_update(batch_size=1, added=[(0, _params("a"), [10], out_a)]))
    _step(state, [[0.0]])
    out_a.append(5)
    for _ in range(3):
        _step(state, [[4.0]])
        out_a.append(5)
    out_c = []
    state.update_batch(_update(batch_size=1, added=[(0, _params("c"), [11], out_c)]))
    _step(state, [[0.0]])
    out_c.append(5)
    _step(state, [[1.0]])  # inherited: running mean (12 + 1) / 4, maximum 4.0
    out_c.append(5)
    assert runtime.policy_scores("c") == {"chemical_biological": 4.0}
    record_c = runtime.buffered_output("c", list(out_c))["execution_record"]
    assert record_c["association_ablation"]["mode"] == "no_reset_on_admission"
    assert record_c["association_ablation"]["foreign_evidence_request_ids"] == ["a"]
    with pytest.raises(ValueError, match="row-keyed ablation mode"):
        _configure(RowScoreEvaluator(), inject_row_keyed_state="sometimes")


def test_row_keyed_ablation_archives_a_departure_hidden_by_condense():
    # vLLM reports a finished request whose slot condense() refills only as a
    # move onto that slot (no `removed` entry).
    evaluator = RowScoreEvaluator()
    state = _configure(evaluator, inject_row_keyed_state=True)
    out_a, out_b = [], []
    state.update_batch(
        _update(batch_size=2, added=[(0, _params("a"), [10], out_a), (1, _params("b"), [11], out_b)])
    )
    _step(state, [[0.0, 0.0], [0.0, 0.0]])
    _grow((out_a, out_b), 5, 5)
    _step(state, [[0.0, 1.0], [0.0, 2.0]])
    _grow((out_a, out_b), 3, 5)  # a finishes
    state.update_batch(_update(batch_size=1, moved=[(1, 0, SimpleNamespace(name="UNIDIRECTIONAL"))]))
    _step(state, [[0.0, 3.0]])  # b now reads row 0, which holds a's state
    out_b.append(5)

    assert runtime.policy_scores("a") == {"trace": [(1, 1.0)]}
    record_b = runtime.buffered_output("b", list(out_b))["execution_record"]
    assert record_b["association_ablation"]["foreign_evidence_request_ids"] == ["a"]
    assert evaluator.seen["row-keyed-state:0"] == [(1, 1.0), (2, 3.0)]

    # Request-keyed state (the default) needs no archive on that move.
    runtime.STATE.reset()
    evaluator = RowScoreEvaluator()
    state = _configure(evaluator)
    out_a, out_b = [], []
    state.update_batch(
        _update(batch_size=2, added=[(0, _params("a"), [10], out_a), (1, _params("b"), [11], out_b)])
    )
    _step(state, [[0.0, 0.0], [0.0, 0.0]])
    _grow((out_a, out_b), 5, 5)
    _step(state, [[0.0, 1.0], [0.0, 2.0]])
    _grow((out_a, out_b), 3, 5)
    state.update_batch(_update(batch_size=1, moved=[(1, 0, SimpleNamespace(name="UNIDIRECTIONAL"))]))
    _step(state, [[0.0, 3.0]])
    assert runtime.policy_scores("a") == {"trace": [(1, 1.0)]}
    assert evaluator.seen["b"] == [(1, 2.0), (2, 3.0)]
    assert "a" in evaluator.active  # with request keying, a's state is never removed on this path


# ---------------------------------------------------------------------------
# Admission and startup checks


def _vllm_config(**overrides):
    values = {
        "enforce_eager": True,
        "mode": SimpleNamespace(name="NONE", value=0),
        "cudagraph_mode": SimpleNamespace(name="NONE", value=0),
        "prefix": False,
        "chunked": False,
        "async_scheduling": False,
        "speculative": None,
        "tp": 1,
        "pp": 1,
        "dp": 1,
    }
    values.update(overrides)
    return SimpleNamespace(
        model_config=SimpleNamespace(
            enforce_eager=values["enforce_eager"],
            hf_config=SimpleNamespace(eos_token_id=[3, 4]),
        ),
        compilation_config=SimpleNamespace(
            mode=values["mode"], cudagraph_mode=values["cudagraph_mode"]
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=values["prefix"]),
        scheduler_config=SimpleNamespace(
            enable_chunked_prefill=values["chunked"],
            async_scheduling=values["async_scheduling"],
        ),
        speculative_config=values["speculative"],
        parallel_config=SimpleNamespace(
            tensor_parallel_size=values["tp"],
            pipeline_parallel_size=values["pp"],
            data_parallel_size=values["dp"],
        ),
    )


def _requirements(violations):
    return [violation.requirement for violation in violations]


def test_resolved_config_check_names_every_violated_requirement():
    assert check_vllm_config(_vllm_config(), GOOD_ENV, "0.19.0") == []
    cases = {
        "eager_execution": dict(
            enforce_eager=False,
            mode=SimpleNamespace(name="VLLM_COMPILE", value=3),
            cudagraph_mode=SimpleNamespace(name="FULL_AND_PIECEWISE", value=(2, 1)),
        ),
        "prefix_caching_off": dict(prefix=True),
        "chunked_prefill_off": dict(chunked=True),
        "async_scheduling_off": dict(async_scheduling=True),
        "no_speculative_decoding": dict(speculative=SimpleNamespace(method="ngram")),
        "tensor_parallel_size_1": dict(tp=2),
        "pipeline_parallel_size_1": dict(pp=2),
        "data_parallel_size_1": dict(dp=2),
    }
    for requirement, overrides in cases.items():
        violations = check_vllm_config(_vllm_config(**overrides), GOOD_ENV, "0.19.0")
        assert _requirements(violations) == [requirement], requirement
    compiled = check_vllm_config(
        _vllm_config(**cases["eager_execution"]), GOOD_ENV, "0.19.0"
    )[0]
    assert "VLLM_COMPILE" in compiled.detail and "CUDA-graph" in compiled.detail

    # CUDA graphs alone (no torch.compile) also skip the Python hook.
    graphs_only = check_vllm_config(
        _vllm_config(cudagraph_mode=SimpleNamespace(name="FULL", value=2)), GOOD_ENV, "0.19.0"
    )
    assert _requirements(graphs_only) == ["eager_execution"]

    assert _requirements(check_vllm_config(_vllm_config(), GOOD_ENV, "0.20.1")) == [
        "vllm_version_0_19"
    ]
    assert _requirements(check_vllm_config(_vllm_config(), {}, "0.19.0")) == [
        "in_process_engine_core"
    ]
    assert _requirements(
        check_vllm_config(
            _vllm_config(),
            {**GOOD_ENV, "VLLM_USE_V2_MODEL_RUNNER": "1"},
            "0.19.0",
        )
    ) == ["model_runner_v1"]


def test_engine_argument_check_matches_the_runner_configuration():
    eager = {
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "enforce_eager": True,
        "enable_chunked_prefill": False,
        "enable_prefix_caching": False,
        "async_scheduling": False,
    }
    assert check_engine_arguments(eager, GOOD_ENV, "0.19.0") == []
    compiled = check_engine_arguments({**eager, "enforce_eager": False}, GOOD_ENV, "0.19.0")
    assert _requirements(compiled) == ["eager_execution"]
    defaults = check_engine_arguments({"enforce_eager": True}, GOOD_ENV, "0.19.0")
    assert _requirements(defaults) == [
        "prefix_caching_off",
        "chunked_prefill_off",
        "async_scheduling_off",
    ]
    with pytest.raises(UnsupportedEngineError, match="eager_execution"):
        raise UnsupportedEngineError(compiled)


def test_gate_refuses_an_unsupported_resolved_config(monkeypatch):
    monkeypatch.setattr(contract, "_installed_vllm_version", lambda: "0.19.0")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    _configure(RowScoreEvaluator())
    compiled = _vllm_config(
        enforce_eager=False,
        mode=SimpleNamespace(name="VLLM_COMPILE", value=3),
        cudagraph_mode=SimpleNamespace(name="FULL_AND_PIECEWISE", value=(2, 1)),
    )
    with pytest.raises(UnsupportedEngineError, match="eager_execution"):
        runtime.VLLMEnforcementGate(compiled, None, False)

    # Test-only bypass: violations are recorded, the gate is constructed.
    _configure(RowScoreEvaluator(), allow_unsupported_engine=True)
    runtime.VLLMEnforcementGate(compiled, None, False)
    assert _requirements(runtime.STATE.engine_violations) == ["eager_execution"]
    assert runtime.STATE.eos_token_id == 3

    _configure(RowScoreEvaluator())
    runtime.VLLMEnforcementGate(_vllm_config(), None, False)
    assert runtime.STATE.engine_violations == []


def _installed_state(**overrides):
    state = _configure(RowScoreEvaluator(), **overrides)
    state.gate_constructions = 1
    state.model_runner_class = runtime.V1_MODEL_RUNNER
    state.hook_name = "decoder.blocks.3.feed_forward"
    state.activation_key = "pre:decoder.blocks.3.feed_forward"
    state.resolved_module = "decoder.blocks.3.feed_forward"
    state.expected_hidden_width = 2560
    state.idle_observed_width = 2560
    return state


def test_engine_installation_checks():
    _installed_state()
    assert runtime.verify_engine()["status"] == "passed"

    cases = {
        "hook_width": dict(idle_observed_width=4096),
        "hook_installed": dict(hook_name=None),
        "hook_installed ": dict(resolved_module="decoder.blocks.4.feed_forward"),
        "in_process_engine_core": dict(gate_constructions=0),
        "model_runner_v1": dict(model_runner_class="vllm.v1.worker.gpu.model_runner.GPUModelRunner"),
    }
    for requirement, fields in cases.items():
        state = _installed_state()
        for key, value in fields.items():
            setattr(state, key, value)
        with pytest.raises(UnsupportedEngineError, match=requirement.strip()):
            runtime.verify_engine()

    state = _installed_state(allow_unsupported_engine=True)
    state.idle_observed_width = None
    result = runtime.verify_engine()
    assert result["status"] == "bypassed"
    assert result["violated_requirements"] == ["hook_width"]

    multiprocess_client = SimpleNamespace(llm_engine=SimpleNamespace(engine_core=SimpleNamespace()))
    _installed_state()
    with pytest.raises(UnsupportedEngineError, match="InprocClient"):
        runtime.verify_engine(multiprocess_client)


def test_idle_hook_call_records_the_startup_width():
    state = _configure(RowScoreEvaluator())
    state.capture(torch.zeros(7, 2))  # startup profiling: no rows, no forward
    assert (state.idle_hook_calls, state.idle_observed_width) == (1, 2)


def test_admission_checks_n_and_the_evaluation_point():
    profile = _profile("lmsm_checkpoint")
    evaluator = CheckpointPolicyEvaluator(
        [_checkpoint_rule("chemical_biological")],
        profile.with_active_rules(("chemical_biological",)),
    )
    _configure(evaluator)
    runtime.VLLMEnforcementGate.validate_params(_params("a", max_tokens=65))
    with pytest.raises(ValueError, match="max_tokens >= 65"):
        runtime.VLLMEnforcementGate.validate_params(_params("a", max_tokens=64))
    with pytest.raises(ValueError, match="n=1"):
        runtime.VLLMEnforcementGate.validate_params(_params("a", n=2))


def test_drop_observation_spec_parsing():
    assert parse_drop_observation_specs(["req-1:3,5", "req:2:7", "req-1:9"]) == {
        "req-1": frozenset({3, 5, 9}),
        "req:2": frozenset({7}),
    }
    for invalid in ("no-steps", "id:", ":4", "id:-1"):
        with pytest.raises(ValueError):
            parse_drop_observation_specs([invalid])


def test_requirements_reject_unknown_dispositions():
    with pytest.raises(ValueError):
        ExecutionRequirements(
            policy_name="p",
            policy_version="1",
            rule_library_version="1",
            schedule={},
            windows=(ObservationWindow("decode", 1, None),),
            not_reached_disposition="fall_back",
        )
