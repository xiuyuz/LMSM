#!/usr/bin/env python3
"""One association run: the 1,704 requests of the duplicate workload through compiled vLLM with LMSM.

The driver calls the library's own runner (examples/run_vllm.py) and runtime unchanged. It only wraps them at run time
to record what the analysis needs:
- every batch update of vLLM (additions, removals, moves and slot reuse) and a fresh-state check at each admission;
- the packed layout of every forward pass (row, request, step and scheduled tokens);
- the request and step of every activation row the policy received;
- for Anytime, every per-step score of all 15 probes;
- for the transcoder and the SAE, the feature values, slot and running maxima of every row the device engine updated.

Options that are not runner flags go to vllm.LLM. A fixed KV cache keeps concurrent runs on one GPU from shrinking each
other's cache, and batch-invariant mode needs the FLASH_ATTN attention backend. With --arrival staggered, requests
arrive at seeded Poisson times measured in engine steps instead of all at once. --inject-row-keyed-state passes the
runtime's test-only switch that keys policy state by batch row (host evaluation only), which the ablation uses.
The engine core must run in this process, because the instrumentation wraps this process's runtime state.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

POLICY_PROFILES = {
    "checkpoint": "profiles/lmsm_checkpoint.yaml",
    "anytime": "profiles/lmsm_anytime.yaml",
    "transcoder": "profiles/lmsm_transcoder.yaml",
    "sae": "profiles/lmsm_sae.yaml",
}
ROW_STATE_KEY_PREFIX = "row-keyed-state:"  # lmsm.integrations.vllm.ROW_STATE_KEY_PREFIX, checked at run time
ARRIVAL_SEED = 7


def is_selected_feature_engine(engine) -> bool:
    """The device engine of the transcoder and the SAE. The Anytime engine also has a maxima buffer, of probe scores."""

    return (
        engine is not None
        and type(engine).__name__ == "DeviceSharedFeatureEngine"
        and hasattr(engine, "maxima")
        and hasattr(engine, "encoder")
        and hasattr(engine, "union")
    )


def request_of_key(state, key) -> str:
    """The request a policy-state key belongs to at this forward. Under row keying it is the row's current request."""

    key = str(key)
    if key.startswith(ROW_STATE_KEY_PREFIX):
        return str(state.rows[int(key[len(ROW_STATE_KEY_PREFIX):])].request_id)
    return key


def save_sf_steps(path: Path, evidence, index_of: dict) -> None:
    """Every row the feature engine evaluated, in order (sf_steps.npz). For each row it keeps the request, the step,
    the step's capture check, the engine's own feature values, the updated slot and its running maxima, and the
    observer call. capture_ok says whether the engine's encode output was captured exactly once for that call."""

    import numpy as np

    np.savez_compressed(
        path,
        request_index=np.asarray([index_of[item] for item in evidence.sf_request], dtype=np.int32),
        step=np.asarray(evidence.sf_step, dtype=np.int32),
        ok=np.concatenate(evidence.sf_ok),
        scores=np.concatenate(evidence.sf_scores, axis=0),
        maxima=np.concatenate(evidence.sf_maxima, axis=0),
        slot=np.concatenate(evidence.sf_slot),
        call=np.concatenate(evidence.sf_call),
        capture_ok=np.concatenate(evidence.sf_capture_ok),
        union=np.asarray(evidence.sf_union, dtype=np.int64),
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", type=Path, default=Path(__file__).resolve().parents[2], help="repository root")
    parser.add_argument("--model", required=True)
    parser.add_argument("--policy", choices=POLICY_PROFILES, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--kv-cache-gib", type=float)
    parser.add_argument("--batch-invariant", action="store_true")
    parser.add_argument("--attention-backend")
    parser.add_argument("--arrival", choices=("burst", "staggered"), default="burst")
    parser.add_argument("--mode", choices=("eager", "compiled"), default="compiled")
    parser.add_argument("--observation", choices=("hook", "graph"), default="graph")
    parser.add_argument("--evaluation", choices=("host", "device"), default="device")
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--max-num-batched-tokens", type=int, default=40960)
    parser.add_argument("--arrival-rate", type=float, help="requests per engine step")
    parser.add_argument("--force-torch-topk-topp", action="store_true",
                        help="use vLLM's PyTorch top-k and top-p sampler at every batch size")
    parser.add_argument("--inject-row-keyed-state", choices=("reset_on_admission", "no_reset_on_admission"),
                        help="the runtime's test-only switch that keys policy state by batch row")
    parser.add_argument("--async-scheduling", action="store_true")
    parser.add_argument("--language-model-only", action="store_true", help="for Gemma 3 checkpoints")
    args = parser.parse_args(argv)
    if args.inject_row_keyed_state and args.evaluation != "host":
        parser.error("--inject-row-keyed-state works with host evaluation only")
    if args.async_scheduling and args.evaluation != "device":
        parser.error("--async-scheduling needs device evaluation")
    if args.policy in ("transcoder", "sae") and args.evaluation != "device":
        parser.error("the transcoder and the SAE run with device evaluation")
    return args


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def poisson_arrival_steps(count: int, rate: float, seed: int) -> list[int]:
    import numpy as np

    gaps = np.random.default_rng(seed).exponential(1.0 / rate, size=count)
    gaps[0] = 0.0
    return [int(value) for value in np.floor(np.cumsum(gaps))]


class BatchEventRecorder:
    """Records each batch update vLLM applies: additions (with slot reuse), removals and moves."""

    def __init__(self, state, request_id_key: str):
        self.state = state
        self.request_id_key = request_id_key
        self.events = []
        self.slot_history = {}
        self.original_update = state.update_batch

    def update(self, batch_update):
        if batch_update is None:
            return self.original_update(batch_update)
        before = {row: request.request_id for row, request in self.state.rows.items()}
        removed = [{"row": int(row), "request_id": before.get(int(row))} for row in batch_update.removed]
        added = []
        for row, params, _prompt_ids, _output_ids in batch_update.added:
            row = int(row)
            request_id = str(params.extra_args[self.request_id_key])
            history = self.slot_history.get(row, [])
            added.append({"row": row, "request_id": request_id,
                          "slot_reused": bool(history and history[-1] != request_id)})
        moved = [{"source": int(source), "destination": int(destination), "direction": direction.name}
                 for source, destination, direction in batch_update.moved]
        self.original_update(batch_update)
        initialized_empty = {}
        evaluator = self.state.evaluator
        for item in added:
            request_id = item["request_id"]
            initialized_empty[request_id] = {
                "features_empty": evaluator.feature_maxima_for(request_id) == {},
                "scores_empty": evaluator.scores_for(request_id) == {},
                "score_steps_empty": evaluator.score_steps_for(request_id) == {},
                "actions_empty": evaluator.actions_for(request_id) == {},
            }
            self.slot_history.setdefault(item["row"], []).append(request_id)
        if added or removed or moved:
            self.events.append({"batch_size": int(batch_update.batch_size), "removed": removed, "added": added,
                                "moved": moved, "initialized_empty": initialized_empty})


class LayoutLog:
    """Per-forward packed layout, read right after the runtime's own mapping."""

    def __init__(self, state):
        self.state = state
        self.original = state.begin_forward
        self.forwards: list[tuple[list[str], list[int], list[int]]] = []

    def begin_forward(self, num_scheduled_tokens):
        self.original(num_scheduled_tokens)
        counts = [int(value) for value in num_scheduled_tokens]
        rows = [self.state.rows[row] for row in range(len(counts))]
        self.forwards.append(
            (
                [request.request_id for request in rows],
                [len(request.output_token_ids) for request in rows],
                counts,
            )
        )


class EvidenceLog:
    """The request and step of every activation row handed to the policy evaluator."""

    def __init__(self, evaluator, torch_module, state=None):
        self.torch = torch_module
        self.original = evaluator.evaluate
        self.state = state  # maps row keys back to the request that owns the row
        self.rows: list[tuple[str, int]] = []
        self.current: dict | None = None
        self.score_calls: list[tuple[list[str], list[int], list]] = []
        # Per evaluated row of the feature engine (transcoder and SAE)
        self.sf_request: list[str] = []
        self.sf_step: list[int] = []
        self.sf_ok: list = []
        self.sf_scores: list = []
        self.sf_maxima: list = []
        self.sf_slot: list = []
        self.sf_call: list = []
        self.sf_capture_ok: list = []
        self.sf_union: list[int] | None = None
        self.sf_dtype: str | None = None
        self.sf_calls = 0
        self.sf_encode_calls = 0
        self.sf_encode_calls_seen = 0
        self.sf_last_scores = None
        self.sf_original_encode = None
        self.sf_problems: list[str] = []
        engine = getattr(state, "device_engine", None) if state is not None else None
        if is_selected_feature_engine(engine):
            # Keep a reference to the feature values the engine computes, without copying or synchronizing,
            # so the observer records the engine's own values instead of recomputing them.
            original_encode = engine.encoder.encode
            self.sf_original_encode = original_encode

            def encode(hidden):
                out = original_encode(hidden)
                self.sf_last_scores = out
                self.sf_encode_calls += 1
                return out

            engine.encoder.encode = encode

    def evaluate(self, state_keys, activations, steps):
        # The runtime passes policy-state keys: request IDs, or row keys under the ablation (mapped back to the
        # request that owns the row at this forward).
        evaluated_keys = list(state_keys)
        request_ids = (
            [request_of_key(self.state, key) for key in evaluated_keys]
            if self.state is not None
            else [str(key) for key in evaluated_keys]
        )
        self.rows.extend((str(request_id), int(step)) for request_id, step in zip(request_ids, steps, strict=True))
        self.current = {"request_ids": list(request_ids), "steps": list(steps), "scores": []}
        try:
            return self.original(evaluated_keys, activations, steps)
        finally:
            if self.current["scores"]:
                self.score_calls.append(
                    (self.current["request_ids"], self.current["steps"], self.current["scores"])
                )
            self.current = None

    def device_observe(self, request_ids, steps, hidden, ok, engine):
        """The runtime's test-only device observer: the rows the device engine read this step."""

        self.rows.extend((str(request_id), int(step)) for request_id, step in zip(request_ids, steps, strict=True))
        last_scores = getattr(engine, "last_scores", None)
        if last_scores is not None and getattr(engine, "observes_prompt", True) is False:
            scores = last_scores[: len(request_ids)].detach().cpu().numpy()
            self.score_calls.append(
                (list(request_ids), list(steps), [scores[:, column] for column in range(scores.shape[1])])
            )
        if self.state is not None and is_selected_feature_engine(engine):
            # The feature engine just ran for these m rows. Record each row's feature values, the slot the engine
            # updated and that slot's running maxima after the update.
            import numpy as np
            from lmsm.integrations.vllm_device import SLOT, STEP

            m = len(request_ids)
            torch = self.torch
            self.sf_calls += 1
            new_encodes = self.sf_encode_calls - self.sf_encode_calls_seen
            self.sf_encode_calls_seen = self.sf_encode_calls
            scores = self.sf_last_scores
            capture_ok = (
                new_encodes == 1 and scores is not None and int(scores.shape[0]) == m
            )
            if not capture_ok:
                # Not expected, because the engine encodes once per observed step. The analysis treats such rows
                # as unverifiable.
                self.sf_problems.append(
                    f"observer call {self.sf_calls}: {new_encodes} encode call(s) since the previous observation, "
                    f"captured rows {None if scores is None else int(scores.shape[0])}, expected {m}"
                )
                encode = self.sf_original_encode or engine.encoder.encode
                scores = encode(hidden[:m])
                self.sf_encode_calls_seen = self.sf_encode_calls
            index = engine.index
            slots = index[SLOT, :m].detach().cpu().numpy().astype(np.int64)
            engine_steps = index[STEP, :m].detach().cpu().numpy().astype(np.int64)
            if [int(x) for x in engine_steps] != [int(x) for x in steps]:
                self.sf_problems.append(f"observer call {self.sf_calls}: engine steps differ from the observer's")
            mapped = [self.state.slot_by_request.get(str(request_id)) for request_id in request_ids]
            if [int(x) if x is not None else None for x in mapped] != [int(x) for x in slots]:
                self.sf_problems.append(
                    f"observer call {self.sf_calls}: slot_by_request disagrees with the engine's slots"
                )
            slot_index = torch.as_tensor(slots, dtype=torch.int64, device=engine.maxima.device)
            maxima = engine.maxima.index_select(0, slot_index).detach().cpu().numpy()
            self.sf_request.extend(str(request_id) for request_id in request_ids)
            self.sf_step.extend(int(step) for step in steps)
            # engine.ok is the step's single capture-epoch check (shape (1,)), applied to every row of the step.
            self.sf_ok.append(np.full(m, bool(ok.reshape(-1)[0].item()), dtype=bool))
            # float64 holds the bf16 (transcoder) and fp32 (SAE) values exactly, as in the engine's comparisons.
            self.sf_scores.append(scores.detach().to(torch.float64).cpu().numpy())
            self.sf_maxima.append(maxima)
            self.sf_slot.append(slots)
            self.sf_call.append(np.full(m, self.sf_calls, dtype=np.int64))
            self.sf_capture_ok.append(np.full(m, bool(capture_ok), dtype=bool))
            self.sf_dtype = str(scores.dtype)
            if self.sf_union is None:
                self.sf_union = [int(feature) for feature in engine.union]

    def wrap_probe(self, probe):
        original = probe.score

        def score(hidden):
            values = original(hidden)
            if self.current is not None:
                self.current["scores"].append(values.copy())
            return values

        probe.score = score


def evaluator_state_is_fresh(evaluator, request_id: str) -> bool:
    """True if the host evaluator holds no evidence for a just admitted request. It reads the evaluator's stores,
    because Checkpoint's fresh state is four empty score maps and an all-False action map."""

    if hasattr(evaluator, "max_scores_by_id"):  # Anytime
        return (
            evaluator.max_scores_by_id.get(request_id) == {}
            and evaluator.max_steps_by_id.get(request_id) == {}
            and evaluator.actions_by_id.get(request_id) == {}
            and evaluator.hidden_counts_by_id.get(request_id) == 0
            and request_id not in evaluator.hidden_sums_by_id
        )
    return (  # Checkpoint
        request_id not in evaluator.prompt_evaluated
        and evaluator.prompt_scores_by_id.get(request_id) == {}
        and evaluator.window_1_scores_by_id.get(request_id) == {}
        and evaluator.window_2_scores_by_id.get(request_id) == {}
        and evaluator.fusion_scores_by_id.get(request_id) == {}
        and evaluator.action_steps_by_id.get(request_id) == {}
        and evaluator.actions_by_id.get(request_id) == {rule.target_id: False for rule in evaluator.rules}
        and evaluator.window_1_counts_by_id.get(request_id) == 0
        and evaluator.window_2_counts_by_id.get(request_id) == 0
        and request_id not in evaluator.window_1_sums_by_id
        and request_id not in evaluator.window_2_sums_by_id
    )


def staggered_generate(llm, arrival_steps: list[int], arrival_log: list):
    """Feed requests at fixed engine steps; return outputs in input order."""

    def generate(prompts, sampling_params, use_tqdm=False):
        del use_tqdm
        count = len(prompts)
        if len(arrival_steps) != count:
            raise ValueError("arrival schedule does not match the request count")
        engine = llm.llm_engine
        outputs = {}
        step = 0
        next_index = 0
        while next_index < count or engine.has_unfinished_requests():
            batch = []
            while next_index < count and arrival_steps[next_index] <= step:
                batch.append(next_index)
                next_index += 1
            if batch:
                # enqueue() returns vLLM's internal IDs ("<n>-<random>"); outputs
                # carry the external counter value <n>, assigned in enqueue order.
                llm.enqueue(
                    [prompts[index] for index in batch],
                    [sampling_params[index] for index in batch],
                    use_tqdm=False,
                )
                arrival_log.append((step, len(batch)))
            if not engine.has_unfinished_requests():
                if next_index < count:
                    step = max(step + 1, arrival_steps[next_index])
                continue
            for output in engine.step():
                if output.finished:
                    outputs[output.request_id] = output
            step += 1
        if len(outputs) != count:
            raise AssertionError(f"{len(outputs)} outputs for {count} requests")
        arrival_log.append(("engine_steps", step))
        # Same ordering rule as LLM._run_engine: external IDs follow enqueue order.
        ordered = [outputs[key] for key in sorted(outputs, key=int)]
        for index, output in enumerate(ordered):
            if list(output.prompt_token_ids) != list(prompts[index]["prompt_token_ids"]):
                raise AssertionError(f"output {output.request_id} does not match input row {index}")
        return ordered

    return generate


def main(argv=None):
    args = parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    # The instrumentation wraps this process's runtime state, so vLLM's engine core must run in this process.
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "0") not in ("", "0"):
        raise SystemExit("VLLM_ENABLE_V1_MULTIPROCESSING must be 0: the association instrumentation needs the "
                         "in-process engine core")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    if args.batch_invariant:
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
    elif os.environ.get("VLLM_BATCH_INVARIANT", "0") not in ("", "0"):
        raise SystemExit("VLLM_BATCH_INVARIANT is set but --batch-invariant was not given")
    if args.arrival == "staggered" and not args.arrival_rate:
        raise SystemExit("--arrival staggered requires --arrival-rate")

    code = args.code.resolve()
    for sub in ("examples", "src"):
        sys.path.insert(0, str(code / sub))
    os.chdir(code)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    workload_path = args.workload.resolve()
    rows = read_jsonl(workload_path)

    import numpy as np
    import torch
    import vllm

    import run_vllm as runner
    import lmsm.integrations.vllm as runtime

    if getattr(runtime, "ROW_STATE_KEY_PREFIX", ROW_STATE_KEY_PREFIX) != ROW_STATE_KEY_PREFIX:
        raise SystemExit("the runtime's row-key prefix differs from the driver's")

    if args.force_torch_topk_topp:
        from vllm.v1.sample.ops import topk_topp_sampler

        topk_topp_sampler.apply_top_k_top_p = topk_topp_sampler.apply_top_k_top_p_pytorch

    extra_llm_kwargs = {}
    if args.attention_backend:
        extra_llm_kwargs["attention_backend"] = args.attention_backend
    if args.kv_cache_gib:
        extra_llm_kwargs["kv_cache_memory_bytes"] = int(args.kv_cache_gib * 2**30)

    holder: dict = {"arrival_log": []}
    original_build = runner._build_components
    original_llm = vllm.LLM

    def llm_factory(**kwargs):
        kwargs.update(extra_llm_kwargs)
        holder["llm_kwargs"] = {
            key: value if isinstance(value, (str, int, float, bool, type(None))) else repr(value)
            for key, value in kwargs.items()
        }
        return original_llm(**kwargs)

    def instrumented_build(run_args, profile, project_root):
        vllm.LLM = llm_factory
        try:
            components = original_build(run_args, profile, project_root)
        finally:
            vllm.LLM = original_llm
        state = runtime.STATE
        recorder = BatchEventRecorder(state, runtime.REQUEST_ID_EXTRA_ARG)
        fresh_checks = holder.setdefault("fresh_checks", [])

        device = args.evaluation == "device"

        def checked_update(batch_update):
            recorder.update(batch_update)
            if batch_update is None:
                return
            for row, params, _prompt, _output in batch_update.added:
                request_id = str(params.extra_args[runtime.REQUEST_ID_EXTRA_ARG])
                if device:
                    fresh = state.device_engine.slot_is_fresh(state.rows[int(row)].slot)
                else:
                    # The state the admitted request starts from: its request ID, or its row key under row keying.
                    key = state._state_key(request_id, int(row)) if args.inject_row_keyed_state else request_id
                    fresh = evaluator_state_is_fresh(state.evaluator, key)
                fresh_checks.append((request_id, fresh))

        state.update_batch = checked_update
        layout = LayoutLog(state)
        state.begin_forward = layout.begin_forward
        evidence = EvidenceLog(state.evaluator, torch, state)
        probe_order = []
        if device:
            state.device_observer = evidence.device_observe
            if args.policy == "anytime":
                probe_order = list(state.device_engine.target_ids)
        else:
            state.evaluator.evaluate = evidence.evaluate
            if args.policy == "anytime":
                for probe in state.evaluator.probes:
                    evidence.wrap_probe(probe)
                    probe_order.append(probe.target_id)
        holder.update(
            recorder=recorder, layout=layout, evidence=evidence, probe_order=probe_order
        )
        if args.arrival == "staggered":
            schedule = poisson_arrival_steps(len(rows), args.arrival_rate, ARRIVAL_SEED)
            holder["arrival_steps"] = schedule
            components.llm.generate = staggered_generate(
                components.llm, schedule, holder["arrival_log"]
            )
        return components

    runner._build_components = instrumented_build
    run_args = runner.parse_args(
        [
            "--model", args.model,
            "--profile", POLICY_PROFILES[args.policy],
            "--input", str(workload_path),
            "--run-dir", str(args.run_dir / "run"),
            "--physical-gpu", str(args.physical_gpu),
            "--sampling", "nucleus",
            "--temperature", "0.6",
            "--top-p", "0.95",
            "--top-k", "20",
            "--max-new-tokens", str(args.max_new_tokens),
            "--max-model-len", "40960",
            "--max-num-seqs", str(args.max_num_seqs),
            "--max-num-batched-tokens", str(args.max_num_batched_tokens),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--mode", args.mode,
            "--observation", args.observation,
            "--evaluation", args.evaluation,
            "--seed", "0",
        ]
        + (["--enable-chunked-prefill"] if args.enable_chunked_prefill else [])
        + (["--enable-prefix-caching"] if args.enable_prefix_caching else [])
        + (["--async-scheduling"] if args.async_scheduling else [])
        + (["--language-model-only"] if args.language_model_only else [])
        + ([f"--inject-row-keyed-state={args.inject_row_keyed_state}"] if args.inject_row_keyed_state else [])
    )

    driver_meta = {
        "driver": str(Path(__file__).resolve()),
        "argv": sys.argv,
        "args": {key: str(value) for key, value in vars(args).items()},
        "code": str(code),
        "runner_args": vars(run_args),
        "env": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "VLLM_ENABLE_V1_MULTIPROCESSING",
                "VLLM_BATCH_INVARIANT",
                "PYTHONPATH",
                "HF_HOME",
            )
        },
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "vllm": vllm.__version__,
            "numpy": np.__version__,
        },
        "extra_llm_kwargs": extra_llm_kwargs,
        "workload": str(workload_path),
        "request_count": len(rows),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = args.run_dir / "driver_metadata.json"
    meta_path.write_text(json.dumps(driver_meta, indent=2, default=str) + "\n", encoding="utf-8")

    started = time.perf_counter()
    try:
        runner.run(run_args)
    except BaseException as error:  # record the failure, then re-raise
        driver_meta["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        driver_meta["llm_kwargs"] = holder.get("llm_kwargs")
        meta_path.write_text(json.dumps(driver_meta, indent=2, default=str) + "\n", encoding="utf-8")
        raise
    driver_meta["wall_seconds"] = time.perf_counter() - started
    driver_meta["llm_kwargs"] = holder.get("llm_kwargs")

    request_ids = [str(row["id"]) for row in rows]
    index_of = {request_id: index for index, request_id in enumerate(request_ids)}

    recorder = holder["recorder"]
    (args.run_dir / "batch_events.json").write_text(
        json.dumps(recorder.events) + "\n", encoding="utf-8"
    )

    layout = holder["layout"]
    offsets = [0]
    flat_request, flat_step, flat_count = [], [], []
    for ids, steps, counts in layout.forwards:
        flat_request.extend(index_of[request_id] for request_id in ids)
        flat_step.extend(steps)
        flat_count.extend(counts)
        offsets.append(len(flat_request))
    np.savez_compressed(
        args.run_dir / "layouts.npz",
        offsets=np.asarray(offsets, dtype=np.int64),
        request_index=np.asarray(flat_request, dtype=np.int32),
        step=np.asarray(flat_step, dtype=np.int32),
        count=np.asarray(flat_count, dtype=np.int32),
    )

    evidence = holder["evidence"]
    np.savez_compressed(
        args.run_dir / "evidence.npz",
        request_index=np.asarray([index_of[request_id] for request_id, _ in evidence.rows], dtype=np.int32),
        step=np.asarray([step for _, step in evidence.rows], dtype=np.int32),
    )
    if args.policy == "anytime":
        call_request, call_step, call_scores = [], [], []
        for ids, steps, scores in evidence.score_calls:
            matrix = np.stack(scores, axis=1)  # rows x probes
            call_request.extend(index_of[request_id] for request_id in ids)
            call_step.extend(steps)
            call_scores.append(matrix)
        np.savez_compressed(
            args.run_dir / "anytime_step_scores.npz",
            request_index=np.asarray(call_request, dtype=np.int32),
            step=np.asarray(call_step, dtype=np.int32),
            scores=np.concatenate(call_scores, axis=0) if call_scores else np.zeros((0, 15)),
            probe_order=np.asarray(holder["probe_order"]),
        )
    if evidence.sf_request:
        save_sf_steps(args.run_dir / "sf_steps.npz", evidence, index_of)

    driver_meta.update(
        {
            "instrumentation": {
                "profile": POLICY_PROFILES[args.policy],
                "inject_row_keyed_state": args.inject_row_keyed_state,
                "async_scheduling": bool(args.async_scheduling),
                "language_model_only": bool(args.language_model_only),
                "engine_process": "inproc (instrumentation)",
                "selected_feature_rows": len(evidence.sf_request),
                "selected_feature_score_dtype": evidence.sf_dtype,
                "selected_feature_observer_calls": evidence.sf_calls,
                "selected_feature_encode_calls": evidence.sf_encode_calls,
                "selected_feature_instrumentation_problems": evidence.sf_problems[:50],
                "selected_feature_instrumentation_problem_count": len(evidence.sf_problems),
                "force_torch_topk_topp": bool(args.force_torch_topk_topp),
            },
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "request_ids_in_input_order": request_ids,
            "forward_count": len(layout.forwards),
            "evaluated_rows": len(evidence.rows),
            "probe_order": holder["probe_order"],
            "arrival_steps": holder.get("arrival_steps"),
            "arrival_log": holder["arrival_log"],
            "fresh_state_checks": len(holder.get("fresh_checks", [])),
            "fresh_state_check_failures": [
                request_id for request_id, fresh in holder.get("fresh_checks", []) if not fresh
            ],
        }
    )
    meta_path.write_text(json.dumps(driver_meta, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(args.run_dir), "wall_seconds": driver_meta["wall_seconds"],
                      "forwards": len(layout.forwards), "evaluated_rows": len(evidence.rows)}))


if __name__ == "__main__":
    main()
