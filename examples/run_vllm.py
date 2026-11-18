#!/usr/bin/env python3
"""Run LMSM with offline vLLM and write responses, actions, and timings."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace


DEFAULT_REFUSAL = "I can't help with that."
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--sampling", choices=("greedy", "nucleus"), default="greedy")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--mode", choices=("eager", "compiled"), default="eager")
    parser.add_argument(
        "--observation",
        choices=("hook", "graph"),
        default="hook",
        help="How LMSM reads the declared activation: a Python forward hook "
        "(eager only) or the graph-captured lmsm::capture_rows operator.",
    )
    parser.add_argument(
        "--evaluation",
        choices=("host", "device"),
        default="host",
        help="Where graph observation evaluates the policy: the host "
        "evaluators (one sync per step) or device-resident GPU state.",
    )
    parser.add_argument(
        "--cudagraph-mode",
        choices=("FULL_AND_PIECEWISE", "PIECEWISE", "FULL", "FULL_DECODE_ONLY", "NONE"),
        help="Override vLLM's CUDA-graph mode in compiled mode (default: vLLM's).",
    )
    parser.add_argument(
        "--enable-prefix-caching",
        action="store_true",
        help="Enable vLLM prefix caching (LMSM supports it with device evaluation).",
    )
    parser.add_argument(
        "--enable-chunked-prefill",
        action="store_true",
        help="Enable vLLM chunked prefill (LMSM supports it with device evaluation).",
    )
    parser.add_argument(
        "--async-scheduling",
        action="store_true",
        help="Enable vLLM async scheduling (LMSM supports it with device evaluation).",
    )
    parser.add_argument(
        "--attention-backend",
        help="vLLM attention backend (batch-invariant mode requires FLASH_ATTN).",
    )
    parser.add_argument(
        "--kv-cache-gib",
        type=float,
        help="Fixed KV-cache size (vLLM kv_cache_memory_bytes) instead of the "
        "utilization-based budget; lets several engines share one GPU safely.",
    )
    parser.add_argument(
        "--language-model-only",
        action="store_true",
        help="Multimodal checkpoints (e.g. Gemma 3): load only the language model "
        "(vLLM language_model_only; text prompts only).",
    )
    parser.add_argument(
        "--engine-process",
        choices=("inproc", "mp"),
        default="inproc",
        help="Run vLLM's engine core in this process (inproc) or in vLLM's "
        "background process (mp, vLLM's default).",
    )
    parser.add_argument(
        "--engine-defaults",
        action="store_true",
        help="Matched-native baseline only: leave chunked prefill, prefix "
        "caching, async scheduling, and the engine-core process at vLLM's "
        "defaults (the deployer's reference configuration).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-base", type=int)
    parser.add_argument("--refusal-text", default=DEFAULT_REFUSAL)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--benchmark-mode",
        choices=("policy_actions", "matched_native", "empty_extension"),
        default="policy_actions",
    )
    parser.add_argument("--benchmark-label")
    parser.add_argument(
        "--active-rule",
        dest="active_rule_ids",
        action="append",
        default=[],
        help="Activate only this rule ID; repeat to select several rules.",
    )
    parser.add_argument("--observe-only", action="store_true")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--performance-mode", action="store_true")
    parser.add_argument("--warmup-repetitions", type=int, default=1)
    parser.add_argument("--measured-repetitions", type=int, default=5)
    parser.add_argument(
        "--not-reached-disposition",
        choices=("allow_with_record", "deny"),
        help="Override the policy's declared release for requests that finish "
        "before their first evaluation point.",
    )
    test_only = parser.add_argument_group(
        "knobs for the mediation tests (off by default and recorded in metadata)"
    )
    test_only.add_argument(
        "--allow-unsupported-engine",
        action="store_true",
        help="TEST ONLY: record startup-check violations instead of refusing to start.",
    )
    test_only.add_argument(
        "--inject-drop-observation",
        dest="inject_drop_observations",
        action="append",
        default=[],
        metavar="REQUEST_ID:STEP[,STEP...]",
        help="TEST ONLY: withhold this request's observation at these steps; repeatable.",
    )
    test_only.add_argument(
        "--graph-capture-disabled-for-test",
        action="store_true",
        help="TEST ONLY: install graph observation without the capture op call.",
    )
    test_only.add_argument(
        "--inject-row-keyed-state",
        nargs="?",
        const="reset_on_admission",
        choices=("reset_on_admission", "no_reset_on_admission"),
        help="TEST ONLY: key policy state by packed batch row instead of request ID "
        "(association ablation). reset_on_admission (default) gives each admitted "
        "request a fresh slot; no_reset_on_admission lets it inherit the slot.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    if args.engine_defaults:
        if args.benchmark_mode != "matched_native":
            raise ValueError("--engine-defaults applies only to --benchmark-mode matched_native")
    elif args.engine_process == "mp":
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
    else:
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    run(args)


def run(args):
    from lmsm import load_profile
    from lmsm.execution_contract import parse_drop_observation_specs

    project_root = Path(args.project_root).resolve()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(Path(args.input))
    if args.limit is not None:
        rows = rows[: args.limit]
    request_ids = [str(row["id"]) for row in rows]
    row_seeds = _row_seeds(len(rows), args.seed, args.seed_base)

    from lmsm.composition import is_composed_profile, load_composed_profile

    args.composed = is_composed_profile(args.profile)
    profile = (
        load_composed_profile(args.profile)[0] if args.composed else load_profile(args.profile)
    )
    if args.composed and (args.active_rule_ids or args.not_reached_disposition is not None):
        raise ValueError("select a composed policy's rules in its profile, not on the command line")
    if args.active_rule_ids:
        selected = tuple(args.active_rule_ids)
        for rule_id in selected:
            profile.rule_for(rule_id)
        profile = replace(
            profile,
            policy=replace(profile.policy, active_rule_ids=selected),
        )
    if args.not_reached_disposition is not None:
        profile = replace(
            profile,
            policy=replace(
                profile.policy,
                not_reached_disposition=args.not_reached_disposition,
            ),
        )
    args.parsed_drop_observations = parse_drop_observation_specs(
        args.inject_drop_observations
    )
    unknown_injected = sorted(set(args.parsed_drop_observations) - set(request_ids))
    if unknown_injected:
        raise ValueError(f"--inject-drop-observation names unknown IDs: {unknown_injected}")

    metadata = {
        "model": args.model,
        "tokenizer": args.model,
        "profile_path": str(Path(args.profile).resolve()),
        "input_path": str(Path(args.input).resolve()),
        "request_ids": request_ids,
        "profile": asdict(profile),
        "sampling": args.sampling,
        "temperature": 0.0 if args.sampling == "greedy" else args.temperature,
        "top_p": 1.0 if args.sampling == "greedy" else args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "seed_base": args.seed_base,
        "row_seeds": row_seeds,
        "ignore_eos": args.ignore_eos,
        "enable_thinking": True,
        "scheduler": {
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "mode": args.mode,
            "observation": args.observation,
            "evaluation": args.evaluation,
            "cudagraph_mode": args.cudagraph_mode,
            "engine_defaults": args.engine_defaults,
            "engine_process": args.engine_process,
            "multiprocessing_env": os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING"),
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "chunked_prefill": None if args.engine_defaults else bool(args.enable_chunked_prefill),
            "prefix_caching": None if args.engine_defaults else bool(args.enable_prefix_caching),
            "async_scheduling": None if args.engine_defaults else bool(args.async_scheduling),
            "language_model_only": bool(args.language_model_only),
        },
        "physical_gpu": args.physical_gpu,
        "refusal_text": args.refusal_text,
        "benchmark_mode": args.benchmark_mode,
        "benchmark_label": args.benchmark_label,
        "observe_only": args.observe_only,
        "execution_contract": {
            "enabled": args.benchmark_mode == "policy_actions",
            "not_reached_disposition": profile.policy.not_reached_disposition,
            "test_knobs": {
                "allow_unsupported_engine": bool(args.allow_unsupported_engine),
                "inject_drop_observation": {
                    request_id: sorted(steps)
                    for request_id, steps in args.parsed_drop_observations.items()
                },
                "inject_row_keyed_state": args.inject_row_keyed_state,
                "graph_capture_disabled_for_test": bool(
                    args.graph_capture_disabled_for_test
                ),
            },
        },
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    started = time.perf_counter()
    init_started = time.perf_counter()
    try:
        components = _build_components(args, profile, project_root)
    except Exception as error:
        violations = getattr(error, "violations", None)
        if violations is None and args.benchmark_mode == "policy_actions":
            from lmsm.integrations.vllm import STATE

            violations = list(STATE.engine_violations) or None
        if violations is not None:
            metadata["execution_contract"]["engine_check"] = {
                "status": "refused",
                "violations": [violation.to_dict() for violation in violations],
                "violated_requirements": sorted(
                    {violation.requirement for violation in violations}
                ),
                "error": str(error),
            }
            (run_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        raise
    init_seconds = time.perf_counter() - init_started
    if components.runtime is not None:
        metadata["execution_contract"]["engine_check"] = components.engine_check
        metadata["execution_contract"]["requirements"] = (
            components.runtime.execution_requirements()
        )

    prompt_token_ids = [
        _prompt_token_ids(components.tokenizer, str(row["prompt"])) for row in rows
    ]
    prompts = [{"prompt_token_ids": token_ids} for token_ids in prompt_token_ids]
    sampling_params = [
        _sampling_params(components, args, request_id, seed)
        for request_id, seed in zip(request_ids, row_seeds, strict=True)
    ]

    if args.performance_mode:
        generated, timing_rows = _run_performance_measurements(
            components,
            prompts,
            sampling_params,
            args,
            init_seconds,
            len(profile.policy.active_rule_ids),
        )
    else:
        generate_started = time.perf_counter()
        generated = components.llm.generate(prompts, sampling_params, use_tqdm=False)
        generate_seconds = time.perf_counter() - generate_started
        output_token_count = sum(
            len(result.outputs[0].token_ids) for result in generated
        )
        timing_rows = [
            _timing_row(
                args,
                init_seconds=init_seconds,
                generate_seconds=generate_seconds,
                prompt_count=len(rows),
                output_token_count=output_token_count,
                repetition=0,
                is_warmup=False,
                rule_count=(
                    len(profile.policy.active_rule_ids)
                    if args.benchmark_mode == "policy_actions"
                    else 0
                ),
            )
        ]

    _write_outputs(
        run_dir,
        rows,
        request_ids,
        row_seeds,
        prompt_token_ids,
        generated,
        components,
    )
    _write_csv(run_dir / "timings.csv", timing_rows)
    if components.runtime is not None and os.environ.get("LMSM_DUMP_OBSERVATIONS"):
        components.runtime.worker_call(
            components.llm, "save_observation_dump", str(run_dir / "observations.npz")
        )
    if args.performance_mode:
        _write_summary(run_dir / "summary.csv", timing_rows)
    if components.runtime is not None:
        metadata["execution_contract"].update(
            _execution_summary(run_dir / "execution_records.jsonl")
        )
        metadata["execution_contract"]["runtime_counters"] = (
            components.runtime.worker_call(components.llm, "runtime_counters")
        )

    metadata["total_seconds"] = time.perf_counter() - started
    metadata["completed_prompt_count"] = len(rows)
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _build_components(args, profile, project_root: Path):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    import lmsm.integrations.vllm as runtime
    from lmsm import build_policy_evaluator

    tokenizer_kwargs = {"local_files_only": args.local_files_only}
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tokenizer_kwargs)
    llm_kwargs = {
        "model": args.model,
        "tokenizer": args.model,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "enforce_eager": args.mode == "eager",
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enable_chunked_prefill": False,
        "enable_prefix_caching": False,
        "async_scheduling": False,
        "disable_log_stats": True,
    }
    if args.async_scheduling:
        llm_kwargs["async_scheduling"] = True
    if args.enable_prefix_caching:
        llm_kwargs["enable_prefix_caching"] = True
    if args.enable_chunked_prefill:
        llm_kwargs["enable_chunked_prefill"] = True
    if args.kv_cache_gib:
        llm_kwargs["kv_cache_memory_bytes"] = int(args.kv_cache_gib * 2**30)
    if args.attention_backend:
        llm_kwargs["attention_backend"] = args.attention_backend
    if args.language_model_only:
        llm_kwargs["language_model_only"] = True
    if args.engine_defaults:
        for key in ("enable_chunked_prefill", "enable_prefix_caching", "async_scheduling"):
            llm_kwargs.pop(key)
    if args.mode == "compiled" and args.cudagraph_mode is not None:
        llm_kwargs["compilation_config"] = {"cudagraph_mode": args.cudagraph_mode}
    if args.benchmark_mode == "matched_native":
        return SimpleNamespace(
            tokenizer=tokenizer,
            llm=LLM(**llm_kwargs),
            sampling_params_cls=SamplingParams,
            runtime=None,
        )

    if args.benchmark_mode == "empty_extension":
        llm_kwargs["logits_processors"] = [runtime.MatchedEmptyExtensionProcessor]
        return SimpleNamespace(
            tokenizer=tokenizer,
            llm=LLM(**llm_kwargs),
            sampling_params_cls=SamplingParams,
            runtime=None,
        )

    # Execution contract, first stage: refuse unsupported engine arguments
    # before any model is loaded.
    argument_violations = runtime.check_engine_arguments(
        llm_kwargs,
        observation_mode=args.observation,
        allow_async_scheduling=args.evaluation == "device",
        allow_engine_process=args.evaluation == "device",
        allow_partial_prefill=args.evaluation == "device",
    )
    if argument_violations and not args.allow_unsupported_engine:
        raise runtime.UnsupportedEngineError(argument_violations)

    schedule = profile.policy.schedule
    if (
        schedule.evaluation_step is not None
        and args.max_new_tokens <= schedule.evaluation_step
    ):
        raise ValueError("max_new_tokens must exceed the policy evaluation step")

    # With vLLM's background engine core (device evaluation only), build the
    # evaluator on the CPU.  Initializing CUDA here would make vLLM spawn the
    # engine process instead of forking it.  A spawned process does not inherit
    # the configured integration, and LMSM then refuses to start.  The device
    # engine moves what it needs to the worker's GPU.
    if args.composed:
        from lmsm.composition import build_composed_evaluator

        _, evaluator = build_composed_evaluator(
            args.profile,
            project_root,
            actions_enabled=not args.observe_only,
            device="cpu" if args.engine_process == "mp" else "cuda",
        )
    else:
        evaluator = build_policy_evaluator(
            profile,
            project_root,
            actions_enabled=not args.observe_only,
            device="cpu" if args.engine_process == "mp" else "cuda",
        )
    refusal_tokens = tokenizer.encode(args.refusal_text, add_special_tokens=False)
    runtime.install_vllm_integration(
        evaluator,
        refusal_tokens,
        allow_unsupported_engine=args.allow_unsupported_engine,
        inject_drop_observations=getattr(args, "parsed_drop_observations", {}),
        inject_row_keyed_state=args.inject_row_keyed_state or False,
        observation_mode=args.observation,
        graph_capture_disabled_for_test=args.graph_capture_disabled_for_test,
        evaluation_mode=args.evaluation,
    )
    llm_kwargs["logits_processors"] = [runtime.VLLMEnforcementGate]
    llm_kwargs.update(runtime.engine_kwargs_for_observation())
    llm = LLM(**llm_kwargs)
    # Second stage: the resolved configuration (checked in the worker) and the
    # installed hook.
    engine_check = runtime.verify_engine(llm, earlier_violations=argument_violations)
    return SimpleNamespace(
        tokenizer=tokenizer,
        llm=llm,
        sampling_params_cls=SamplingParams,
        runtime=runtime,
        engine_check=engine_check,
    )


def _sampling_params(components, args, request_id: str, seed: int):
    params = components.sampling_params_cls(
        temperature=0.0 if args.sampling == "greedy" else args.temperature,
        top_p=1.0 if args.sampling == "greedy" else args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        min_tokens=0,
        ignore_eos=args.ignore_eos,
        seed=seed,
    )
    if components.runtime is not None:
        components.runtime.attach_request_id(params, request_id)
    return params


def _prompt_token_ids(tokenizer, prompt: str) -> list[int]:
    token_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def _write_outputs(
    run_dir,
    rows,
    request_ids,
    row_seeds,
    prompt_token_ids,
    generated,
    components,
):
    action_records = []
    execution_records = []
    released = None
    if components.runtime is not None:
        # One call releases every request where the worker state lives
        # (in-process, or the engine-core process through collective_rpc).
        released = components.runtime.worker_call(
            components.llm,
            "release_views",
            [
                (request_id, list(result.outputs[0].token_ids), result.outputs[0].finish_reason)
                for request_id, result in zip(request_ids, generated, strict=True)
            ],
        )
    with (run_dir / "outputs.jsonl").open("w", encoding="utf-8") as output_handle:
        for index, (row, request_id, seed, input_tokens, result) in enumerate(zip(
            rows,
            request_ids,
            row_seeds,
            prompt_token_ids,
            generated,
            strict=True,
        )):
            completion = result.outputs[0]
            ordinary_tokens = list(completion.token_ids)
            if components.runtime is None:
                view = {
                    "admitted_prefix_tokens": ordinary_tokens,
                    "released_output_tokens": ordinary_tokens,
                    "action_record": None,
                }
                feature_maxima = {}
                policy_scores = {}
                policy_score_steps = {}
                policy_actions = {}
            else:
                item = released[index]
                view = item["view"]
                feature_maxima = item["feature_maxima"]
                policy_scores = item["policy_scores"]
                policy_score_steps = item["policy_score_steps"]
                policy_actions = item["policy_actions"]

            action_record = view["action_record"]
            if action_record is not None:
                action_records.append(action_record)
            execution_record = view.get("execution_record")
            if execution_record is not None:
                execution_records.append(execution_record)
            response_tokens = list(view["released_output_tokens"])
            admitted_tokens = list(view["admitted_prefix_tokens"])
            payload = {
                **row,
                "request_id": request_id,
                "seed": seed,
                "prompt_token_ids": input_tokens,
                "ordinary_output_tokens": ordinary_tokens,
                "admitted_prefix_tokens": admitted_tokens,
                "released_output_tokens": response_tokens,
                "feature_maxima": feature_maxima,
                "policy_scores": policy_scores,
                "policy_score_steps": policy_score_steps,
                "policy_actions": policy_actions,
                "action_record": action_record,
                "admitted_prefix": components.tokenizer.decode(
                    admitted_tokens, skip_special_tokens=True
                ),
                "released_output": components.tokenizer.decode(
                    response_tokens, skip_special_tokens=True
                ),
                "finish_reason": completion.finish_reason,
                "stop_reason": completion.stop_reason,
            }
            if execution_record is not None:
                payload["execution_record"] = execution_record
            output_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    with (run_dir / "actions.jsonl").open("w", encoding="utf-8") as action_handle:
        for action in action_records:
            action_handle.write(json.dumps(action, ensure_ascii=False) + "\n")
    if components.runtime is not None:
        with (run_dir / "execution_records.jsonl").open(
            "w", encoding="utf-8"
        ) as record_handle:
            for record in execution_records:
                record_handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _execution_summary(path: Path) -> dict:
    from collections import Counter

    records = _read_jsonl(path)
    return {
        "record_count": len(records),
        "state_counts": dict(sorted(Counter(row["state"] for row in records).items())),
        "release_counts": dict(
            sorted(Counter(row["release_outcome"] for row in records).items())
        ),
        "decision_counts": dict(
            sorted(Counter(str(row["decision"]) for row in records).items())
        ),
    }


def _run_performance_measurements(
    components,
    prompts,
    sampling_params,
    args,
    init_seconds,
    active_rule_count,
):
    import torch

    timing_rows = []
    last_generated = None
    repetitions = [True] * args.warmup_repetitions + [
        False
    ] * args.measured_repetitions
    for repetition, is_warmup in enumerate(repetitions):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        encode_before = _backend_evaluation_count(components)
        profile_path = os.environ.get("LMSM_TORCH_PROFILE")
        profiler = None
        if profile_path and repetition == len(repetitions) - 1:
            # Diagnostic only: profile the last measured repetition.
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            )
            profiler.__enter__()
        started = time.perf_counter()
        generated = components.llm.generate(prompts, sampling_params, use_tqdm=False)
        torch.cuda.synchronize()
        generate_seconds = time.perf_counter() - started
        if profiler is not None:
            profiler.__exit__(None, None, None)
            Path(profile_path).write_text(
                profiler.key_averages().table(sort_by="cuda_time_total", row_limit=60)
                + "\n\n"
                + profiler.key_averages().table(sort_by="cpu_time_total", row_limit=40),
                encoding="utf-8",
            )
        output_token_count = sum(
            len(result.outputs[0].token_ids) for result in generated
        )
        encode_after = _backend_evaluation_count(components)
        row = _timing_row(
            args,
            init_seconds=init_seconds,
            generate_seconds=generate_seconds,
            prompt_count=len(prompts),
            output_token_count=output_token_count,
            repetition=repetition,
            is_warmup=is_warmup,
            rule_count=(
                active_rule_count if args.benchmark_mode == "policy_actions" else 0
            ),
        )
        row.update(
            {
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "encode_calls": (
                    encode_after - encode_before
                    if encode_before is not None and encode_after is not None
                    else None
                ),
            }
        )
        timing_rows.append(row)
        if not is_warmup:
            last_generated = generated
    return last_generated, timing_rows


def _timing_row(
    args,
    *,
    init_seconds,
    generate_seconds,
    prompt_count,
    output_token_count,
    repetition,
    is_warmup,
    rule_count,
):
    return {
        "configuration": args.benchmark_label or args.benchmark_mode,
        "benchmark_mode": args.benchmark_mode,
        "rule_count": rule_count,
        "max_num_seqs": args.max_num_seqs,
        "repetition": repetition,
        "is_warmup": is_warmup,
        "init_seconds": init_seconds,
        "generate_seconds": generate_seconds,
        "total_seconds": init_seconds + generate_seconds,
        "prompt_count": prompt_count,
        "raw_output_token_count": output_token_count,
        "model_tokens_per_second": output_token_count / generate_seconds,
        "milliseconds_per_model_token": (
            1000.0 * generate_seconds / output_token_count
        ),
    }


def _backend_evaluation_count(components):
    if components.runtime is None:
        return None
    return components.runtime.backend_evaluation_count()


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _row_seeds(row_count: int, seed: int, seed_base: int | None) -> list[int]:
    if seed_base is None:
        return [seed] * row_count
    return [seed_base + index for index in range(row_count)]


def _write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, timing_rows: list[dict]):
    measured = [row for row in timing_rows if not row["is_warmup"]]
    throughput = sorted(row["model_tokens_per_second"] for row in measured)
    latency = sorted(row["milliseconds_per_model_token"] for row in measured)
    summary = {
        "configuration": measured[0]["configuration"],
        "benchmark_mode": measured[0]["benchmark_mode"],
        "rule_count": measured[0]["rule_count"],
        "max_num_seqs": measured[0]["max_num_seqs"],
        "measurement_count": len(measured),
        "median_model_tokens_per_second": _quantile(throughput, 0.5),
        "iqr_model_tokens_per_second": (
            _quantile(throughput, 0.75) - _quantile(throughput, 0.25)
        ),
        "median_milliseconds_per_model_token": _quantile(latency, 0.5),
        "iqr_milliseconds_per_model_token": (
            _quantile(latency, 0.75) - _quantile(latency, 0.25)
        ),
    }
    _write_csv(path, [summary])


def _quantile(values, probability):
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


if __name__ == "__main__":
    main()
