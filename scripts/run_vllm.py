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
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    run(args)


def run(args):
    from lmsm import load_profile

    project_root = Path(args.project_root).resolve()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(Path(args.input))
    if args.limit is not None:
        rows = rows[: args.limit]
    request_ids = [str(row["id"]) for row in rows]
    row_seeds = _row_seeds(len(rows), args.seed, args.seed_base)

    profile = load_profile(args.profile)
    if args.active_rule_ids:
        selected = tuple(args.active_rule_ids)
        for rule_id in selected:
            profile.rule_for(rule_id)
        profile = replace(
            profile,
            policy=replace(profile.policy, active_rule_ids=selected),
        )

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
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "chunked_prefill": False,
            "prefix_caching": False,
            "async_scheduling": False,
        },
        "physical_gpu": args.physical_gpu,
        "refusal_text": args.refusal_text,
        "benchmark_mode": args.benchmark_mode,
        "benchmark_label": args.benchmark_label,
        "observe_only": args.observe_only,
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    started = time.perf_counter()
    init_started = time.perf_counter()
    components = _build_components(args, profile, project_root)
    init_seconds = time.perf_counter() - init_started

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
    if args.performance_mode:
        _write_summary(run_dir / "summary.csv", timing_rows)

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

    schedule = profile.policy.schedule
    if (
        schedule.evaluation_step is not None
        and args.max_new_tokens <= schedule.evaluation_step
    ):
        raise ValueError("max_new_tokens must exceed the policy evaluation step")

    evaluator = build_policy_evaluator(
        profile,
        project_root,
        actions_enabled=not args.observe_only,
        device="cuda",
    )
    refusal_tokens = tokenizer.encode(args.refusal_text, add_special_tokens=False)
    runtime.install_vllm_integration(evaluator, refusal_tokens)
    llm_kwargs["logits_processors"] = [runtime.VLLMEnforcementGate]
    return SimpleNamespace(
        tokenizer=tokenizer,
        llm=LLM(**llm_kwargs),
        sampling_params_cls=SamplingParams,
        runtime=runtime,
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
    with (run_dir / "outputs.jsonl").open("w", encoding="utf-8") as output_handle:
        for row, request_id, seed, input_tokens, result in zip(
            rows,
            request_ids,
            row_seeds,
            prompt_token_ids,
            generated,
            strict=True,
        ):
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
                view = components.runtime.buffered_output(request_id, ordinary_tokens)
                feature_maxima = components.runtime.feature_maxima(request_id)
                policy_scores = components.runtime.policy_scores(request_id)
                policy_score_steps = components.runtime.policy_score_steps(request_id)
                policy_actions = components.runtime.policy_actions(request_id)

            action_record = view["action_record"]
            if action_record is not None:
                action_records.append(action_record)
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
            output_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    with (run_dir / "actions.jsonl").open("w", encoding="utf-8") as action_handle:
        for action in action_records:
            action_handle.write(json.dumps(action, ensure_ascii=False) + "\n")


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
        started = time.perf_counter()
        generated = components.llm.generate(prompts, sampling_params, use_tqdm=False)
        torch.cuda.synchronize()
        generate_seconds = time.perf_counter() - started
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
