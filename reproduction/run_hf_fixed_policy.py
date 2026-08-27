#!/usr/bin/env python3
"""Run a provided SAE or transcoder policy with Hugging Face Transformers."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path


DEFAULT_MAX_NEW_TOKENS = 8192
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.95


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("unguarded", "full", "selected"),
    )
    parser.add_argument(
        "--active-rule",
        action="append",
        default=[],
        help="Rule ID to activate in selected mode; repeat for multiple rules.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sampling",
        choices=("nucleus", "greedy"),
        default="nucleus",
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Forward the Qwen thinking switch to the chat template.",
    )
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


def read_jsonl(
    path: Path,
    limit: int | None = None,
    num_shards: int | None = None,
    shard_index: int | None = None,
) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            source_index = len(rows)
            rows.append((source_index, json.loads(line)))
            if limit is not None and len(rows) >= limit:
                break
    if num_shards is not None:
        rows = [row for index, row in rows if index % num_shards == shard_index]
    else:
        rows = [row for _index, row in rows]
    return rows


def select_profile(profile, mode: str, active_rules: list[str]):
    if mode == "selected":
        if not active_rules:
            raise ValueError("selected mode requires at least one --active-rule")
        for rule_id in active_rules:
            profile.rule_for(rule_id)
        return profile.with_active_rules(active_rules)
    if active_rules:
        raise ValueError("--active-rule is only used with selected mode")
    return profile


class ActionCapture:
    """Collect the single non-allow decision made for the current prompt."""

    def __init__(self) -> None:
        self.request_id = ""
        self.action = None

    def begin(self, request_id: str) -> None:
        self.request_id = request_id
        self.action = None

    def log_step(self, record: dict) -> None:
        decision = dict(record["decision"])
        if self.action is None and decision["action"] != "allow":
            self.action = {
                "request_id": self.request_id,
                "step": int(record["step"]),
                "action": decision["action"],
                "reason": decision.get("reason", ""),
                **dict(decision.get("params") or {}),
            }


def main() -> None:
    args = parse_args()
    validate_args(args)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from lmsm import build_transformers_engine, load_profile

    rows = read_jsonl(
        args.input,
        args.limit,
        args.num_shards,
        args.shard_index,
    )
    for index, row in enumerate(rows):
        if "prompt" not in row:
            raise KeyError(f"input row {index} has no prompt")

    profile = select_profile(load_profile(args.profile), args.mode, args.active_rule)
    dtype = resolve_dtype(args.torch_dtype, torch)
    set_seed(args.seed, torch)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    capture = ActionCapture()
    engine = None
    if args.mode != "unguarded":
        engine = build_transformers_engine(
            model,
            tokenizer,
            profile,
            args.profile.resolve().parents[1],
            device=args.device,
            runtime_kwargs={
                "enable_thinking": args.enable_thinking,
                "do_sample": args.sampling == "nucleus",
                "temperature": args.temperature,
                "top_p": args.top_p,
            },
            step_logger=capture,
        )

    outputs = []
    actions = []
    timings = []
    if engine is None:
        run_unguarded_rows(
            rows,
            model=model,
            tokenizer=tokenizer,
            args=args,
            torch_module=torch,
            outputs=outputs,
            timings=timings,
        )

    for index, row in enumerate(rows if engine is not None else []):
        request_id = str(row.get("id", f"row_{index:05d}"))
        capture.begin(request_id)
        set_seed(args.seed + index, torch)
        started = time.perf_counter()
        output = engine.generate(
            str(row["prompt"]),
            max_new_tokens=args.max_new_tokens,
        )
        output_tokens = token_count(tokenizer, output)
        wall_time = time.perf_counter() - started

        if capture.action is not None:
            actions.append(capture.action)
        outputs.append(
            {
                **row,
                "request_id": request_id,
                "mode": args.mode,
                "released_output": output,
                "output_token_count": output_tokens,
                "intervened": capture.action is not None,
                "action": capture.action,
            }
        )
        timings.append(
            {
                "request_id": request_id,
                "released_output_tokens": output_tokens,
                "wall_time_seconds": wall_time,
                "released_tokens_per_second": (
                    output_tokens / wall_time if wall_time else 0.0
                ),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "outputs.jsonl", outputs)
    write_jsonl(args.output_dir / "actions.jsonl", actions)
    write_timings(args.output_dir / "timings.csv", timings)
    metadata = {
        "model": args.model,
        "tokenizer": args.model,
        "profile": str(args.profile.resolve()),
        "input": str(args.input.resolve()),
        "mode": args.mode,
        "active_rules": (
            list(profile.policy.active_rule_ids) if args.mode != "unguarded" else []
        ),
        "device": args.device,
        "torch_dtype": args.torch_dtype,
        "sampling": args.sampling,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "enable_thinking": args.enable_thinking,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "prompt_count": len(outputs),
        "intervention_count": len(actions),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_unguarded_rows(
    rows,
    *,
    model,
    tokenizer,
    args,
    torch_module,
    outputs,
    timings,
) -> None:
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        started = time.perf_counter()
        generated = generate_unguarded_batch(
            model,
            tokenizer,
            [str(row["prompt"]) for row in batch_rows],
            args.device,
            args.enable_thinking,
            args.sampling == "nucleus",
            args.temperature,
            args.top_p,
            args.max_new_tokens,
            torch_module,
        )
        wall_time = time.perf_counter() - started
        for local_index, (row, (output, output_tokens)) in enumerate(
            zip(batch_rows, generated, strict=True),
            start=start,
        ):
            request_id = str(row.get("id", f"row_{local_index:05d}"))
            outputs.append(
                {
                    **row,
                    "request_id": request_id,
                    "mode": args.mode,
                    "released_output": output,
                    "output_token_count": output_tokens,
                    "intervened": False,
                    "action": None,
                }
            )
            timings.append(
                {
                    "request_id": request_id,
                    "released_output_tokens": output_tokens,
                    "wall_time_seconds": wall_time,
                    "released_tokens_per_second": (
                        output_tokens / wall_time if wall_time else 0.0
                    ),
                }
            )


def generate_unguarded_batch(
    model,
    tokenizer,
    prompts: list[str],
    device: str,
    enable_thinking: bool | None,
    do_sample: bool,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    torch_module,
) -> list[tuple[str, int]]:
    encoded = collate_prompts(
        [encode_prompt(tokenizer, prompt, enable_thinking) for prompt in prompts],
        tokenizer.pad_token_id or tokenizer.eos_token_id,
        torch_module,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_width = int(encoded["input_ids"].shape[-1])
    generation_args = {
        **encoded,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if do_sample:
        generation_args.update(temperature=temperature, top_p=top_p)
    with torch_module.inference_mode():
        generated = model.generate(**generation_args)
    results = []
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    for continuation in generated[:, prompt_width:]:
        token_ids = list(continuation.tolist())
        while token_ids and token_ids[-1] == pad_token_id:
            token_ids.pop()
        output = tokenizer.decode(token_ids, skip_special_tokens=True)
        results.append((output, token_count(tokenizer, output)))
    return results


def collate_prompts(encoded_items, pad_token_id: int, torch_module) -> dict:
    input_rows = [item["input_ids"] for item in encoded_items]
    mask_rows = [
        item.get("attention_mask", torch_module.ones_like(item["input_ids"]))
        for item in encoded_items
    ]
    width = max(int(row.shape[-1]) for row in input_rows)
    padded_ids = []
    padded_masks = []
    for input_ids, attention_mask in zip(input_rows, mask_rows, strict=True):
        padding = width - int(input_ids.shape[-1])
        if padding:
            input_ids = torch_module.cat(
                [
                    torch_module.full(
                        (1, padding),
                        pad_token_id,
                        dtype=input_ids.dtype,
                    ),
                    input_ids,
                ],
                dim=-1,
            )
            attention_mask = torch_module.cat(
                [
                    torch_module.zeros(
                        (1, padding),
                        dtype=attention_mask.dtype,
                    ),
                    attention_mask,
                ],
                dim=-1,
            )
        padded_ids.append(input_ids)
        padded_masks.append(attention_mask)
    return {
        "input_ids": torch_module.cat(padded_ids),
        "attention_mask": torch_module.cat(padded_masks),
    }


def encode_prompt(tokenizer, prompt: str, enable_thinking: bool | None):
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        **kwargs,
    )


def token_count(tokenizer, text: str) -> int:
    encoded = tokenizer(text, add_special_tokens=False)
    return len(encoded["input_ids"])


def resolve_dtype(name: str, torch_module):
    if name == "auto":
        return "auto"
    return {
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "float32": torch_module.float32,
    }[name]


def set_seed(seed: int, torch_module) -> None:
    random.seed(seed)
    torch_module.manual_seed(seed)


def validate_args(args) -> None:
    if (args.num_shards is None) != (args.shard_index is None):
        raise ValueError("--num-shards and --shard-index must be used together")
    if args.num_shards is not None and not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index is outside --num-shards")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if args.temperature <= 0 or not 0 < args.top_p <= 1:
        raise ValueError("invalid sampling parameters")
    if args.mode != "unguarded" and args.batch_size != 1:
        raise ValueError("guarded Hugging Face generation uses batch size 1")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_timings(path: Path, rows: list[dict]) -> None:
    fields = (
        "request_id",
        "released_output_tokens",
        "wall_time_seconds",
        "released_tokens_per_second",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
