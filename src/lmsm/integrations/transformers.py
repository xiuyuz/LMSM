from __future__ import annotations

import math
import random
from types import SimpleNamespace

from lmsm.runtime import BaseRuntime
from lmsm.state import StepDraft, StepState
from lmsm.utils.config import split_activation_targets


class TransformersRuntime(BaseRuntime):
    def __init__(
        self,
        model,
        tokenizer,
        target_activations=None,
        *,
        decode_skip_special_tokens: bool = True,
        enable_thinking: bool | None = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        hidden_targets, hook_targets = split_activation_targets(target_activations or [])
        self.target_hidden_layers = hidden_targets
        self.target_hook_names = hook_targets
        self.decode_skip_special_tokens = decode_skip_special_tokens
        self.enable_thinking = enable_thinking
        self.do_sample = bool(do_sample)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = None if top_k is None else int(top_k)
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in the interval (0, 1]")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be at least 1 when provided")
        self.input_device = _infer_input_device(model)

    def start(self, prompt: str) -> StepState:
        encoded = _encode_prompt(
            self.tokenizer,
            prompt,
            enable_thinking=self.enable_thinking,
        )
        input_ids = _move_to_device(_extract_input_ids(encoded), self.input_device)
        draft = self._run_full_step(
            step_idx=0,
            prompt_input_ids=input_ids,
            generated_ids=_empty_like_sequence(input_ids),
            use_cache=True,
        )
        return StepState(
            step_idx=0,
            input_ids=input_ids,
            generated_ids=_empty_like_sequence(input_ids),
            cache={
                "pending_draft": draft,
                "past_key_values": draft.metadata.get("past_key_values"),
            },
            metadata={"prompt": prompt, "terminated": False},
        )

    def inspect_step(self, state: StepState) -> StepDraft:
        pending_draft = state.cache.get("pending_draft")
        if pending_draft is not None:
            return pending_draft
        return self._run_cached_decode_step(state)

    def commit(self, state: StepState, draft: StepDraft) -> StepState:
        next_token_id = draft.next_token_id
        if next_token_id is None:
            raise ValueError("No pending token. Call inspect_step() before commit().")

        state.generated_ids = _append_token(state.generated_ids, next_token_id)
        state.step_idx += 1
        state.cache["past_key_values"] = draft.metadata.get("past_key_values")
        state.cache.pop("pending_draft", None)
        return state

    def decode(self, state: StepState) -> str:
        override = state.metadata.get("output_text_override")
        if override is not None:
            return str(override)

        token_ids = _flatten_token_ids(state.generated_ids)
        if not token_ids:
            return ""
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=self.decode_skip_special_tokens,
        )

    def _run_full_step(
        self,
        *,
        step_idx: int,
        prompt_input_ids,
        generated_ids,
        use_cache: bool,
    ) -> StepDraft:
        full_input_ids = _move_to_device(
            _concat_sequences(prompt_input_ids, generated_ids),
            self.input_device,
        )
        return self._run_model_step(
            step_idx=step_idx,
            input_ids=full_input_ids,
            attention_mask=_full_attention_mask(full_input_ids),
            prompt_input_ids=prompt_input_ids,
            generated_ids=generated_ids,
            past_key_values=None,
            use_cache=use_cache,
        )

    def _run_cached_decode_step(self, state: StepState) -> StepDraft:
        past_key_values = state.cache.get("past_key_values")
        if past_key_values is None or _is_empty_sequence(state.generated_ids):
            return self._run_full_step(
                step_idx=state.step_idx,
                prompt_input_ids=state.input_ids,
                generated_ids=state.generated_ids,
                use_cache=True,
            )

        decode_input_ids = _move_to_device(_last_token_ids(state.generated_ids), self.input_device)
        return self._run_model_step(
            step_idx=state.step_idx,
            input_ids=decode_input_ids,
            attention_mask=_full_attention_mask(_concat_sequences(state.input_ids, state.generated_ids)),
            prompt_input_ids=state.input_ids,
            generated_ids=state.generated_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )

    def _run_model_step(
        self,
        *,
        step_idx: int,
        input_ids,
        attention_mask,
        prompt_input_ids,
        generated_ids,
        past_key_values,
        use_cache: bool,
    ) -> StepDraft:
        import torch

        activations: dict[str, object] = {}
        handles = self._register_hooks(activations)
        try:
            with torch.inference_mode():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=bool(self.target_hidden_layers),
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                )
        finally:
            for handle in handles:
                handle.remove()
        outputs = _as_namespace(outputs)

        logits = _last_position_logits(outputs.logits)
        next_token_id = _select_next_token(
            logits,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
        )
        activations.update(self._select_hidden_states(getattr(outputs, "hidden_states", None)))
        return StepDraft(
            step_idx=step_idx,
            input_ids=prompt_input_ids,
            generated_ids=generated_ids,
            logits=logits,
            next_token_id=next_token_id,
            activations=activations,
            metadata={
                "past_key_values": _extract_past_key_values(outputs),
            },
        )

    def _select_hidden_states(self, hidden_states):
        if hidden_states is None:
            return {}

        selected = {}
        for layer_key, layer_index in self.target_hidden_layers.items():
            tuple_index = layer_index + 1
            try:
                selected[layer_key] = hidden_states[tuple_index]
            except (IndexError, KeyError) as exc:
                raise ValueError(f"Layer {layer_key} is not present in model hidden states") from exc
        return selected

    def _register_hooks(self, activations):
        if not self.target_hook_names:
            return []
        if not hasattr(self.model, "named_modules"):
            raise RuntimeError("Hook-based activations require a model with named_modules()")

        modules = dict(self.model.named_modules())
        handles = []
        for hook_name in self.target_hook_names:
            hook_mode, module_name = _split_hook_target(hook_name)
            module = modules.get(module_name)
            if module is None:
                raise ValueError(f"Hook target {hook_name} is not present in the model")
            if hook_mode == "pre":
                handles.append(
                    module.register_forward_pre_hook(_capture_pre_hook(hook_name, activations))
                )
                continue
            handles.append(
                module.register_forward_hook(_capture_hook(hook_name, activations))
            )
        return handles


def _get_field(obj, name: str):
    if isinstance(obj, dict):
        return obj[name]
    return getattr(obj, name)


def _encode_prompt(tokenizer, prompt: str, *, enable_thinking: bool | None = None):
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        try:
            kwargs = {
                "tokenize": True,
                "add_generation_prompt": True,
                "return_dict": True,
                "return_tensors": "pt",
            }
            if enable_thinking is not None:
                kwargs["enable_thinking"] = enable_thinking
            return apply_chat_template(
                [{"role": "user", "content": prompt}],
                **kwargs,
            )
        except Exception:
            try:
                fallback_kwargs = {
                    "tokenize": True,
                    "add_generation_prompt": True,
                    "return_tensors": "pt",
                }
                if enable_thinking is not None:
                    fallback_kwargs["enable_thinking"] = enable_thinking
                return apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    **fallback_kwargs,
                )
            except Exception:
                pass
    return tokenizer(prompt, return_tensors="pt")


def _extract_input_ids(encoded):
    if isinstance(encoded, dict) or hasattr(encoded, "input_ids"):
        return _get_field(encoded, "input_ids")
    return encoded


def _infer_input_device(model):
    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_input_embeddings):
        embeddings = get_input_embeddings()
        weight = getattr(embeddings, "weight", None)
        device = getattr(weight, "device", None)
        if device is not None:
            return device

    device = getattr(model, "device", None)
    if device is not None:
        return device

    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            first_parameter = next(parameters())
        except (StopIteration, TypeError):
            return None
        return getattr(first_parameter, "device", None)

    return None


def _move_to_device(values, device):
    if device is None or not hasattr(values, "to"):
        return values
    return values.to(device)


def _as_namespace(outputs):
    if isinstance(outputs, dict):
        return SimpleNamespace(**outputs)
    return outputs


def _extract_past_key_values(outputs):
    return getattr(outputs, "past_key_values", None)


def _empty_like_sequence(sequence):
    if hasattr(sequence, "dim") and hasattr(sequence, "clone"):
        return sequence[:, :0].clone()
    if isinstance(sequence, list):
        if sequence and isinstance(sequence[0], list):
            return [[]]
        return []
    return [[]]


def _is_empty_sequence(sequence) -> bool:
    if sequence is None:
        return True
    if hasattr(sequence, "numel"):
        return bool(sequence.numel() == 0 or sequence.shape[-1] == 0)
    if isinstance(sequence, list):
        if not sequence:
            return True
        if isinstance(sequence[0], list):
            return len(sequence[0]) == 0
        return len(sequence) == 0
    return False


def _concat_sequences(prompt_ids, generated_ids):
    if _is_empty_sequence(generated_ids):
        return prompt_ids
    if hasattr(prompt_ids, "dim"):
        import torch

        return torch.cat([prompt_ids, generated_ids], dim=-1)
    return [_flatten_token_ids(prompt_ids) + _flatten_token_ids(generated_ids)]


def _last_token_ids(token_ids):
    if hasattr(token_ids, "dim"):
        return token_ids[:, -1:]
    flattened = _flatten_token_ids(token_ids)
    return [[flattened[-1]]]


def _append_token(generated_ids, token_ids):
    if _is_empty_sequence(generated_ids):
        return token_ids
    if hasattr(generated_ids, "dim"):
        import torch

        return torch.cat([generated_ids, token_ids], dim=-1)
    return [_flatten_token_ids(generated_ids) + _flatten_token_ids(token_ids)]


def _flatten_token_ids(token_ids):
    if token_ids is None:
        return []
    if hasattr(token_ids, "tolist"):
        values = token_ids.tolist()
        if values and isinstance(values[0], list):
            return list(values[0])
        return list(values)
    if isinstance(token_ids, list):
        if token_ids and isinstance(token_ids[0], list):
            return list(token_ids[0])
        return list(token_ids)
    return [int(token_ids)]


def _full_attention_mask(token_ids):
    if hasattr(token_ids, "dim"):
        import torch

        return torch.ones_like(token_ids)
    flattened = _flatten_token_ids(token_ids)
    return [[1 for _ in flattened]]


def _last_position_logits(logits):
    if hasattr(logits, "dim"):
        if logits.dim() == 3:
            return logits[:, -1, :]
        return logits
    if isinstance(logits, list):
        if logits and isinstance(logits[0], list) and logits[0] and isinstance(logits[0][0], list):
            return [logits[0][-1]]
        return logits
    return logits


def _select_next_token(
    logits,
    *,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int | None = None,
):
    if hasattr(logits, "argmax"):
        if not do_sample:
            return logits.argmax(dim=-1, keepdim=True)
        return _sample_next_token_tensor(
            logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
    rows = logits if isinstance(logits, list) else [logits]
    row = rows[0]
    if do_sample:
        return [[_sample_next_token_row(row, temperature=temperature, top_p=top_p, top_k=top_k)]]
    best_idx = max(range(len(row)), key=lambda idx: row[idx])
    return [[best_idx]]


def _sample_next_token_tensor(logits, *, temperature: float, top_p: float, top_k: int | None):
    import torch

    working = logits.float()
    if temperature != 1.0:
        working = working / temperature

    if top_k is not None:
        capped_top_k = min(top_k, int(working.shape[-1]))
        topk_values, _ = torch.topk(working, capped_top_k, dim=-1)
        cutoff = topk_values[..., -1, None]
        working = working.masked_fill(working < cutoff, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(working, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = torch.zeros_like(sorted_indices_to_remove, dtype=torch.bool)
        indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        working = working.masked_fill(indices_to_remove, float("-inf"))

    probs = torch.softmax(working, dim=-1)
    invalid_rows = torch.isnan(probs).any(dim=-1) | probs.sum(dim=-1).eq(0)
    if bool(invalid_rows.any()):
        fallback = logits.argmax(dim=-1, keepdim=True)
        sampled = torch.multinomial(
            torch.where(
                invalid_rows[:, None],
                torch.softmax(logits.float(), dim=-1),
                probs,
            ),
            num_samples=1,
        )
        return torch.where(invalid_rows[:, None], fallback, sampled)
    return torch.multinomial(probs, num_samples=1)


def _sample_next_token_row(row, *, temperature: float, top_p: float, top_k: int | None) -> int:
    if not row:
        raise ValueError("Cannot sample from an empty logits row")

    scaled = [float(value) / temperature for value in row]
    indexed = list(enumerate(scaled))
    indexed.sort(key=lambda item: item[1], reverse=True)

    if top_k is not None:
        indexed = indexed[:top_k]

    max_logit = indexed[0][1]
    exp_values = [(idx, math.exp(value - max_logit)) for idx, value in indexed]
    total = sum(value for _, value in exp_values)
    probs = [(idx, value / total) for idx, value in exp_values]

    if top_p < 1.0:
        cumulative = 0.0
        filtered: list[tuple[int, float]] = []
        for idx, prob in probs:
            filtered.append((idx, prob))
            cumulative += prob
            if cumulative >= top_p:
                break
        probs = filtered
        total = sum(value for _, value in probs)
        probs = [(idx, value / total) for idx, value in probs]

    draw = random.random()
    cumulative = 0.0
    for idx, prob in probs:
        cumulative += prob
        if draw <= cumulative:
            return idx
    return probs[-1][0]


def _capture_hook(hook_name, activations):
    def hook(_module, _inputs, output):
        activations[hook_name] = _extract_hook_activation(output)
        return output

    return hook


def _capture_pre_hook(hook_name, activations):
    def hook(_module, inputs):
        activations[hook_name] = _extract_hook_input(inputs)
        return None

    return hook


def _extract_hook_activation(output):
    if isinstance(output, tuple) and output:
        return output[0]
    return output


def _extract_hook_input(inputs):
    if isinstance(inputs, tuple) and inputs:
        return inputs[0]
    return inputs


def _split_hook_target(hook_name: str) -> tuple[str, str]:
    if hook_name.startswith("pre:"):
        return "pre", hook_name[len("pre:") :]
    return "post", hook_name
