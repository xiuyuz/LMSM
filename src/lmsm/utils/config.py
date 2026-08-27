from __future__ import annotations


def parse_layer_key(layer_key: str) -> int:
    prefix = "layer_"
    if not layer_key.startswith(prefix):
        raise ValueError(f"Layer keys must look like 'layer_20', got: {layer_key}")
    return int(layer_key[len(prefix) :])


def normalize_layer_key_map(target_layers) -> dict[str, int]:
    layer_map: dict[str, int] = {}
    for item in target_layers:
        if isinstance(item, int):
            layer_map[f"layer_{item}"] = item
        else:
            layer_map[str(item)] = parse_layer_key(str(item))
    return dict(sorted(layer_map.items(), key=lambda pair: pair[1]))


def split_activation_targets(targets) -> tuple[dict[str, int], list[str]]:
    hidden_targets = normalize_layer_key_map(
        [target for target in targets if _is_layer_key(str(target))]
    )
    hook_targets = [str(target) for target in targets if not _is_layer_key(str(target))]
    return hidden_targets, list(dict.fromkeys(hook_targets))


def collect_required_activation_keys(*spec_groups) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for specs in spec_groups:
        for spec in specs:
            activation_key = str(
                spec.get("activation_key")
                or spec.get("hook_name")
                or spec.get("layer_key")
            )
            if not activation_key or activation_key == "None":
                continue
            if activation_key not in seen:
                ordered.append(activation_key)
                seen.add(activation_key)
    hidden_targets, hook_targets = split_activation_targets(ordered)
    return list(hidden_targets.keys()) + hook_targets


def _is_layer_key(value: str) -> bool:
    return value.startswith("layer_")
