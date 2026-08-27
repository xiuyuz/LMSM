from __future__ import annotations


def last_token_representation(hidden):
    if hasattr(hidden, "dim"):
        if hidden.dim() == 3:
            return hidden[:, -1, :]
        if hidden.dim() == 2:
            return hidden[-1]
    if isinstance(hidden, list):
        if not hidden:
            return hidden
        if isinstance(hidden[0], list):
            if hidden and hidden[0] and isinstance(hidden[0][0], list):
                return hidden[0][-1]
            return hidden[-1]
    return hidden


def take_feature(values, index: int):
    if hasattr(values, "dim"):
        if values.dim() == 2:
            return values[0, index]
        return values[index]
    if hasattr(values, "__getitem__"):
        selected = values[index]
        if is_scalar_like(selected):
            return selected
        if hasattr(selected, "__getitem__"):
            return selected[0]
    raise TypeError("Values do not support feature indexing")


def coerce_float(value) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (list, tuple)):
            return coerce_float(value[0][0])
        if value:
            return coerce_float(value[0])
    return float(value)


def is_scalar_like(value) -> bool:
    return not isinstance(value, (list, tuple))


def to_row_vector(value):
    if hasattr(value, "dim"):
        if value.dim() == 1:
            return value.unsqueeze(0)
        return value
    if isinstance(value, list) and value and not isinstance(value[0], list):
        return [value]
    return value
