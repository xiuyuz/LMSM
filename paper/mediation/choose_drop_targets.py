"""Choose the 8 requests whose observation the dropped-observation test removes, from the uninjected device run.

The targets are requests the policy allowed, so a failure changes no other request's generation length and the
batches stay the same. They are spaced evenly over the eligible requests in input order.
Checkpoint drops steps 0, 1, 17, 32, 33, 48, 63 and 64 (the prompt, both window edges and the checkpoint) from
requests that generated at least 65 tokens. Anytime drops step (k + 1) * N // 9 of the k-th target (N generated
tokens, at least 9). If fewer than 8 allowed requests qualify, refused requests fill the rest, at a step no later
than their refusal.
Usage: python choose_drop_targets.py OUTPUTS.jsonl checkpoint|anytime TARGETS.json    (prints the runner options)
"""

import json
import sys
from pathlib import Path

CHECKPOINT_STEPS = (0, 1, 17, 32, 33, 48, 63, 64)
COUNT = 8


def evenly_spaced(items, count):
    return list(items) if len(items) <= count else [items[(i * len(items)) // count] for i in range(count)]


def main(outputs: Path, policy: str, out: Path):
    rows = [json.loads(line) for line in outputs.read_text(encoding="utf-8").splitlines() if line.strip()]
    minimum = 65 if policy == "checkpoint" else 9
    allowed = [r for r in rows if r["action_record"] is None and len(r["ordinary_output_tokens"]) >= minimum]
    chosen = [(r, False) for r in evenly_spaced(allowed, COUNT)]
    if len(chosen) < COUNT:
        chosen += [(r, True) for r in rows if r["action_record"] is not None][: COUNT - len(chosen)]
    targets = []
    for index, (row, refused) in enumerate(chosen):
        generated = len(row["ordinary_output_tokens"])
        if policy == "checkpoint":
            step = CHECKPOINT_STEPS[index % len(CHECKPOINT_STEPS)]
            if refused:
                step = min(step, row["action_record"]["step"])
        else:
            step = max(1, ((index + 1) * generated) // 9)
            if refused:
                step = max(1, row["action_record"]["step"] // 2)
        targets.append({"request_id": row["request_id"], "drop_step": step, "refused_request": refused})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"policy": policy, "targets": targets}, indent=2) + "\n", encoding="utf-8")
    print(" ".join(f"--inject-drop-observation={t['request_id']}:{t['drop_step']}" for t in targets))


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))
