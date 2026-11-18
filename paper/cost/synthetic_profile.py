"""Write a Checkpoint profile with synthetic probe parameters for a larger Qwen3 model (Figure 7b).

The device engine's work per step depends on the hidden width, the 15 rules and the schedule, not on parameter values.
The profile therefore keeps the released Checkpoint profile and changes only what depends on the model.
- The activation site is the MLP input at the same relative depth as Qwen3-4B's layer 24 of 36.
- Each window probe has mean 0, scale 1 and coefficients drawn from N(0, 0.001) with seed 0, and intercept -10.
- Each rule keeps its released fusion parameters, and its threshold is 1e6, which no score reaches.

Usage: python synthetic_profile.py MODEL KEY OUT_DIR     (MODEL is a Hugging Face ID or a local model directory)
Prints the path of the profile.
"""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from transformers import AutoConfig

REPO = Path(__file__).resolve().parents[2]


def main(model: str, key: str, out_dir: Path) -> Path:
    config = AutoConfig.from_pretrained(model)
    width = int(config.hidden_size)
    layer = round(int(config.num_hidden_layers) * 24 / 36)
    released = yaml.safe_load((REPO / "profiles/lmsm_checkpoint.yaml").read_text())
    fitted = json.loads((REPO / "profiles/parameters/lmsm_checkpoint.json").read_text())
    name = f"LMSM-Checkpoint-synthetic-{key}"
    rng = np.random.default_rng(0)

    def window():
        return {"scaler": {"mean": [0.0] * width, "scale": [1.0] * width},
                "classifier": {"coef": [float(x) for x in rng.normal(0.0, 1e-3, width)], "intercept": -10.0}}

    params = {"name": name, "activation_key": f"pre:model.layers.{layer}.mlp", "synthetic": True, "rules": {}}
    for target in (t["id"] for t in released["targets"]):
        params["rules"][target] = {"threshold": 1e6, "prompt": window(), "steps_1_32": window(),
                                   "steps_33_64": window(), "fusion": fitted["rules"][target]["fusion"]}

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = (out_dir / f"lmsm_checkpoint_synthetic_{key}.json").resolve()
    yaml_path = json_path.with_suffix(".yaml")
    json_path.write_text(json.dumps(params))
    profile = copy.deepcopy(released)
    binding = profile["backend_bindings"][0]
    binding["activation_key"] = params["activation_key"]
    binding["report_path"] = str(json_path)
    binding["config"] = {"model_id": model, "hidden_width": width}
    profile["name"] = profile["policy"]["name"] = name
    yaml_path.write_text("# Synthetic Checkpoint profile for cost measurements only.\n"
                         + yaml.safe_dump(profile, sort_keys=False))
    return yaml_path


if __name__ == "__main__":
    print(main(sys.argv[1], sys.argv[2], Path(sys.argv[3])))
