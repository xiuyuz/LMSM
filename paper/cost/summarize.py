"""Turn the timing sessions under $RUNS/cost into results.json, the numbers of Figure 7 and Section 6.5.

Each ratio divides the median throughput of an arm's ten measured repetitions (five per round, two rounds) by the
median of native vLLM on the same model in the same session. A session must have no foreign_processes.log entry.
Usage: python summarize.py [RUNS]
"""

import csv
import json
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RUNS", HERE.parent / "runs")) / "cost"
WIDTHS = (1, 32, 128)
DEFAULTS = {"probes_checkpoint": "compiled_checkpoint_graph_device_vdefaults",
            "probes_anytime": "compiled_anytime_graph_device_vdefaults",
            "transcoder": "compiled_transcoder_graph_device_vdefaults",
            "sae": "gemma_compiled_sae_graph_device_vdefaults"}
MATCHED = {"probes_checkpoint": "compiled_checkpoint_graph_device", "probes_anytime": "compiled_anytime_graph_device",
           "transcoder": "compiled_transcoder_graph_device", "sae": "gemma_compiled_sae_graph_device"}
ALTERNATIVES = {"matched_settings": ("alternatives_probes", "compiled_native"),
                "python_hook": ("alternatives_probes", "eager_checkpoint_hook_host"),
                "transcoder_host": ("alternatives_transcoder", "compiled_transcoder_graph_host"),
                "sae_host": ("alternatives_sae", "gemma_compiled_sae_graph_host")}
SIZES = (4, 8, 14, 32)


def runs(session, arm, width):
    """Tokens per second of the ten measured repetitions of one arm."""
    out = []
    for rnd in ("round1", "round2"):
        with open(RUNS / session / rnd / f"{arm}_w{width}" / "timings.csv") as f:
            out += [float(r["model_tokens_per_second"]) for r in csv.DictReader(f) if r["is_warmup"] == "False"]
    if len(out) != 10:
        sys.exit(f"{session} {arm} width {width}: {len(out)} measured repetitions, expected 10")
    return out


def native(arm, settings):
    prefix = "gemma_" if arm.startswith("gemma_") else ""
    return f"{prefix}compiled_native{'_defaults' if settings == 'defaults' else ''}"


def ratio(session, arm, base, width, spread=False):
    """Percent of native throughput, and optionally every repetition of both arms relative to the native median."""
    lmsm, nat = runs(session, arm, width), runs(session, base, width)
    med = statistics.median(nat)
    pct = 100 * statistics.median(lmsm) / med
    if not spread:
        return pct
    return pct, [100 * x / med for x in lmsm], [100 * x / med for x in nat]


def main():
    for session in sorted(p for p in RUNS.iterdir() if (p / "round1").is_dir()):
        log = session / "foreign_processes.log"
        if log.exists() and log.read_text().strip():
            sys.exit(f"{session.name}: another process used the GPU during a measurement ({log})")

    out = {"widths": list(WIDTHS), "defaults": {}, "matched": {}, "model_size": {}, "alternatives": {}}
    for key, arm in DEFAULTS.items():
        cells = [ratio("defaults", arm, native(arm, "defaults"), w, spread=True) for w in WIDTHS]
        out["defaults"][key] = {"pct": [c[0] for c in cells], "runs_pct": [c[1] for c in cells],
                                "native_runs_pct": [c[2] for c in cells]}
    for key, arm in MATCHED.items():
        out["matched"][key] = [ratio("matched", arm, native(arm, "matched"), w) for w in WIDTHS]
    for size in SIZES:
        cells = [ratio(f"size_{size}b", "compiled_checkpoint_graph_device_vdefaults", "compiled_native_defaults", w,
                       spread=True) for w in (32, 128)]
        out["model_size"][str(size)] = {"widths": [32, 128], "pct": [c[0] for c in cells],
                                        "runs_pct": [c[1] for c in cells]}
    for key, (session, arm) in ALTERNATIVES.items():
        out["alternatives"][key] = [ratio(session, arm, native(arm, "defaults"), w) for w in WIDTHS]

    defaults = [x for v in out["defaults"].values() for x in v["pct"]]
    probes = out["matched"]["probes_checkpoint"] + out["matched"]["probes_anytime"]
    larger = [out["model_size"][str(s)]["pct"][0] for s in SIZES[1:]]
    fitted = out["defaults"]["probes_checkpoint"]["pct"][1:]
    alt = out["alternatives"]
    out["text"] = {
        "defaults_lowest_pct": min(defaults),
        "matched_probes_pct": [min(probes), max(probes)],
        "matched_settings_loss_pct": [100 - max(alt["matched_settings"]), 100 - min(alt["matched_settings"])],
        "transcoder_host_width128_pct": alt["transcoder_host"][2],
        "sae_host_width128_pct": alt["sae_host"][2],
        "python_hook_pct": [min(alt["python_hook"]), max(alt["python_hook"])],
        "larger_models_width32_pct": [min(larger), max(larger)],
        "qwen3_4b_width32_pct": out["model_size"]["4"]["pct"][0],
        "synthetic_minus_fitted_points": max(abs(a - b) for a, b in zip(out["model_size"]["4"]["pct"], fitted)),
    }
    (HERE / "results.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["text"], indent=1))


if __name__ == "__main__":
    main()
