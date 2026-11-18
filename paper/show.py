"""Print the numbers behind Section 6 from the results.json files, one line per number.

A rate over three seeds prints as its mean followed by the lowest and highest seed, as in the paper's tables.
Each folder's README maps every table cell, figure value and in-text number to its key.

Usage:
  python paper/show.py                          # every number of every folder
  python paper/show.py effectiveness            # one folder
  python paper/show.py effectiveness table5     # the numbers of one folder whose key starts with a prefix
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FOLDERS = ("mediation", "effectiveness", "released_artifacts", "cost")


def fmt(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, list):
        return "[" + ", ".join(fmt(v) for v in value) + "]"
    return str(value)


def show(node, path, prefix):
    if isinstance(node, dict) and not {"mean", "min", "max"} <= node.keys():
        for key, value in node.items():
            show(value, f"{path}.{key}" if path else str(key), prefix)
    elif path.startswith(prefix):
        value = f"{fmt(node['mean'])} ({fmt(node['min'])} to {fmt(node['max'])})" if isinstance(node, dict) else fmt(node)
        print(f"{path}: {value}")


def main(args):
    folders = [args[0]] if args else FOLDERS
    prefix = args[1] if len(args) > 1 else ""
    for folder in folders:
        results = json.loads((HERE / folder / "results.json").read_text())
        print(f"== {folder}")
        show(results, "", prefix)


if __name__ == "__main__":
    main(sys.argv[1:])
