# run_experiments.py - Load experiment units, filter by task/model, call LLM, save results.
# Reads from output/experiment_units.jsonl by default, writes to output/results.jsonl.

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env before importing backend (needs API keys)
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.llm.backend import generate

DEFAULT_INPUT = "output/experiment_units.jsonl"
DEFAULT_OUTPUT = "output/results.jsonl"


def _experiment_id(unit: dict) -> str:
    """Generate a stable experiment ID from unit fields."""
    key = json.dumps(
        {
            "record_id": unit.get("record_id"),
            "task": unit.get("task"),
            "prompt_variant": unit.get("prompt_variant"),
            "model": unit.get("model"),
            "run_index": unit.get("run_index"),
        },
        sort_keys=True,
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def load_units(path: str) -> list[dict]:
    """Load experiment units from JSON or JSONL file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    units = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                units.append(json.loads(line))
    return units


def save_units(units: list[dict], path: str) -> None:
    """Save experiment units to JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for u in units:
            f.write(json.dumps(u) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiments: load units, call LLM, save results.")
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Path to read experiment units from (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Path to write LLM results to (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--task",
        choices=["classification", "summarization", "patch_fix"],
        help="Filter by task",
    )
    parser.add_argument(
        "--model",
        choices=["Claude", "Gemini", "Deepseek", "Llama"],
        help="Filter by model",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip units that already have output_text",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Max units to run (for testing)",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    units = load_units(str(input_path))
    if not units:
        print("No experiment units found.")
        return

    # Filter by task
    if args.task:
        units = [u for u in units if u.get("task") == args.task]
    # Filter by model
    if args.model:
        units = [u for u in units if u.get("model") == args.model]
    if not units:
        print("No units match the filter.")
        return

    # Load existing results for --resume
    existing_ids: set[str] = set()
    if args.resume and output_path.exists():
        for line in open(output_path):
            line = line.strip()
            if line:
                row = json.loads(line)
                if row.get("output_text"):
                    existing_ids.add(_experiment_id(row))

    # Apply limit
    to_run = []
    for u in units:
        if args.limit and len(to_run) >= args.limit:
            break
        if args.resume and _experiment_id(u) in existing_ids:
            continue
        to_run.append(u)

    # Without --resume, truncate output so we don't append to old results
    if not args.resume and output_path.exists():
        output_path.unlink()

    total = len(to_run)
    for i, unit in enumerate(to_run):
        ex_id = _experiment_id(unit)
        print(f"[{i + 1}/{total}] {unit.get('record_id', '')} | {unit.get('model', '')} | {unit.get('prompt_variant', '')}")
        try:
            output_text, tokens_used = generate(unit["prompt_text"], unit["model"])
            unit["output_text"] = output_text
            unit["tokens_used"] = tokens_used
            unit["timestamp"] = datetime.utcnow().isoformat() + "Z"
            unit["experiment_id"] = ex_id
        except Exception as e:
            print(f"  Error: {e}")
            unit["output_text"] = ""
            unit["tokens_used"] = 0
            unit["timestamp"] = datetime.utcnow().isoformat() + "Z"
            unit["experiment_id"] = ex_id
            unit["error"] = str(e)

        # Append to output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "a") as f:
            f.write(json.dumps(unit) + "\n")

    print(f"Done. Results written to {output_path}")


if __name__ == "__main__":
    main()
