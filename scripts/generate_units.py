# generate_units.py - Load records and prompts, expand across models/variants/runs,
# write output/experiment_units.jsonl for run_experiments.py.

# Output format:
# Classification example:
# {
#   "record_id": "issue_123",
#   "task": "classification",
#   "prompt_variant": "I1",
#   "model": "Claude",
#   "run_index": 1,
#   "prompt_text": "Classify the following issue as one of: Bug, Feature Request, or Question.\n\nTitle: NullPointerException when config file is missing\nBody: When trying to start the server without a config file...\n\nReturn only the label.",
#   "ground_truth": "Bug"
# }

# Summarization example:
# {
#   "record_id": "elastic/elasticsearch_37980",
#   "task": "summarization",
#   "prompt_variant": "S1",
#   "model": "Gemini",
#   "run_index": 1,
#   "prompt_text": "Summarize the following pull request in a concise paragraph.\n\nDescription: Add support for X and fix Y.\nCommit Messages:\nfix: resolve issue\nfeat: add feature\n\nProvide a brief summary of the main changes.",
#   "ground_truth": null
# }

# Patch fix example:
# {
#   "record_id": "Lang_1",
#   "task": "patch_fix",
#   "prompt_variant": "P1",
#   "model": "Llama",
#   "run_index": 1,
#   "prompt_text": "Fix the bug in the following code.\n\nBuggy Code:\npublic int divide(int a, int b) { return a / b; }\n\nReturn only the corrected code.",
#   "ground_truth": "public int divide(int a, int b) { if (b == 0) throw new IllegalArgumentException(); return a / b; }"
# }

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TASKS = ["classification", "summarization", "patch_fix"]
PROMPT_VARIANTS = {
    "classification": ["I1", "I2", "I3", "I4"],
    "summarization": ["S1", "S2", "S3", "S4"],
    "patch_fix": ["P1", "P2", "P3", "P4"],
}
MODELS = ["Claude", "Gemini", "Deepseek", "Llama"]
NUM_RUNS = 3

DEFAULT_OUTPUT = "output/experiment_units.jsonl"


def load_records(task: str) -> list[dict]:
    """Load normalized records from data/<task>/records.json."""
    path = PROJECT_ROOT / "data" / task / "records.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def load_prompt(task: str, variant: str) -> str:
    """Load prompt template from prompts/<task>/<variant>.txt."""
    path = PROJECT_ROOT / "prompts" / task / f"{variant}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text()


def fill_prompt(template: str, inputs: dict) -> str:
    """Replace placeholders in template with values from inputs."""
    text = template
    for key, value in inputs.items():
        text = text.replace("{" + key + "}", str(value or ""))
    return text


def format_cms_as_bullets(cms) -> str:
    """Format commit messages as one sub-bullet per message (for S4).
    Supports PRSummarizer format (cms is a list) or newline-separated string."""
    if cms is None:
        return ""
    if isinstance(cms, list):
        parts = [str(item).strip() for item in cms if str(item).strip()]
        return "\n".join("  • " + item for item in parts)
    lines = [line.strip() for line in str(cms).splitlines() if line.strip()]
    return "\n".join("  • " + line for line in lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate experiment units from records and prompts.")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Path to write experiment units (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--task",
        choices=TASKS,
        help="Generate for a single task only",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=NUM_RUNS,
        help=f"Number of runs per unit (default: {NUM_RUNS})",
    )
    args = parser.parse_args()

    tasks = [args.task] if args.task else TASKS
    output_path = PROJECT_ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    units = []
    for task in tasks:
        records = load_records(task)
        if not records:
            print(f"No records for {task}, skipping.")
            continue
        variants = PROMPT_VARIANTS[task]
        for record in records:
            record_id = record.get("record_id", "")
            inputs = record.get("inputs", {})
            for variant in variants:
                try:
                    template = load_prompt(task, variant)
                except FileNotFoundError:
                    continue
                fill_inputs = dict(inputs)
                if task == "summarization":
                    cms = inputs.get("cms", "")
                    if variant == "S4":
                        fill_inputs["cms"] = format_cms_as_bullets(cms)
                    elif isinstance(cms, list):
                        fill_inputs["cms"] = "\n".join(str(m).strip() for m in cms if str(m).strip())
                prompt_text = fill_prompt(template, fill_inputs)
                for model in MODELS:
                    for run_index in range(args.runs):
                        units.append({
                            "record_id": record_id,
                            "task": task,
                            "prompt_variant": variant,
                            "model": model,
                            "run_index": run_index + 1,
                            "prompt_text": prompt_text,
                            "ground_truth": record.get("ground_truth"),
                        })

    with open(output_path, "w") as f:
        for u in units:
            f.write(json.dumps(u) + "\n")

    print(f"Wrote {len(units)} experiment units to {output_path}")


if __name__ == "__main__":
    main()
