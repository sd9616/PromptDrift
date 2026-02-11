# Patch fix data loader - reads raw buggy code and outputs normalized records.json.
# Each record has record_id, task, inputs (buggy_code), ground_truth (corrected code).
#Writes all normalized records to data/<task>/records.json.

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_raw(path: Path) -> list[dict]:
    """Load raw data from path (JSON or JSONL)."""
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def transform(record: dict) -> dict:
    """Transform raw record to normalized format."""
    inputs = record.get("inputs", {})
    return {
        "record_id": record.get("record_id", record.get("id", "")),
        "task": "patch_fix",
        "inputs": {
            "buggy_code": record.get("buggy_code", inputs.get("buggy_code", "")),
        },
        "ground_truth": record.get("ground_truth", record.get("corrected_code", "")),
    }


def main() -> None:
    """Load raw patch fix data and write records.json."""
    raw_dir = PROJECT_ROOT / "data" / "patch_fix" / "raw"
    out_path = PROJECT_ROOT / "data" / "patch_fix" / "records.json"

    records = []
    for f in raw_dir.glob("*.json"):
        for r in load_raw(f):
            records.append(transform(r))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Wrote {len(records)} patch fix records to {out_path}")


if __name__ == "__main__":
    main()
