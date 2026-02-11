# Classification data loader - reads raw issue data and outputs normalized records.json.
# Each record has record_id, task, inputs (title, body), ground_truth (label).
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
    return {
        "record_id": record.get("record_id", record.get("id", "")),
        "task": "classification",
        "inputs": {
            "title": record.get("title", record.get("inputs", {}).get("title", "")),
            "body": record.get("body", record.get("inputs", {}).get("body", "")),
        },
        "ground_truth": record.get("ground_truth", record.get("label", "")),
    }


def main() -> None:
    """Load raw classification data and write records.json."""
    raw_dir = PROJECT_ROOT / "data" / "classification" / "raw"
    out_path = PROJECT_ROOT / "data" / "classification" / "records.json"

    records = []
    for f in raw_dir.glob("*.json"):
        for r in load_raw(f):
            records.append(transform(r))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Wrote {len(records)} classification records to {out_path}")


if __name__ == "__main__":
    main()
