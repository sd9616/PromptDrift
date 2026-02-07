# Scripts

Documentation for `run_experiments.py` and `generate_units.py`.

---

## run_experiments.py

Load experiment units, filter by task/model, call LLM, save results to `output/results.jsonl`.

### Arguments

| Argument   | Default                         | Description                                                    |
| ---------- | ------------------------------- | -------------------------------------------------------------- |
| `--input`  | `output/experiment_units.jsonl` | Path to read experiment units from                             |
| `--output` | `output/results.jsonl`          | Path to write LLM results to (units + output_text, tokens_used)|
| `--task`   | —                               | Filter by task: `classification`, `summarization`, `patch_fix` |
| `--model`  | —                               | Filter by model: `Claude`, `Gemini`, `Deepseek`, `Llama`       |
| `--resume` | —                               | Skip units that already have `output_text`                     |
| `--limit`  | —                               | Max units to run (for testing)                                 |

### Examples

- Run everything (uses defaults): `python scripts/run_experiments.py`
- Override input: `python scripts/run_experiments.py --input my_units.jsonl`
- Override output: `python scripts/run_experiments.py --output my_results.jsonl`
- Run one task: `python scripts/run_experiments.py --task classification`
- Run one model: `python scripts/run_experiments.py --model Claude`
- Test run (10 units): `python scripts/run_experiments.py --limit 10`
- Resume after crash: `python scripts/run_experiments.py --resume`

---

## generate_units.py

Load records from `data/<task>/records.json` and prompts from `prompts/<task>/`, expand across models and runs, write `output/experiment_units.jsonl`.

### Arguments

| Argument   | Default                         | Description                          |
| ---------- | ------------------------------- | ------------------------------------ |
| `--output` | `output/experiment_units.jsonl` | Path to write experiment units to    |
| `--task`   | —                               | Generate for a single task only      |
| `--runs`   | 3                               | Number of runs per unit              |

### Examples

- Generate all units: `python scripts/generate_units.py`
- Generate for one task: `python scripts/generate_units.py --task classification`
- Use 5 runs per unit: `python scripts/generate_units.py --runs 5`
