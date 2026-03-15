import json
import random
from pathlib import Path
from tqdm import tqdm
import sys
import os
from dotenv import load_dotenv

# Ensure src module can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
sys.path.append(str(PROJECT_ROOT))
from src.llm.backend import generate

def extract():
    records_file = PROJECT_ROOT / "data" / "summarization" / "records.json"
    out_file = PROJECT_ROOT / "output" / "summarization" / "extracted_facts.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(records_file, "r", encoding="utf-8") as f:
        records = json.load(f)
        
    sample_records = records
    
    # Resume capability
    extracted_ids = set()
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    extracted_ids.add(data["record_id"])
                    
    prompt_template = """You are an expert software engineer.
Extract a list of atomic, objective facts about the code changes from the provided Pull Request Body and Commit Messages.
Each fact should be a single, standalone sentence describing what was added, removed, modified, or fixed.
Do not include opinions, greetings, or meta-commentary.

Input PR Body:
{body}

Input Commit Messages:
{cms}

Output a strictly formatted JSON list of strings, like this:
[
  "Added a new feature X.",
  "Fixed a bug in module Y."
]
Return ONLY the JSON list, with no markdown formatting or other text.
"""

    # Track how many we've done in this run
    added = 0
    with open(out_file, "a", encoding="utf-8") as f:
        for record in tqdm(sample_records, desc="Extracting facts"):
            rec_id = record["record_id"]
            if rec_id in extracted_ids:
                continue
                
            body = record["inputs"].get("body", "")
            cms = record["inputs"].get("cms", "")
            
            prompt = prompt_template.format(body=body, cms=cms)
            try:
                output_text, tokens = generate(prompt, model="Llama")
                
                # Cleanup markdown
                clean_output = output_text.strip()
                if clean_output.startswith("```json"):
                    clean_output = clean_output[7:]
                if clean_output.startswith("```"):
                    clean_output = clean_output[3:]
                if clean_output.endswith("```"):
                    clean_output = clean_output[:-3]
                clean_output = clean_output.strip()
                
                try:
                    facts = json.loads(clean_output)
                    if not isinstance(facts, list):
                        facts = [clean_output]
                except json.JSONDecodeError:
                    facts = [clean_output] # Save raw if fail
                    
                result = {
                    "record_id": rec_id,
                    "facts": facts,
                    "raw_output": output_text
                }
                
                f.write(json.dumps(result) + "\n")
                f.flush()
                added += 1
                
            except Exception as e:
                print(f"Failed on {rec_id}: {e}")
                
    print(f"Finished extracting facts. Added {added} new records.")

if __name__ == "__main__":
    extract()
