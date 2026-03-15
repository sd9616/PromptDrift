import json
from pathlib import Path
from tqdm import tqdm
import sys
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
sys.path.append(str(PROJECT_ROOT))
from src.llm.backend import generate

def verify():
    facts_file = PROJECT_ROOT / "output" / "summarization" / "extracted_facts.jsonl"
    summaries_file = PROJECT_ROOT / "output" / "summarization" / "combined_llama_gemini_s1_s4.jsonl"
    out_file = PROJECT_ROOT / "output" / "summarization" / "verified_facts.jsonl"
    
    if not facts_file.exists():
        print(f"File not found: {facts_file}")
        return
        
    extracted_facts = {}
    with open(facts_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                extracted_facts[data["record_id"]] = data["facts"]
                
    if not extracted_facts:
        print("No facts extracted yet.")
        return
        
    print(f"Loaded facts for {len(extracted_facts)} records.")
    
    # Load summaries for these records
    summaries_to_verify = []
    with open(summaries_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if data["record_id"] in extracted_facts:
                    summaries_to_verify.append(data)
                    
    print(f"Found {len(summaries_to_verify)} summaries to verify.")
    
    # Resume capability
    verified_keys = set()
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    key = f"{data['record_id']}_{data['model']}_{data['prompt_variant']}"
                    verified_keys.add(key)
                    
    prompt_template = """You are an expert factual consistency evaluator.
You will be given a list of Ground-Truth Facts and a Generated Summary.

Your tasks:
1. Examine each Ground-Truth Fact. If the information is NOT present in the Summary, it is an "omitted_fact".
2. Examine the Generated Summary. If it contains specific claims about the code changes that are not supported by the Ground-Truth Facts, it is a "hallucinated_fact".

Ground-Truth Facts:
{facts}

Generated Summary:
{summary}

Count the number of omitted_facts and hallucinated_facts.
Output a strictly formatted JSON object exactly like this:
{{
  "omitted_facts": ["fact 1 that was missed", "fact 2 that was missed"],
  "hallucinated_facts": ["unsupported claim 1"]
}}
Return ONLY the JSON object, with no markdown formatting.
"""

    added = 0
    with open(out_file, "a", encoding="utf-8") as f:
        for summary_data in tqdm(summaries_to_verify, desc="Verifying summaries"):
            rec_id = summary_data["record_id"]
            model = summary_data["model"]
            variant = summary_data["prompt_variant"]
            
            key = f"{rec_id}_{model}_{variant}"
            if key in verified_keys:
                continue
                
            facts = extracted_facts[rec_id]
            summary_text = summary_data.get("output_text", "")
            if not summary_text:
                continue
            
            if not isinstance(facts, list):
                facts = [facts]
            
            # Format facts as a numbered list
            facts_str = "\n".join([f"{i+1}. {fact}" for i, fact in enumerate(facts)])
            
            prompt = prompt_template.format(facts=facts_str, summary=summary_text)
            
            try:
                output_text, tokens = generate(prompt, model="Llama")
                
                clean_output = output_text.strip()
                if clean_output.startswith("```json"):
                    clean_output = clean_output[7:]
                if clean_output.startswith("```"):
                    clean_output = clean_output[3:]
                if clean_output.endswith("```"):
                    clean_output = clean_output[:-3]
                clean_output = clean_output.strip()
                
                try:
                    result_json = json.loads(clean_output)
                    omitted = result_json.get("omitted_facts", [])
                    hallucinated = result_json.get("hallucinated_facts", [])
                    if not isinstance(omitted, list): omitted = [omitted]
                    if not isinstance(hallucinated, list): hallucinated = [hallucinated]
                except Exception:
                    # Fallback if unparseable
                    omitted = []
                    hallucinated = []
                    
                total_facts = len(facts)
                num_omitted = len(omitted)
                num_hallucinated = len(hallucinated)
                
                # FOR = omitted / total facts
                for_rate = num_omitted / total_facts if total_facts > 0 else 0.0
                
                result = {
                    "record_id": rec_id,
                    "model": model,
                    "prompt_variant": variant,
                    "omitted_facts": omitted,
                    "hallucinated_facts": hallucinated,
                    "num_facts": total_facts,
                    "for_rate": for_rate,
                    "hr_count": num_hallucinated,
                    "raw_output": output_text
                }
                
                f.write(json.dumps(result) + "\n")
                f.flush()
                added += 1
                
            except Exception as e:
                print(f"Failed on {key}: {e}")
                
    print(f"Finished verifying facts. Added {added} new verifications.")

if __name__ == "__main__":
    verify()
