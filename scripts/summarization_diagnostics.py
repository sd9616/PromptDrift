
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from pathlib import Path

def calculate_summarization_diagnostics(results_path):
    print(f"Loading results from {results_path}...")
    data = []
    with open(results_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    
    # Filter for models
    models = ['Llama', 'Gemini']
    df = df[df['model'].isin(models)]
    
    model_st = SentenceTransformer('all-MiniLM-L6-v2')
    
    results = {}
    
    for model in models:
        m_df = df[df['model'] == model]
        print(f"Calculating for {model}...")
        
        # 1. Self-Consistency Rate (SCR)
        # For each (record_id, variant), calculate pairwise similarity between its 3 runs
        scr_scores = []
        for (record_id, variant), group in m_df.groupby(['record_id', 'prompt_variant']):
            if len(group) < 2: continue
            texts = group['output_text'].fillna("").astype(str).tolist()
            if not any(texts): continue
            
            embeddings = model_st.encode(texts, convert_to_tensor=True)
            cos_sim = util.cos_sim(embeddings, embeddings).cpu().numpy()
            
            # Get upper triangle (excluding diagonal) to get unique pairs
            indices = np.triu_indices(len(texts), k=1)
            pair_sims = cos_sim[indices]
            scr_scores.append(np.mean(pair_sims))
            
        avg_scr = np.mean(scr_scores) if scr_scores else 0
        
        # 2. Cross-Prompt Agreement (CPA)
        # Use the "voted" results (or just trial 1 for simplicity if voted isn't easily accessible, 
        # but combined has all. Let's average all trials per prompt and then compare).
        # Actually, let's use the voted results file for CPA to be consistent with our previous PRA.
        
        results[model] = {
            'SCR': avg_scr
        }
        
    return results

def calculate_cpa_and_pairs(voted_results_path):
    # This matches our previous calculate_cross_prompt_agreement.py logic but 
    # we'll extract "Worst pair" and LFR-equivalent (Semantic Divergence)
    data = []
    with open(voted_results_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    models = ['Llama', 'Gemini']
    df = df[df['model'].isin(models)]
    
    pivot_df = df.pivot(index=['model', 'record_id'], columns='prompt_variant', values='output_text')
    model_st = SentenceTransformer('all-MiniLM-L6-v2')
    
    results = {}
    for model in models:
        m_df = pivot_df.loc[model].dropna()
        pairs = [('S1', 'S2'), ('S1', 'S3'), ('S1', 'S4'), ('S2', 'S3'), ('S2', 'S4'), ('S3', 'S4')]
        divergences = {}
        similarities = []
        
        for p1, p2 in pairs:
            texts1 = m_df[p1].astype(str).tolist()
            texts2 = m_df[p2].astype(str).tolist()
            
            emb1 = model_st.encode(texts1, convert_to_tensor=True)
            emb2 = model_st.encode(texts2, convert_to_tensor=True)
            
            cos_sim = util.cos_sim(emb1, emb2)
            avg_sim = np.diag(cos_sim.cpu().numpy()).mean()
            similarities.append(avg_sim)
            divergences[f"{p1}--{p2}"] = 1.0 - avg_sim
            
        cpa = np.mean(similarities)
        worst_pair = max(divergences, key=divergences.get)
        
        results[model] = {
            'CPA': cpa,
            'Avg_SD': np.mean(list(divergences.values())),
            'Worst_pair': worst_pair,
            'Worst_SD': divergences[worst_pair]
        }
    return results

def calculate_factual_metrics(verified_results_path):
    if not verified_results_path.exists():
        return None
        
    data = []
    with open(verified_results_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
                
    if not data: return None
    
    df = pd.DataFrame(data)
    
    # Aggregate by model
    metrics = {}
    for model, group in df.groupby('model'):
        metrics[model] = {
            'FOR': group['for_rate'].mean(),
            'HR': group['hr_count'].sum(),
            'Evaluated': len(group['record_id'].unique())
        }
        
    return metrics


if __name__ == "__main__":
    combined_file = Path("output/summarization/combined_llama_gemini_s1_s4.jsonl")
    voted_file = Path("output/summarization/voted_llama_gemini_s1_s4.jsonl")
    verified_file = Path("output/summarization/verified_facts.jsonl")
    
    scr_results = calculate_summarization_diagnostics(combined_file)
    cpa_results = calculate_cpa_and_pairs(voted_file)
    factual_results = calculate_factual_metrics(verified_file)
    
    print("\nSummarization Stability & Harm Diagnostics:")
    
    if factual_results:
        print(f"{'Model':<12} | {'CPA':<8} | {'SCR':<8} | {'Avg SD':<8} | {'Worst Pair (SD)':<20} | {'FOR (Harm)':<12} | {'HR (Harm)':<10}")
        print("-" * 105)
    else:
        print(f"{'Model':<12} | {'CPA':<8} | {'SCR':<8} | {'Avg SD':<8} | {'Worst Pair (SD)':<20}")
        print("-" * 70)
    
    for model in ['Llama', 'Gemini']:
        scr = scr_results[model]['SCR']
        cpa = cpa_results[model]['CPA']
        avg_sd = cpa_results[model]['Avg_SD']
        worst_p = cpa_results[model]['Worst_pair']
        worst_sd = cpa_results[model]['Worst_SD']
        
        row_str = f"{model:<12} | {cpa:7.1%} | {scr:7.1%} | {avg_sd:7.1%} | {worst_p} ({worst_sd:.1%})"
        if factual_results and model in factual_results:
            f_rate = factual_results[model]['FOR']
            hr_cnt = factual_results[model]['HR']
            evals = factual_results[model]['Evaluated']
            row_str += f" | {f_rate:10.1%} | {hr_cnt:<10}"
            
        print(row_str)
