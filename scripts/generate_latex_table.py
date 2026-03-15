import json
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def generate_table():
    verified_file = PROJECT_ROOT / "output" / "summarization" / "verified_facts.jsonl"
    cpa_cache = PROJECT_ROOT / "output" / "summarization" / "plots" / "cpa_data_cache.csv"
    
    # Load Factual Metrics
    factual_data = []
    if verified_file.exists():
        with open(verified_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    factual_data.append(json.loads(line))
    
    fact_df = pd.DataFrame(factual_data)
    
    # Load CPA (Semantic Agreement) Metrics
    cpa_df = pd.DataFrame()
    if cpa_cache.exists():
        cpa_df = pd.read_csv(cpa_cache)
    
    print("Metrics calculated on the subset processed so far (Verification is still running in background):")
    print(f"Total evaluated summaries so far: {len(fact_df)}")
    print("-" * 50)
    
    # Process Metrics
    metrics = []
    models = ['Llama', 'Gemini']
    variants = ['S1', 'S2', 'S3', 'S4']
    variant_names = {
        'S1': 'S1 - Baseline',
        'S2': 'S2 - Role-based',
        'S3': 'S3 - Paraphrased',
        'S4': 'S4 - Bullet Format'
    }
    
    for model in models:
        for var in variants:
            # Semantic Divergence = 1 - CPA (Stability)
            sd = np.nan
            cpa_val = np.nan
            if not cpa_df.empty:
                cpa_row = cpa_df[(cpa_df['Model'] == model) & (cpa_df['Prompt Variant'] == var)]
                if not cpa_row.empty:
                    cpa_val = cpa_row['Stability'].values[0]
                    sd = 1.0 - cpa_val
            
            # Factual Metrics
            for_rate = np.nan
            avg_omitted = np.nan
            avg_hr = np.nan
            
            if not fact_df.empty:
                f_row = fact_df[(fact_df['model'] == model) & (fact_df['prompt_variant'] == var)]
                if not f_row.empty:
                    for_rate = f_row['for_rate'].mean()
                    # avg missing and hallucinated per summary
                    avg_omitted = f_row.apply(lambda row: len(row['omitted_facts']), axis=1).mean()
                    avg_hr = f_row['hr_count'].mean()
            
            metrics.append({
                'ModelStr': f"{model} ({variant_names[var]})",
                'CPA': cpa_val,
                'SD': sd,
                'FOR': for_rate,
                'Omitted': avg_omitted,
                'Hallucinated': avg_hr
            })
            
    # Print LaTeX Table
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{Pull Request Summarization stability and harm metrics. \textit{Semantic Agreement} (Cross-Prompt Agreement) measures the similarity of outputs across prompt templates. \textit{Semantic Divergence} measures the variance (1 - CPA). \textit{FOR} is the Factual Omission Rate, and \textit{Missing/Hallucinated} represent the average count of omitted and fabricated code-change facts per summary.}")
    print(r"\label{tab:summarization_metrics}")
    print(r"\begin{tabular}{lccccc}")
    print(r"\toprule")
    print(r"\textbf{Model} & \textbf{Semantic Agreement (\%)} & \textbf{Semantic Divergence} & \textbf{FOR (\%)} & \textbf{Missing Facts} & \textbf{Hallucinated Facts} \\")
    print(r"\midrule")
    
    for i, m in enumerate(metrics):
        if i == 4:
            print(r"\midrule")
            
        cpa_str = f"{m['CPA']*100:.1f}\%" if not np.isnan(m['CPA']) else "N/A"
        sd_str = f"{m['SD']:.3f}" if not np.isnan(m['SD']) else "N/A"
        for_str = f"{m['FOR']*100:.1f}\%" if not np.isnan(m['FOR']) else "N/A"
        omit_str = f"{m['Omitted']:.2f}" if not np.isnan(m['Omitted']) else "N/A"
        hr_str = f"{m['Hallucinated']:.2f}" if not np.isnan(m['Hallucinated']) else "N/A"
        
        print(f"{m['ModelStr']} & {cpa_str} & {sd_str} & {for_str} & {omit_str} & {hr_str} \\\\")
        
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

if __name__ == "__main__":
    generate_table()
