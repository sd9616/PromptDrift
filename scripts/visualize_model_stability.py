
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

# Use a premium style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid", palette="muted")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def calculate_scr_per_variant(combined_results_path, cache_path=None):
    if cache_path and Path(cache_path).exists():
        print(f"Loading SCR data from cache: {cache_path}")
        return pd.read_csv(cache_path)
        
    print("Calculating SCR per variant...")
    data = []
    with open(combined_results_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    models = ['Llama', 'Gemini']
    df = df[df['model'].isin(models)]
    model_st = SentenceTransformer('all-MiniLM-L6-v2')
    
    scr_data = []
    
    for (model, variant), group in df.groupby(['model', 'prompt_variant']):
        print(f"  Processing {model} {variant}...")
        # For each record in this (model, variant), calculate pairwise similarity
        record_scores = []
        for record_id, r_group in group.groupby('record_id'):
            if len(r_group) < 2: continue
            texts = r_group['output_text'].fillna("").astype(str).tolist()
            if not any(texts): continue
            
            embeddings = model_st.encode(texts, convert_to_tensor=True)
            cos_sim = util.cos_sim(embeddings, embeddings).cpu().numpy()
            
            indices = np.triu_indices(len(texts), k=1)
            pair_sims = cos_sim[indices]
            record_scores.append(np.mean(pair_sims))
            
        if record_scores:
            scr_data.append({
                'Model': model,
                'Prompt Variant': variant,
                'SCR': np.mean(record_scores)
            })
            
    df_result = pd.DataFrame(scr_data)
    if cache_path:
        df_result.to_csv(cache_path, index=False)
        print(f"Saved SCR data to {cache_path}")
    return df_result

def calculate_cpa_per_model(voted_results_path, cache_path=None):
    if cache_path and Path(cache_path).exists():
        print(f"Loading CPA data from cache: {cache_path}")
        return pd.read_csv(cache_path)
        
    print("Calculating CPA per model...")
    data = []
    with open(voted_results_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    models = ['Llama', 'Gemini']
    df = df[df['model'].isin(models)]
    model_st = SentenceTransformer('all-MiniLM-L6-v2')
    
    pivot_df = df.pivot(index=['model', 'record_id'], columns='prompt_variant', values='output_text')
    
    cpa_data = []
    for model in models:
        print(f"  Processing CPA for {model}...")
        m_df = pivot_df.loc[model].dropna()
        variants = sorted(m_df.columns)
        
        # Calculate similarity of each variant to all others
        for v in variants:
            other_variants = [ov for ov in variants if ov != v]
            texts_v = m_df[v].astype(str).tolist()
            emb_v = model_st.encode(texts_v, convert_to_tensor=True)
            
            variant_sims = []
            for ov in other_variants:
                texts_ov = m_df[ov].astype(str).tolist()
                emb_ov = model_st.encode(texts_ov, convert_to_tensor=True)
                cos_sim = util.cos_sim(emb_v, emb_ov)
                avg_sim = np.diag(cos_sim.cpu().numpy()).mean()
                variant_sims.append(avg_sim)
                
            cpa_data.append({
                'Model': model,
                'Prompt Variant': v,
                'Stability': np.mean(variant_sims)
            })
            
    df_result = pd.DataFrame(cpa_data)
    if cache_path:
        df_result.to_csv(cache_path, index=False)
        print(f"Saved CPA data to {cache_path}")
    return df_result

def plot_grouped_stability(df, metric_col, title, ylabel, output_path):
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x='Prompt Variant', y=metric_col, hue='Model', palette=['#3498db', '#e74c3c'])
    
    plt.title(title, fontsize=8, pad=15)
    plt.xlabel('Prompt Variant', fontsize=8)
    plt.ylabel(ylabel, fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.ylim(0.7, 1.0) # Zoom in to see differences
    
    # Annotate bars
    for p in plt.gca().patches:
        plt.gca().annotate(f'{p.get_height():.1%}', 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'center', 
                       xytext = (0, 9), 
                       textcoords = 'offset points',
                       fontsize=8)
    
    plt.legend(title='Model', fontsize=8, title_fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    combined_file = PROJECT_ROOT / "output" / "summarization" / "combined_llama_gemini_s1_s4.jsonl"
    voted_file = PROJECT_ROOT / "output" / "summarization" / "voted_llama_gemini_s1_s4.jsonl"
    out_dir = PROJECT_ROOT / "output" / "summarization" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    scr_cache = out_dir / "scr_data_cache.csv"
    cpa_cache = out_dir / "cpa_data_cache.csv"
    
    scr_df = calculate_scr_per_variant(combined_file, cache_path=scr_cache)
    plot_grouped_stability(scr_df, 'SCR', 
                          "Semantic Similarity Comparison: Llama vs Gemini", 
                          "SemSim (Semantic Similarity)", 
                          out_dir / "stability_comparison_scr.pdf")
    
    cpa_df = calculate_cpa_per_model(voted_file, cache_path=cpa_cache)
    plot_grouped_stability(cpa_df, 'Stability', 
                          "Semantic Agreement Comparison: Llama vs Gemini", 
                          "Semantic Agreement (CPA)", 
                          out_dir / "semantic_agreement_comparison.pdf")
    
    print(f"Comparison plots saved to {out_dir}")
