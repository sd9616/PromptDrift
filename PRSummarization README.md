# Pull Request Summarization Evaluation Workflow

This guide outlines how to perform a full stability and harm evaluation for PR summarization using **Llama** and **Gemini**. The workflow moves away from traditional ROUGE metrics to focus on **Semantic Agreement** and **Factual Correctness**.

## Overview of Metrics
- **Semantic Agreement (CPA)**: Measures how similar summaries are across different prompt layouts.
- **Semantic Divergence (SD)**: Measures the variance/instability (1 - CPA).
- **Factual Omission Rate (FOR)**: The percentage of ground-truth technical facts missing from a summary.
- **Hallucination Rate (HR)**: The count of fabricated claims not mentioned in the PR description or commits.

---

## 1. Run Experiments
Ensure you have generated the summaries for your models (Llama, Gemini) across all four prompt variants (S1, S2, S3, S4).
The evaluation scripts expect the results in:
`output/summarization/combined_llama_gemini_s1_s4.jsonl`

---

## 2. Generate Ground-Truth Facts
Use Llama to extract objective, atomic facts from the PR Body and Commit Messages. This serves as the "gold standard" for harm evaluation.

```bash
python3 scripts/extract_facts.py
```
- **Input**: `data/summarization/records.json`
- **Output**: `output/summarization/extracted_facts.jsonl`
- *Note: This script uses resume capability and can skip already processed records.*

---

## 3. Verify Summaries Against Facts
Evaluate each generated summary against the ground-truth facts extracted in the previous step.

```bash
python3 scripts/verify_facts.py
```
- **Input**: `output/summarization/extracted_facts.jsonl` & `output/summarization/combined_llama_gemini_s1_s4.jsonl`
- **Output**: `output/summarization/verified_facts.jsonl`

---

## 4. Generate Results & LaTeX Table
Once verification is complete, generate the diagnostic metrics and the formatted LaTeX table for your paper.

```bash
python3 scripts/generate_latex_table.py
```
This script will output a table containing:
- Semantic Agreement (%)
- Semantic Divergence
- Factual Omission Rate (FOR %)
- Average Missing Facts
- Average Hallucinated Facts

---

## 5. Generate Visualizations
Generate publication-quality PDF plots for Self-Consistency (SCR) and Semantic Agreement (CPA).

```bash
python3 scripts/visualize_model_stability.py
```
- **Output**: 
  - `output/summarization/plots/stability_comparison_scr.pdf`
  - `output/summarization/plots/semantic_agreement_comparison.pdf`

---

## Technical Details
- **LLM Judge**: All factual extraction and verification are performed using **Llama 3.3 70B** for speed and cost-effectiveness.
- **Embeddings**: Semantic similarity is calculated using `SentenceTransformer('all-MiniLM-L6-v2')`.
- **Aesthetics**: All plots are generated with a uniform font size (size 8) and high-resolution PDF format.
