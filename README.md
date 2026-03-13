# PromptDrift

## Patch Suggestions — Prompt Drift Notebook

This Colab notebook demonstrates how we tested **prompt drift** in automated patch generation using AI tools. The workflow combines Defects4J, Groq, and Gemini to evaluate how variations in prompts affect patch quality and correctness.

### Overview

The notebook automates generating, applying, and validating patches suggested by AI models. We focus on understanding how small changes in prompts influence patch outputs and their success on real codebases.

### Workflow

1. **Environment Setup**
   - Installs dependencies: `Defects4J` (Java bug datasets), `Groq`, and `Gemini`.
   - Mounts Google Drive to access the prompt dataset.

2. **Prompt Dataset**
   - Loads prompts from Google Drive containing anchor prompts and variants for each bug scenario.

3. **Patch Generation**
   - Sends each prompt to the AI model.
   - Uses **regex extraction** to parse AI outputs and identify modified code snippets.

4. **Patch Application**
   - Replaces the corresponding file in the local project with the AI-suggested patch.

5. **Validation**
   - Runs `compile` and `test` commands for the patched project.
   - Records which prompts lead to successful fixes versus compile/test failures.

### Usage

1. Open the notebook in Colab.
2. Mount your Google Drive and ensure the prompt dataset is accessible.
3. Run the cells sequentially to install dependencies, load prompts, generate patches, and run validations.
4. Review results in the output tables or logs to study prompt drift.

### Notes

- Compile or test failures are expected for some AI-generated patches; the notebook captures these for analysis.
- Regex extraction assumes consistent patch formatting; small modifications may require adapting the pattern.
- Supports batch evaluation of multiple prompt variants for systematic drift analysis.
