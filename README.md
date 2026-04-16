# RCT GPT Perturbation

This repository contains experiments for comparing token-level explanation methods on an RCT text classifier, with a focus on perturbation-based evaluation.

The project combines:
- SHAP attributions
- Integrated Gradients attributions
- GPT-based perturbation importance estimates
- AOPC-style perturbation evaluation

## Project goals

- Measure which tokens drive predictions for a binary RCT rigor classifier.
- Compare attribution methods under consistent perturbation tests.
- Summarize agreement and differences across explainers.

## Main data and folders

- dataset.txt: Input dataset used by scripts and notebooks.
- model/: Local classifier and tokenizer artifacts.
- sv_results/: SHAP outputs and intermediate files.
- ig_results/ (generated and/or archived): IG outputs.
- gpt_index_masking/: GPT index-based masking messages and results.
- gpt_index_word_masking/: GPT token/word masking messages and results.
- results/: Main merged outputs, figures, AOPC tables, and per-token perturbation curves.
- results_correct_only/: Analysis outputs restricted to correctly predicted instances.
- archive/: Older experiments, scripts, and poster-related notebooks.

## Core scripts

- shapley.py: Runs SHAP over a row range and exports per-token attributions.
- ig.py: Runs Integrated Gradients over a row range and exports per-token attributions.
- workers.py: Utility functions for merging explainers, GPT integration, masking, and AOPC computations.
- calculate_aopc_separate.py: Builds per-instance perturbation curves for positive and negative token sets.

## Cluster scripts

- shapley.sh: SLURM batch array for SHAP jobs.
- ig.sh: SLURM batch array for IG jobs.
- calculate_aopc_separate.sh: SLURM batch array for perturbation-curve generation.

## Notebook guide

### gpt_importance.ipynb
Purpose:
- Analyze how GPT defines and applies token importance concepts.
- Review and inspect generated message files and outputs.

Typical use:
- Inspect prompt/response behavior and definition consistency before running larger GPT perturbation sweeps.

### gpt_perturbation_explainer.ipynb
Purpose:
- Main GPT perturbation workflow.
- Runs iterative masking with function-calling loops.
- Produces GPT importance scores for index-based and word-based masking.

Typical outputs:
- JSON results in gpt_index_masking/results and gpt_index_word_masking/results.
- Serialized message histories in the corresponding messages folders.

### gpt_sign_inversion_analysis.ipynb
Purpose:
- Flip GPT attribution sign conventions.
- Recompute perturbation curves and AOPC summaries after sign inversion.

Typical outputs:
- Flipped-sign AOPC files and summary tables under results.

### perturbation_data_cleaning.ipynb
Purpose:
- Data preparation and table construction.
- Recompute logits/probabilities, assign probability groups, sample instances.
- Build combined feature attribution tables from SHAP, IG, and GPT outputs.

Typical outputs:
- feature_attributions.csv and helper merged files used by downstream analysis notebooks.

### perturbation_analysis.ipynb
Purpose:
- Main analysis notebook over the selected dataset split/sample.
- Dataset summary, feature importance bar charts, explainer correlation/regression views.
- AOPC computation and perturbation-curve visualization.

Typical outputs:
- Figures, AOPC summaries, and per-token perturbation curve files in results.

### perturbation_analysis_correct_only.ipynb
Purpose:
- Repeats major analyses using only correctly classified instances.
- Produces comparable bar charts, correlation plots, and AOPC summaries.

Typical outputs:
- Filtered tables and figures under results_correct_only.

## Typical end-to-end workflow

1. Generate SHAP attributions with shapley.py.
2. Generate IG attributions with ig.py.
3. Build/clean merged attribution tables (perturbation_data_cleaning.ipynb).
4. Generate GPT perturbation scores (gpt_perturbation_explainer.ipynb).
5. Run perturbation and AOPC analyses (perturbation_analysis.ipynb and/or perturbation_analysis_correct_only.ipynb).
6. Optionally run sign-inversion sensitivity analysis (gpt_sign_inversion_analysis.ipynb).

## Quick command examples

```bash
python shapley.py 0 2000 0
python ig.py 0 2000 30
python calculate_aopc_separate.py 0
```

## Notes

- Most scripts and notebooks assume GPU availability for practical runtime.
- The model is loaded from the local model directory.
- File paths are generally written relative to the repository root.
