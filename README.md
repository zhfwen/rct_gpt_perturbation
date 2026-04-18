# RCT GPT Perturbation

This repository contains experiments for comparing token-level explanation methods on an RCT text classifier (fine-tuned BioLinkBERT), with a focus on perturbation-based evaluation.

The project examines the following explainers:
- SHAP attributions
- Integrated Gradients attributions
- GPT-based perturbation importance estimates with two schemes: (1) GPT-index and (2) GPT-token

## Main data and folders

- dataset.txt: Input dataset used by scripts and notebooks.
- model/: Local classifier and tokenizer artifacts.
- sv_results/: SHAP outputs and intermediate files.
- ig_results/ (generated and/or archived): IG outputs.
- gpt_index_masking/: GPT index-based masking messages and results.
- gpt_index_word_masking/: GPT token/word masking messages and results.
- results/: Main merged outputs, figures, AOPC tables, and per-token perturbation curves.
- results_correct_only/: Analysis outputs restricted to correctly predicted instances.

**All result files and the dataset is available upon reasonable request to lokkerc@mcmaster.ca**

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
- Analyze how GPT defines and applies token importance concepts.
- Review and inspect generated message files and outputs.

### gpt_perturbation_explainer.ipynb
- Main GPT perturbation workflow.
- Runs iterative masking with function-calling loops.
- Produces GPT importance scores for index-based and word-based masking.

### gpt_sign_inversion_analysis.ipynb
- Flip GPT attribution sign conventions.
- Recompute AOPC summaries after sign inversion.

### perturbation_data_cleaning.ipynb
- Data preparation and table construction.
- Recompute logits/probabilities, assign probability groups, sample instances.
- Build combined feature attribution tables from SHAP, IG, and GPT outputs.

### perturbation_analysis.ipynb
- Main analysis notebook over the selected dataset split/sample.
- Dataset summary, feature importance bar charts, explainer correlation/regression views.
- AOPC computation and perturbation-curve visualization.

### perturbation_analysis_correct_only.ipynb
- Repeats major analyses using only correctly classified instances.
- Produces comparable bar charts, correlation plots, and AOPC summaries.

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
