# Three-condition fulltrace analysis

Executed `analysis/notebooks/fulltrace_analysis_NEJ.ipynb` using the project's
`.mccvenv` on 2026-09-29. The original notebook was preserved.

Inputs: all 67 individual whole-trace CSVs in `dataNEW` (20 neutral, 20 emotional,
27 joy), validated against `all_whole.csv`. No per-prompt feature data is used.

Changes from the original notebook:
- Load `dataNEW`; include N, E, and J and all three condition pairs.
- Validate unique run labels, whole-trace mode, all three conditions, and finite
  values for the four selected throttle features.
- Retain the original four features, test methods, effect-size calculation,
  distance calculations, and prompt TTR analysis.
- Apply Bonferroni correction across 12 feature/pair tests separately for Welch
  and Mann–Whitney; across three pairs per test family for prompt TTR.
- Use three class colors and compact distance-plot labels.
- Save outputs in the notebook and export `pairwise_tests.csv`,
  `feature_distances.csv`, and `prompt_ttr_tests.csv` here.

Execution dependencies `nbclient` and `nbformat` were installed in `.mccvenv`.
Select that Python environment when rerunning all cells in the notebook.

At alpha 0.05, corrected Mann–Whitney tests identify five comparisons: joy has
higher mean rate and lower spectral entropy than both neutral and emotional,
and higher variance than neutral. No neutral–emotional feature comparison is
significant after correction. Corrected Welch tests identify the four mean-rate
and spectral-entropy joy comparisons, but not the variance comparison.

Joy and the other conditions belong to different collection batches, and
repeated runs on a node may be dependent. These exploratory trace-level tests
do not isolate a causal effect of condition. Prompt TTR is supplementary and
does not control workload confounding.
