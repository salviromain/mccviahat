# E–N fulltrace analysis

Executed `analysis/notebooks/fulltrace_analysis_EN.ipynb` on 2026-09-29 using
`.mccvenv`. All eight code cells completed successfully with saved outputs.
The original and three-class notebooks are preserved.

Includes 20 emotional and 20 neutral whole traces from `dataNEW`; excludes
27 joy traces. Individual inputs were verified against the E–N subset of
`all_whole.csv`. No per-prompt feature data is used.

Retains the four throttle features, Welch and Mann–Whitney tests, original
standardized effect-size calculation, distance plots, and supplementary prompt
TTR analysis. Bonferroni multiplies feature p-values by 4 separately per test
family; TTR has one pair and therefore multiplier 1.

Exports: `pairwise_tests.csv`, `feature_distances.csv`, `prompt_ttr_tests.csv`.
Mean-rate Mann–Whitney p=0.00771, corrected p=0.0308; corrected Welch p=0.187.
No other feature is significant at 0.05 after correction.

This focused analysis follows inspection of the three-class results and is
exploratory. Trace-level tests assume independent observations; repeated runs
on a node may be dependent. Select `.mccvenv` to rerun the notebook.
