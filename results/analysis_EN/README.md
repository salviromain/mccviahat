# E–N fulltrace analysis

Executed `analysis/notebooks/fulltrace_analysis_EN.ipynb` with `.mccvenv` on
2026-10-01. All code cells completed with saved outputs and two plots.

Loads individual whole-trace CSVs recursively from the three batch folders in
`dataNEW`, preserving batch/source provenance and validating local aggregates.
Includes 30 emotional and 30 neutral traces; excludes 30 joy traces. Run labels
are read as strings. Duplicate labels and nonfinite selected features are rejected.

The original four throttle features are supplemented with `throttle_per_cycle`:
`core_power.throttle__mean_rate / cycles__mean_rate`. Because the rates share a
trace duration, this equals the ratio of summed throttle counts to summed cycles,
assuming matching valid counter intervals. Feature CSVs do not verify interval
coverage. Cycle rates must be positive; both rates must be finite and throttle
rates nonnegative. The feature is a dimensionless ratio, not a verified physical
percentage of throttled cycles.

All five features appear in the Welch and Mann–Whitney tests, standardized
effect sizes, distribution plots, and distance analysis. Bonferroni multiplies
feature p-values by 5 separately per test family. Supplementary prompt TTR
still has one condition pair and multiplier 1.

The new ratio averages 0.009096 for neutral and 0.009376 for emotional.
Corrected p-values are 0.3524 (Welch) and 0.3741 (Mann–Whitney), so this
comparison is not significant at 0.05.

Exports: `pooled_whole_traces.csv`, `batch_condition_counts.csv`,
`pairwise_tests.csv`, `feature_distances.csv`, and `prompt_ttr_tests.csv`.

This analysis is exploratory. Pooled tests do not adjust for batch or
repeated-node dependence. The ratio does not eliminate all workload confounding.
