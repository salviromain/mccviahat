# Node and calendar-date analysis: fulltrace E/N

Open the [executed notebook](../../analysis/notebooks/fulltrace_node_day_EN.ipynb).

All 70 available 70B emotional/neutral whole traces are included: 35 per condition,
17 node IDs, seven calendar dates, and 17 existing features. Dates use America/Chicago.
Five traces cross midnight; each belongs to the earlier date, when it started.

The crossed random-intercept model is fitted separately within each condition.
Positive rates, variances, and throttle-per-cycle use log scales; entropy and slope
retain their original scales. Variance fractions therefore refer to the analysis scale.
Maximum-likelihood component tests use 5,999 null simulations when the observed
likelihood-ratio statistic is positive; a zero statistic has p=1 without simulation.
Holm correction covers all 68 feature × condition × effect hypotheses together.
Approximate, pointwise 95% variance-fraction intervals use 999 full-model simulations.

| Condition | Features with node-associated variation | Features with date-associated variation |
|---|---:|---:|
| Emotional | 2/17 | 6/17 |
| Neutral | 1/17 | 8/17 |

Counts use corrected p < 0.05 under the model. These are within-condition associations.
One decision is sensitive to the reported pointwise Monte Carlo precision bounds.
The flagged test is neutral mean throttle rate versus date: Holm p = 0.0935,
with adjusted pointwise Monte Carlo precision bounds of approximately 0.0408–0.1562.

**Interpretation:** measurements can vary across node/date groupings within the same
condition. Only six nodes recur on different dates, always across collection batches;
the overlap graph has three disconnected groups. Separate node/date variance estimates
depend on the random-effects assumptions. Date cannot be cleanly separated from batch
and other collection differences. A nonsignificant test is not evidence of stability.
Conditional residual Q–Q plots show where the Gaussian approximation is questionable.

Files:

- `readable_summary.csv`: percentages, uncertainty intervals, and corrected p-values.
- `variance_components.csv`: numeric variance shares and diagnostic flags.
- `bootstrap_tests.csv`: all tests, Holm correction, and Monte Carlo precision bounds.
- `input_recordings.csv`, `node_date_counts.csv`: provenance, dates, coverage, and inputs.
- `same_node_cross_date_changes.csv`: descriptive same-node changes on the analysis scale.
- `conditional_residuals.csv` and `residual_diagnostics.png`: model-assumption checks.
- `configuration.json`, `input_checksums.csv`, `validation.json`, `execution_status.json`: reproducibility checks.

Raw recordings, processed feature tables, and previous analyses were not modified.
Rerunning the notebook replaces only the exports in this results directory.
