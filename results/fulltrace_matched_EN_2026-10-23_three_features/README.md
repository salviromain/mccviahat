# Matched fulltrace E/N analysis

[Open the executed notebook](../../analysis/notebooks/fulltrace_matched_EN.ipynb).

Compare emotional and neutral recordings within the same node and calendar date.
The data contain 70 traces, 23 matched groups, 17 nodes, and seven dates across
batches 001, 003, and 004. Each group contains either one or two recordings per condition.
Dates use America/Chicago; midnight-crossing traces belong to their start date.

## Method

Average recording-level feature values within each condition/node/date group.
For mean throttle rate and PTE/cycles, model log(E mean / N mean) and convert to percentages.
Spectral entropy uses E mean minus N mean in original units. PTE/cycles is calculated
per trace before averaging. One matched group contributes one contrast.

A crossed Gaussian mixed-effects model estimates the overall contrast and node/date
heterogeneity of that contrast. It assumes common remaining variance across groups;
uncertainty from one-versus-two repetitions is not estimated separately per group.
Zero-overall-effect tests use 19,999 parametric-bootstrap likelihood-ratio simulations.
Holm correction covers three feature tests. Pointwise 95% bootstrap-t intervals use
1,999 simulations. Approximate prediction intervals target a matched comparison on a
new node and new date under a comparable design; they are not validated held-out results.

## Results

| Feature | Overall E−N | Units | Pointwise 95% CI | Holm p | E > N groups |
|---|---:|---|---|---:|---:|
| Mean throttle rate | +2.07 | % vs neutral | -0.721 to +4.89 | 0.1311 | 20/23 |
| Spectral entropy | +0.0146 | entropy units | +0.00249 to +0.0273 | 0.0866 | 21/23 |
| PTE / cycles | +3.62 | % vs neutral | +1.01 to +6.2 | 0.0382 | 19/23 |

**1/3 features pass the Holm-corrected 0.05 threshold.**
0 decisions have Monte Carlo precision bounds crossing the corrected threshold.
Positive pointwise intervals can coexist with a nonsignificant Holm test: those intervals
are not corrected for the three comparisons, and use a different inferential procedure.

All three approximate new-group prediction intervals include zero. Group-level reversals
are therefore compatible with the model even when the average estimate keeps its sign.
See the matched plots and leave-one-node/date-out results for direction and magnitude;
unchanged omission signs do not establish significance or future generalization.

## Interpretation and limits

These three features were selected after the earlier five-feature results were viewed.
The reduced-family correction is exploratory; it does not retroactively replace the
original five-test inference. Both output sets are preserved.

This analysis addresses condition differences within matched groups. The earlier node/day
notebook addressed variation of raw measurements within each condition; those are different questions.
It also uses more batches and a different sampling unit than the batch_001-only pooled notebook,
so changed p-values cannot be attributed to matching alone.

Only six nodes recur across dates. Date and collection batch overlap, and the grouping
graph is sparse/disconnected. Gaussian assumptions, outliers, and uncertain variance
components limit inference. Matching removes shared baselines but cannot remove all
condition-specific ordering or setup differences. A nonsignificant result does not prove equality.
A stability claim needs a practical tolerance and confirmation on new recordings.

## Outputs

- `matched_groups.csv`, `matched_contrasts.csv`: group provenance, E/N means, and differences.
- `overall_effects.csv`, `readable_summary.csv`: estimates, intervals, tests, and precision flags.
- `leave_one_group_out.csv`, `sensitivity_summary.csv`: all 72 omission refits.
- `descriptive_group_summaries.csv`: summaries by date and node, not separate significance tests.
- `model_residuals.csv`, `model_diagnostics.png`: model-assumption checks.
- `input_checksums.csv`, `configuration.json`, `execution_status.json`, `validation.json`: audit and reproducibility.
- Eight PNG figures are embedded in the notebook and saved here.

Raw data, feature CSVs, other notebooks, and previous results were preserved.
Rerunning replaces only this analysis directory and its notebook outputs.
