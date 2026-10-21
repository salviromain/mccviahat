# All-indicator analysis

Both analyses now discover indicators from the existing recomputed feature
columns instead of selecting only `core_power.throttle`. The same four metrics
are used for each indicator: **mean rate, variance, spectral entropy, and slope**.
LZ complexity remains excluded. No feature extraction formulas or source data
were changed.

The sample selection is unchanged: 320 recomputed isolated trials for 70B blocks
and 70 emotional/neutral whole traces across all eligible fulltrace batches.
Older legacy trials, duplicate extraction versions, joy traces, and mechanical
traces are not added.

## 70B isolated trials

- 320 trials: 160 emotional and 160 neutral, from eight paired runs.
- 44 indicators were found in the existing feature tables.
- 28 indicators supply **112 usable features**.
- 56 features from 14 indicators are constant; eight features from two indicators
  have unavailable/nonfinite measurements. These 64 features are explicitly
  reported and excluded. No missing values are imputed.

The included indicators cover perf counters, instructions, cycles, branches,
caches, TLB activity, interrupts, power, thermal margin, and CPU frequency.
[feature_availability.csv](70b_blocks/feature_availability.csv) lists every
indicator/metric combination and its inclusion status.

All 112 usable features enter the original single-feature clustering and
within-run direction screen. **27 pass the 75% direction threshold.** Mann–Whitney
p-values are corrected across the complete 112-feature search, including features
that do not pass the screen. Correction is not restarted for each indicator.
This is deliberately broader than the previous correction over passing features
only.

**No per-prompt feature passes the corrected 0.05 threshold.** The smallest
corrected p-value is approximately **0.0869**. The highest in-sample clustering
accuracy is **58.75%**, for `branch-misses__mean_rate`, which is higher in neutral
trials in 8/8 runs. Accuracy after selecting a feature from these same observations
is not held-out predictive performance.

Random-label checks use the best clustering feature of each usable indicator.
Prompt-subset robustness sweeps cover every usable feature, using the original
subset sizes, seeds, 20 repetitions, and K-means configuration. Figures show the
global best feature and paginated best-feature curves for every usable indicator.

- [Per-indicator summary](70b_blocks/indicator_summary.csv)
- [All clustering results](70b_blocks/single_feature_clustering.csv)
- [Corrected Mann–Whitney tests](70b_blocks/mwu_tests.csv)
- [Random-label checks](70b_blocks/random_label_null.csv)
- [All-feature robustness](70b_blocks/all_feature_robustness.csv)
- [Best feature per indicator: robustness](70b_blocks/best_per_indicator_robustness.csv)
- [Input trial features](70b_blocks/input_trials.csv)
- [Executed notebook](70b_blocks/70b_blocks_analysis.ipynb)
- [Figures](70b_blocks/figures/)

## Fulltrace E/N

- 70 traces: 35 emotional and 35 neutral; 17 node IDs, with repeated recordings.
- All four indicators present in the existing whole-trace feature tables are
  included: `tlb:tlb_flush`, `instructions`, `cycles`, and `core_power.throttle`.
- Four metrics per indicator plus the existing throttle-per-cycle ratio give
  **17 features**. No new indicators are reconstructed from raw collection logs.
- Correction covers all 17 comparisons together, separately within the Welch
  and Mann–Whitney test families.

Three features pass corrected Mann–Whitney tests; instruction slope also passes
Welch's mean-comparison test:

| Feature | Corrected Welch p | Corrected Mann–Whitney p |
|---|---:|---:|
| Instruction slope | 5.54e-20 | 1.57e-11 |
| Instruction variance | 0.7191 | 0.00421 |
| Cycle variance | 1.0000 | 0.00605 |

These features have higher numeric values in neutral traces. For instruction
slope, both means are negative: approximately -1,516 in neutral versus -2,009 in
emotional traces. The variance findings concern distribution/rank differences;
the corresponding Welch tests do not pass correction.

Throttle features, the throttle-per-cycle ratio, and TLB features do not pass the
expanded correction. The original throttle effect estimates and uncorrected
p-values are unchanged; their corrected p-values now use 17 comparisons rather
than five.

- [All corrected tests](fulltrace_EN/pairwise_tests.csv)
- [Feature availability](fulltrace_EN/feature_availability.csv)
- [Analyzed recordings](fulltrace_EN/pooled_whole_traces.csv)
- [Batch counts](fulltrace_EN/batch_condition_counts.csv)
- [Feature distances](fulltrace_EN/feature_distances.csv)
- [Prompt TTR comparison](fulltrace_EN/prompt_ttr_tests.csv)
- [Executed notebook](fulltrace_EN/fulltrace_analysis_EN.ipynb)
- [Distribution plots](fulltrace_EN/figures/cell_06_00.png)
- [Distance plots](fulltrace_EN/figures/cell_09_00.png)

## Interpretation and preservation

These are exploratory, pooled analyses. Trials within runs and full traces from
the same node may be dependent; batch and workload differences remain possible
explanations. Differences in instructions or cycles do not by themselves
establish an emotion-specific substrate effect.

The two main notebooks were updated. Their immediately preceding versions are
in [prior_notebooks/](prior_notebooks/). The earlier throttle-only run remains
unchanged in [all_trials_2026-10-20](../all_trials_2026-10-20/README.md).

[validation.json](validation.json) records input/result preservation, feature
coverage, corrections, successful notebook execution, unchanged original throttle
calculations, and robustness coverage. Existing files are recorded in
[preserved_file_checksums.json](preserved_file_checksums.json).
