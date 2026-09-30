# Recomputed 70B isolated-trial features

Source: all 16 condition folders under `runs/70b` (eight paired runs).
Destination: `dataNEW/70b`. Original `data/` feature files are preserved.

## Reproduce

From the repository root:

```bash
.mccvenv/bin/python -u scripts/run/recompute_70b_features.py > dataNEW/70b/extraction.log 2>&1
```

Then run all cells in `analysis/notebooks/70b_blocks_analysis_dataNEW.ipynb`
using `.mccvenv`. This is a copy of `70b_blocks_analysis.ipynb` configured for
these recomputed features, with saved outputs and exported result tables.

## Extraction

- Each row represents an isolated trial, not a continuous fulltrace experiment.
- `emotional` and `neutral` form `run_unsuffixed`; folders with matching numeric
  suffixes form `run1` through `run7`.
- One CSV per paired run plus `all_trials.csv`; rows retain `source_trial` and
  `run` for provenance.
- Reuses `extract_features.py` for perf and interrupt features, including its
  existing duration normalization. LZ complexity is omitted for all signals
  at the user's request; all other metric formulas are preserved.
- Added optional `include_lz=False` to the extractor API; its default still
  computes LZ. Verified skipping LZ does not call it and preserves other metrics.
- `input_manifest.json` records every trial and SHA-256 hashes of input files.
- `extraction.log` records the completed batch. Pandas fragmentation warnings
  concern performance when loading wide interrupt tables, not failed trials.

## Analysis

Four throttle features: slope, variance, spectral entropy, and mean rate.
Preserves k-means settings, 75% per-run direction threshold, Mann–Whitney tests,
random-label checks, and balanced prompt-subset robustness sweeps from the
original notebook. LZ is excluded from all analysis sections.

Result tables are saved in `analysis/`. Bonferroni covers only features passing
the original direction filter. This data-dependent selection and within-run
trial dependence are not accounted for by that correction. Treat p-values as
exploratory. Clustering accuracy is in-sample, not held-out predictive accuracy.

## Completed results (2026-09-29)

All 320 trials extracted successfully: 160 E, 160 N, eight runs. Verified every
source trial appears once, all run CSVs agree with the combined file, all four
selected features are finite, and no LZ columns exist.

Best feature: mean rate, in-sample k-means accuracy 54.6875%, ARI 0.00594, higher
in E in 5/8 runs. Other accuracies: variance 52.8125%, spectral entropy 51.25%,
slope 50.625%. No feature reaches the required 6/8 direction majority. Thus the
original pipeline performs no Mann–Whitney tests; `mwu_tests.csv` has headers
but no rows. This is a filter outcome, not a set of nonsignificant test results.

Random-label mean accuracies: 51.875% within E and 52.125% within N.
Best-feature prompt-subset robustness mean accuracy ranges from
54.12% to 54.85%.
