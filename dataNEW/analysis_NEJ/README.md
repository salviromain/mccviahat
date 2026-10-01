# Three-batch fulltrace analysis

`analysis/notebooks/fulltrace_analysis_NEJ.ipynb` pools individual whole-trace
CSVs recursively from these `dataNEW` folders:

| Folder | Neutral | Emotional | Joy |
| --- | ---: | ---: | ---: |
| `fulltrace_0ld_batch` (or `fulltrace_old_batch`) | 20 | 20 | 0 |
| `fulltrace_1st_batch` | 0 | 0 | 27 |
| `fulltrace_2nd_batch` (including nested `2nd_batch`) | 10 | 10 | 3 |
| Total | 30 | 30 | 30 |

The 90 traces retain batch and source-file provenance. Run labels are read as
strings so names such as `222E1` remain intact. Duplicate run labels are rejected.
Combined CSVs are checked against individual files in their directory subtree
and are never counted as additional observations. Root-level CSVs outside the
three batch folders are not analysis inputs.

The original four throttle features (mean rate, variance, spectral entropy,
and slope), pairwise tests, distance calculations, and prompt TTR analysis are
preserved. Use `.mccvenv` and run all notebook cells to refresh the outputs:

- `pooled_whole_traces.csv`: combined observations with provenance.
- `batch_condition_counts.csv`: batch-by-condition sample counts.
- `pairwise_tests.csv`: Welch and Mann–Whitney comparisons, Bonferroni-corrected
  over 12 feature/pair tests separately for each test family.
- `feature_distances.csv`: within-/between-condition feature distances.
- `prompt_ttr_tests.csv`: supplementary comparisons of the three prompt sets.

The analyses pool batches and treat traces as independent. Batch composition
is uneven and repeated runs on a node may be dependent, so pooled differences
are exploratory and do not isolate causal effects of condition. TTR does not
control workload confounding.
