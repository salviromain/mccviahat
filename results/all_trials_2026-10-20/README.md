# Analysis rerun — October 20, 2026

Both notebooks were executed from fresh kernels. Raw data and existing feature
files were not regenerated or edited. Previous results and the pre-run notebook
files were preserved.

## Scope

| Analysis | Inputs | Analyzed observations |
|---|---|---|
| [70B blocks](../../analysis/notebooks/70b_blocks_analysis.ipynb) | All eight recomputed isolated-trial runs | 320 trials: 160 emotional, 160 neutral |
| [Fulltrace E/N](../../analysis/notebooks/fulltrace_analysis_EN.ipynb) | Every canonical recomputed fulltrace batch | 70 traces: 35 emotional, 35 neutral |

The per-prompt scope follows your explicit choice to use **only the 320
recomputed trials with the current four-feature definition**. The additional 320
older block trials are excluded. The 280 legacy `independent*.csv` rows overlap
the selected recordings and are not counted again. Per-prompt windows from
continuous fulltrace recordings are not mixed with isolated trials.

Legacy per-prompt mean rates use `mean(samples) / duration`; the selected
recomputed mean rates use `sum(finite samples) / duration`. These definitions
were not mixed. LZ complexity is omitted from the four-feature analysis.

## 70B per-prompt results

The original clustering, direction filter, random-label checks, and robustness
sweeps were rerun with their existing settings and seeds.

| Throttle feature | In-sample clustering accuracy | Direction | Runs agreeing |
|---|---:|---|---:|
| Mean rate | 54.69% | Emotional higher | 5/8 |
| Variance | 52.81% | Neutral higher | 4/8 |
| Spectral entropy | 51.25% | Neutral higher | 5/8 |
| Slope | 50.63% | Emotional higher | 5/8 |

**No feature passes the original 75% direction filter**, which requires at least
6 of 8 runs. Consequently, the pipeline performs no Mann–Whitney tests; the
exported test table contains headers but no rows. This is a filtering outcome,
not a collection of nonsignificant test results.

Mean rate is the best clustering feature. Its prompt-subset robustness mean
accuracy ranges from 54.12% to 54.85%. Random-label mean accuracies are 51.88%
within emotional trials and 52.13% within neutral trials. Clustering accuracy
describes in-sample association, not held-out prediction.

These five exported result tables reproduce the previous recomputed-trial
analysis within numerical tolerance.

- [Analyzed trial inputs](70b_blocks/input_trials.csv)
- [Clustering table](70b_blocks/single_feature_clustering.csv)
- [Mann–Whitney table](70b_blocks/mwu_tests.csv)
- [Random-label checks](70b_blocks/random_label_null.csv)
- [Best-feature robustness](70b_blocks/best_feature_robustness.csv)
- [Other-feature robustness](70b_blocks/other_feature_robustness.csv)
- [Executed notebook copy](70b_blocks/70b_blocks_analysis.ipynb)
- [Robustness figure](70b_blocks/figures/cell_18_00.png)

## Fulltrace E/N across all batches

All 105 individual trace files were validated. The analysis excludes 30 joy and
5 mechanical traces, retaining the following emotional/neutral observations:

| Batch | Emotional | Neutral |
|---|---:|---:|
| `batch_001_en` | 20 | 20 |
| `batch_002_joy` | 0 | 0 |
| `batch_003_enj` | 10 | 10 |
| `batch_004_enm` | 5 | 5 |
| **Total** | **35** | **35** |

The existing four throttle features and throttle-to-cycles ratio were tested.
Bonferroni correction covers five comparisons separately for each test family.

| Feature | Corrected Welch p | Corrected Mann–Whitney p |
|---|---:|---:|
| Mean rate | 0.4685 | 0.4107 |
| Variance | 1.0000 | 1.0000 |
| Spectral entropy | 0.5439 | 0.2423 |
| Slope | 1.0000 | 1.0000 |
| Throttle per cycle | 0.3052 | 0.3806 |

**None of the five hardware features passes the corrected 0.05 threshold.**
The inter/intra distance ratios range from 0.984 to 1.035. These findings do not
establish equivalence or absence of an effect.

The 70 traces come from 17 node IDs, with repeated observations. The retained
tests pool traces and treat them as independent; repeated-node dependence and
batch effects remain limitations. Results are exploratory.

The supplementary prompt type/token-ratio comparison also ran. It finds a text
difference between the E and N prompt sets (t-test p approximately 0.000003;
Mann–Whitney p approximately 0.000020). This is not a hardware-signal result and
does not adjust the hardware comparisons for workload or text differences.

- [Analyzed trace inputs](fulltrace_EN/pooled_whole_traces.csv)
- [Batch counts](fulltrace_EN/batch_condition_counts.csv)
- [Per-feature tests](fulltrace_EN/pairwise_tests.csv)
- [Feature distances](fulltrace_EN/feature_distances.csv)
- [Prompt TTR tests](fulltrace_EN/prompt_ttr_tests.csv)
- [Executed notebook copy](fulltrace_EN/fulltrace_analysis_EN.ipynb)
- [Feature distributions](fulltrace_EN/figures/cell_06_00.png)
- [Distance comparisons](fulltrace_EN/figures/cell_09_00.png)

## Preservation and verification

- [Validation results](validation.json): input coverage, unique recordings,
  matching raw metadata, successful execution, and unchanged historical files.
- [Input checksums](input_checksums.json): SHA-256 hashes of individual feature
  files used for validation/loading and the E/N prompt files.
- [Historical-file checksums](preserved_file_checksums.json): all pre-existing
  feature and result files, checked again after execution.
- [Original notebooks](prior_notebooks/): exact pre-run copies of both edited
  notebooks and the existing recomputed-trial reference notebook.

Only the main 70B blocks notebook and the fulltrace E/N notebook were updated.
The other analysis notebooks and existing results were left unchanged.
