# Interpreting the three-batch full-trace results

Results reviewed on 2026-10-01 from [fulltrace_analysis_NEJ.ipynb](fulltrace_analysis_NEJ.ipynb) and its exported tables in [dataNEW/analysis_NEJ](../../dataNEW/analysis_NEJ/).

The strongest pooled finding is a **higher mean throttle rate in joy traces** than in either neutral or emotional traces. Joy also has higher throttle variance, supported by the corrected Mann–Whitney tests but not the corrected Welch tests. **None of the four features significantly distinguishes emotional from neutral after correction.** These findings describe this collection of traces; uneven batch composition and repeated measurements limit their interpretation as effects of condition.

## What was analyzed

The notebook pools 90 whole traces, with 30 observations per condition, using only four `core_power.throttle` features: mean rate, variance, spectral entropy, and slope. It does not use instructions, cycles, TLB features, or LZ complexity.

| Collection batch | Neutral | Emotional | Joy |
| --- | ---: | ---: | ---: |
| Old | 20 | 20 | 0 |
| First | 0 | 0 | 27 |
| Second | 10 | 10 | 3 |
| **Total** | **30** | **30** | **30** |

Equal pooled class counts do not imply balanced sampling within batches. Ninety percent of joy traces come from the first batch, whereas two-thirds of neutral and emotional traces come from the old batch.

For each feature, the notebook compares all three condition pairs using Welch's t-test and a two-sided Mann–Whitney U test. Each test family receives its own Bonferroni correction across 12 comparisons. Significance below means **corrected p < 0.05**. The two test families are not jointly corrected as a single family.

## Feature-level findings

The following are pooled means. Mean rate is shown in units of 10⁹ and variance in units of 10¹³ for readability; these are the extractor's numerical feature units, not percentages of CPU throttling. Slope is fitted against sample index, not elapsed seconds.

| Feature | Neutral | Emotional | Joy |
| --- | ---: | ---: | ---: |
| Mean rate / 10⁹ | 1.1697 | 1.1959 | 1.2571 |
| Variance / 10¹³ | 3.6130 | 3.5904 | 4.1046 |
| Spectral entropy | 0.8723 | 0.8921 | 0.8629 |
| Slope | 11.8967 | 12.0380 | 10.0249 |

### Mean rate: the clearest pooled difference

Joy's mean rate is approximately **7.5% higher than neutral** and **5.1% higher than emotional**. Both tests support these comparisons after correction:

| Comparison | Corrected Welch p | Corrected Mann–Whitney p | Absolute standardized mean difference |
| --- | ---: | ---: | ---: |
| Neutral–joy | 3.13 × 10⁻⁶ | 1.09 × 10⁻⁶ | 1.65 |
| Emotional–joy | 5.63 × 10⁻⁵ | 2.78 × 10⁻⁵ | 1.39 |

The standardized differences are large relative to the observed within-condition spread. The notebook calls this statistic `cohens_d` and computes the absolute mean difference divided by the square root of the average sample variance. Direction must therefore be read from the means, not the sign of this statistic.

Emotional exceeds neutral by about 2.2%, but that comparison is not significant: both corrected p-values are 1.00. A value capped at 1.00 by correction does not imply that the conditions are identical.

### Variance: higher in joy, with test-dependent support

Joy's mean variance is approximately 13.6% higher than neutral and 14.3% higher than emotional. Standardized differences are 0.72 and 0.74, respectively.

| Comparison | Corrected Welch p | Corrected Mann–Whitney p |
| --- | ---: | ---: |
| Neutral–joy | 0.0886 | **0.00304** |
| Emotional–joy | 0.0706 | **0.000673** |

The rank-based tests detect a difference, while the mean-based Welch tests do not meet the corrected threshold. This is evidence of a distributional difference under the test assumptions, but not agreement between both tests about a mean shift. Mann–Whitney should not automatically be interpreted as a test of medians when distribution shapes may differ.

Neutral and emotional have almost identical mean variance, with an absolute standardized difference of 0.032 and no significant comparison.

### Spectral entropy and slope: no corrected significance

None of the six feature/pair comparisons for spectral entropy or slope survives correction. Emotional–joy spectral entropy has a noticeable standardized difference (0.71), but corrected p-values are 0.0977 for Welch and 0.129 for Mann–Whitney. It should not be reported as a significant result.

Joy has a lower average slope than the other classes, but the observed differences are insufficient to establish separation with these tests. Failure to reject a difference is not an equivalence result or proof that a larger study would find no effect.

Overall, **two comparisons survive corrected Welch testing and four survive corrected Mann–Whitney testing**. All involve joy; none is emotional–neutral.

## How the distance analysis fits

The inter/intra ratio divides the mean cross-condition absolute feature distance by the average of the two within-condition distances. It is descriptive, not a p-value or a classifier score.

- Mean rate gives the strongest separation: **1.81 for neutral–joy** and **1.56 for emotional–joy**.
- Variance gives smaller joy-related ratios: **1.15** and **1.26**.
- All emotional–neutral ratios are close to one: **1.020, 0.997, 1.031, and 0.983** for mean rate, variance, spectral entropy, and slope.
- The other entropy/slope ratios range from about 1.00 to 1.12.

This agrees with mean rate being the clearest individual feature for joy comparisons, while emotional–neutral distances resemble within-condition variation. These univariate summaries do not establish multivariate classification accuracy; this notebook does not train a classifier.

## Why collection batch matters

The following additional descriptive means were computed from the notebook's exported `pooled_whole_traces.csv`. They are a diagnostic breakdown, not new hypothesis tests.

| Batch | Condition | Mean rate / 10⁹ | Spectral entropy |
| --- | --- | ---: | ---: |
| Old | Neutral | 1.1386 | 0.9055 |
| Old | Emotional | 1.1781 | 0.9187 |
| First | Joy | 1.2569 | 0.8692 |
| Second | Neutral | 1.2319 | 0.8060 |
| Second | Emotional | 1.2314 | 0.8390 |
| Second | Joy | 1.2588 | 0.8062 |

Neutral and emotional mean rates both increase between the old and second batches. In the second batch, their mean rates are almost equal, and joy is only about 2.2% above either. Joy and neutral also have almost identical second-batch mean spectral entropy. Thus, pooled contrasts are not a substitute for comparing conditions within a shared collection batch.

The second batch offers all three conditions, but it contains only **three joy traces**, limiting precision. Its inclusion reduces the complete separation of joy by batch found in the earlier collection, but does not eliminate batch confounding from the pooled analysis.

The tests also treat individual traces as independent. Repeated runs on the same node may be correlated, so the effective independent sample size may be smaller than 90. Bonferroni correction addresses multiple comparisons; it does not correct batch confounding or dependence between observations.

## Prompt type/token ratios

The supplementary TTR analysis uses 20 prompt texts per condition, counted once per text, rather than 30 hardware traces per condition. It measures unique lowercase whitespace tokens divided by total whitespace tokens.

| Prompt set | Mean TTR |
| --- | ---: |
| Neutral | 0.6044 |
| Emotional | 0.6723 |
| Joy | 0.5988 |

Emotional prompts have higher TTR than both other sets under both corrected tests. Neutral–joy is not significant. Corrected Mann–Whitney p-values are 0.000061 for neutral–emotional and 0.000033 for emotional–joy.

This identifies another difference between the prompt sets, but does not establish that lexical diversity causes the hardware results. TTR depends on text length, uses whitespace words rather than model tokens, and does not adjust the hardware comparisons for workload. The analysis describes the configured prompt files; it is not a per-trace audit of which text was executed.

## Interpretation and next analysis

A defensible statement is:

> Across 90 pooled whole traces, joy is associated with higher mean throttle rate than neutral and emotional conditions under both corrected test families. Joy-related variance differences are supported by corrected rank-based tests only. No emotional–neutral comparison, spectral-entropy comparison, or slope comparison survives correction. Uneven collection-batch composition and repeated-node dependence prevent attributing these associations to condition alone.

These results do not establish consciousness, subjective experience, or a causal emotional response in the hardware. They establish associations in measured throttle features under the current experimental design.

The most useful next steps are to report within-batch comparisons alongside pooled results, collect more joy traces in batches that also contain neutral and emotional traces, and account for node-level dependence using an analysis matched to the experimental pairing or clustering. Independent prompt sets and balanced collection order would help distinguish condition-related behavior from repeatable workload or collection effects.
