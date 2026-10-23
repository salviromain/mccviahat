# Do node and date affect the measurements?

**Measurements vary within the same condition, especially across recording dates.
Comparisons remain useful when node and date are accounted for.**

The [node/day notebook](fulltrace_node_day_EN.ipynb) analyzes 70 Llama 70B
recordings: 35 emotional and 35 neutral, from 17 nodes and seven dates.
Dates use America/Chicago; midnight-crossing traces belong to their start date.

## Method

Analyze each condition separately, with one observation per whole recording:

1. Fit a **mixed-effects model** for each feature:
   `measurement = overall average + node contribution + date contribution + remaining variation`.
   Node and date are random effects, accounting for related recordings.
   Rates, variances, and throttle-per-cycle use log scales; entropy and slope do not.
2. Test node by comparing models with and without node, keeping date included.
   Test date similarly, keeping node included. **Parametric-bootstrap
   likelihood-ratio tests**, using up to 5,999 simulations per test, ask whether
   the improvement could arise without that contribution.
3. Apply **Holm correction** across all 68 tests: 17 features × two conditions
   × two effects. Corrected p < 0.05 supports an association under the model.

Method references: [mixed-effects models](https://www.statsmodels.org/stable/mixed_linear.html),
[bootstrap model comparisons](https://lme4.github.io/lme4/reference/pvalues.html),
[Holm correction](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html).

## Findings

| Condition | Features associated with date | Features associated with node |
|---|---:|---:|
| Emotional | 6 of 17 | 2 of 17 |
| Neutral | 8 of 17 | 1 of 17 |

For **power throttle (PTE)**, only spectral entropy and throttle-per-cycle passed
the corrected date tests, in both conditions. No PTE feature passed the node test.
The neutral mean-rate date test has uncertainty from the finite simulation count.

### How large were the observed PTE differences?

| Measurement | Across dates: emotional | Across dates: neutral | Between nodes: emotional | Between nodes: neutral |
|---|---:|---:|---:|---:|
| Mean throttle rate | +8.9% | +6.6% | 3.0% | 3.8% |
| Spectral entropy | −9.1% | −12.4% | 1.7% | 1.6% |
| Throttle per cycle | +10.6% | +11.6% | 3.1% | 3.9% |

These are **descriptive medians**, averaging repeated recordings within each
node/date/condition first:

- **Across dates:** six same-node comparisons per condition;
  `100 × (later − earlier) / earlier`.
- **Between nodes:** 37 same-date node-pair comparisons per condition;
  `100 × absolute difference / average of the two nodes`.

These percentages describe measurement values. Date comparisons span May to
September/October, rather than consecutive days.

## Implications

- Compare conditions within the **same node and date**, accounting for these groups
  when combining comparisons.
  Both conditions have equal recording counts in all 23 node/date groups.
- Only six nodes recur across dates, always across batches. Date associations
  may reflect batch or setup differences.
- Results are exploratory and model-dependent. Nonsignificance does not prove
  stability. Whether an emotional–neutral difference survives node/date adjustment
  has not yet been tested.

Detailed estimates, uncertainty intervals, and diagnostic checks are in the
[saved results](../../results/fulltrace_node_day_2026-10-21/README.md).
