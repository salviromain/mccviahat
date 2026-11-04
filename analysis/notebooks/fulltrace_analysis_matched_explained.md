# Emotional vs neutral prompts, accounting for node and day.

*Companion to `fulltrace_analysis_matched.ipynb`. Results as of 2026-11-03 (all batches through 2 Nov); outputs in `results/fulltrace_matched_EN_2026-11-03_two_features/`.*

---

## The question

When the 70B model answers **emotional (E)** prompts, does the CPU throttle more than when it answers **neutral (N)** prompts?

"Throttling" is counted as **PTE**: processor throttle events, the moments when the CPU slows itself down because of power or heat limits.

---

## Step 1 — Turn each run into two numbers

Every recording (a "run") gets two measurements, both as **whole-recording totals**:

```
PTE / cycles   = Σ throttled cycles / Σ unhalted CPU cycles     ← primary
PTE mean rate  = Σ throttled cycles / run duration (seconds)    ← secondary
```

**Why is PTE / cycles primary?** It asks how much throttling happens **per unit of CPU work**. Numerator and denominator come from the same counter reads, so it doesn't depend on how long a run lasted or on how well perf's sampling kept up (perf was asked for 1 ms samples but delivered ~11 ms on average). Fixing one primary outcome also reduces the "we picked the measure after looking" problem.

---

## Step 2 — Only compare runs from the same machine on the same day

We group runs by **node** (the physical machine) and **calendar date** (Chicago time). We keep only node/date pairs that have at least one E run **and** one N run.

**Why?** Throttling differs from machine to machine, and from day to day. Those differences can be bigger than the E/N difference we are looking for. Comparing E and N on the same machine on the same day cancels most of that out — like weighing two objects on the same scale instead of on two different scales.

Result: **38 node/dates**, from **158 runs**, on **25 nodes** across **9 dates**:
- 10 node/dates with 1 E + 1 N, 16 with 2 E + 2 N, 1 with 2 E + 1 N (node 187, 25 Oct — its second N run was not collected),
- 11 node/dates from the **2 Nov** batch with 3 E + 3 N (or 4 + 4).

That gives **189 E−N comparisons** per measure. One N run (node 232, 12 May) has no E partner and is dropped.

---

## Step 3 — Compare every E run with every N run

Within each node/date, every E run is compared with every N run. With two runs per condition that gives **four** comparisons:

```
E1 vs N1,  E1 vs N2,  E2 vs N1,  E2 vs N2
```

**Example (node 244, 2026-05-04, mean rate, in units of 10⁹ events/s):**
E1 = 1.185, E2 = 1.128, N1 = 1.079, N2 = 1.089

| | vs N1 | vs N2 |
|---|---|---|
| **E1** | +9.8% | +8.9% |
| **E2** | +4.5% | +3.6% |

E wins all 4 comparisons here, and is on average about 6.7% higher.

These comparisons reuse the same runs, so they are **not independent results**: if E1 happened to be high, both E1 comparisons look high. Both analyses below account for this.

We then ask two different questions of these comparisons.

---

## Step 4 — Primary: how *often* is E higher? (θ, "probability of superiority")

For each node/date, count the share of its E-vs-N comparisons in which E is higher (a tie counts as ½):

```
θ(node/date) = (comparisons with E > N) / (number of comparisons)
θ            = average of θ over node/dates   (each node/date counts once)
```

- **θ = 0.5** → E and N are indistinguishable; **θ = 1** → every E run beat every N run.
- In the example above, θ = 4/4 = 1.
- Statistically, θ for one node/date is its Mann–Whitney statistic, the same as the area under an ROC curve.
- **Per-pair θ** pools all comparisons instead, so a 3 + 3 node/date (9 comparisons) counts 9× a 1 + 1 one. The large 2 Nov groups weigh heavily there.

**Why this is the primary analysis:** it uses only "which run is higher", so it needs no assumption about bell-shaped noise, and a single extreme run can't dominate it.

**Uncertainty (95% interval):** resample whole node/dates with replacement 20,000 times and recompute θ (a *cluster bootstrap*). As a check, we also resample whole nodes or whole dates.

**p-value (is θ above 0.5 by more than chance?):** inside each node/date, shuffle which runs are called "E" and which "N" (keeping the counts), and see how often the shuffled data gives a θ at least as high as the real one. Because shuffles in different node/dates are independent, we can compute this **exactly** (every possible shuffle, no simulation noise). One-sided, because the original study predicted E > N.

---

## Step 5 — Secondary: how *much* higher is E? (mixed model)

θ ignores the size of the gap. To estimate it in %, we model each comparison on the log scale as

```
log(E / N) = overall effect + node bump + date bump + node/date bump
           + E-run noise − N-run noise
```

- **overall effect** — the typical E−N difference, converted back to %.
- **node / date / node/date bumps** — machines, days, or one machine on one day may react a bit more or less.
- **run noise** — comparisons that share a run share its noise; this is how the model knows they are linked. (In a 2 + 2 node/date the fourth comparison is exactly determined by the other three, so it adds no information.)

**p-value:** simulate 19,999 datasets where the true effect is zero but all the variation is as large as in our data, and count how often they look at least as convincing as the real one. Bonferroni-corrected for the two measures (p × 2).

**Intervals:** a 95% CI for the average difference, and a wider **new-pair range** for one comparison on a new machine on a new day.

**Robustness:** refit after dropping one date at a time (9 refits) or one node at a time (25 refits).

---

## Results

### Primary — θ for PTE / cycles (one-sided exact p for E > N)

| Data | Node/dates | θ per group (95% interval) | θ per pair | p (group / pair) |
|---|---|---|---|---|
| **All (main)** | 38 | **0.69 (0.57 to 0.80)** | 0.61 (0.51 to 0.73) | **0.0005** / 0.018 |
| Without node 183 | 37 | 0.71 (0.60 to 0.82) | 0.63 | 0.0002 / 0.010 |
| Without node 187 | 37 | 0.71 (0.60 to 0.82) | 0.62 | 0.0002 / 0.014 |
| Before 2 Nov | 27 | 0.74 (0.59 to 0.88) | 0.70 | 0.0009 / 0.0037 |
| Before 2 Nov, without 187 (professor's set) | 26 | 0.77 (0.62 to 0.89) | 0.72 | 0.0003 / 0.0018 |
| **2 Nov alone** | 11 | **0.57 (0.41 to 0.73)** | 0.56 | 0.19 / 0.24 |
| 25 Oct alone | 5 | 0.45 (too few groups for a reliable interval) | 0.50 | 0.68 / 0.57 |

- Resampling nodes or dates instead of node/dates gives similar intervals for the main θ: 0.58–0.79 (nodes), 0.57–0.87 (dates).
- **By date** (θ): 4 May 0.50 · 6 May 1.00 · 7 May 1.00 · 10 May 0.80 · 12 May 1.00 · 30 Sep 0.55 · 12 Oct 1.00 · 25 Oct 0.45 · 2 Nov 0.57.
- E beat N in every comparison in 19 node/dates; N beat E in every comparison in 4.
- PTE mean rate (secondary) is similar: θ = 0.71 (0.60 to 0.82), p = 0.0001; on 2 Nov alone 0.63 (p = 0.038).
- On the professor's 26-group set this notebook reproduces the professor's numbers (0.77, p = 0.0002; 25 Oct without 187: 0.56, p = 0.42), so the two pipelines agree.

### Secondary — average size of the gap (mixed model)

| | PTE / cycles | PTE mean rate |
|---|---|---|
| **Overall: E higher than N by** | **+2.2%** | **+2.2%** |
| 95% CI for the average | −0.4% to +5.1% | −0.0% to +4.5% |
| Corrected p (Bonferroni) | 0.136 | 0.065 |
| Comparisons where E > N | 116 of 189 | 124 of 189 |
| Node/dates where E > N on average | 26 of 38 | 29 of 38 |
| Dates where E > N on average | 6 of 9 | 7 of 9 |
| Range for one new comparison (new node, new date) | −8.3% to +13.9% | −6.7% to +12.0% |
| Drop-one-date estimates | +1.6% to +2.8% (9/9 positive) | +1.8% to +2.7% (9/9 positive) |
| Drop-one-node estimates | +1.8% to +2.5% (25/25 positive) | +1.9% to +2.5% (25/25 positive) |

About 76% of the spread in a single comparison is run-to-run noise; node, date and node/date add the rest.

### In plain words

1. **E is higher than N more often than chance.** On the same machine and day, an emotional run out-throttles a neutral run about **69%** of the time (chance = 50%), and the exact test makes chance very unlikely (p = 0.0005). This holds without node 183 or 187.
2. **But the gap is small, and its size is not clearly beyond chance.** On average E is about **+2%** higher. After correcting for two measures, p = 0.14 (per cycle) and 0.065 (per second). The two analyses differ because θ only counts *which* run is higher, while the mixed model also weighs *how much* — and the ~2% gap is small compared with run-to-run noise.
3. **The newest data replicate weakly.** The 2 Nov batch, the largest so far (11 node/dates, 3–4 runs per condition), gives θ = 0.57 on its own, not significant for PTE / cycles. 30 Sep (0.55) and 25 Oct (0.45) were weak too. The strong dates are mostly the early ones (May, 12 Oct). Adding 2 Nov lowered θ from 0.74 to 0.69 (0.77 → 0.69 on the professor's set) and the mixed-model gap from about +2.7% to +2.2%; before 25 Oct the gap was about +4% and passed the correction.
4. **The result depends on pooling across days.** The direction never flips when any single date or node is dropped, but 3 of the last 4 collection dates show little or no effect on their own.

### Caveats
- Matching removes machine and day differences, but not everything — e.g. run order (on 12 Oct, a strong date, E always ran last) and setup details.
- PTE / cycles was declared primary after the five candidate measures had been inspected, so the p-values remain somewhat optimistic; treat the results as exploratory.
- Bootstrap intervals with 9–38 clusters are approximate.
- All runs share one hardware type, model, quantization and the same 20 prompts per condition.
- The mechanical (M) control is analysed separately (`fulltrace_analysis_matched_mechanical.ipynb`, `fulltrace_analysis_matched_ENM.ipynb`); in the professor's analysis M throttled at least as much as E, which matters for how any E–N difference is interpreted.
- The best confirmation would be **new recordings** on new node/date pairs with rotated condition order.
