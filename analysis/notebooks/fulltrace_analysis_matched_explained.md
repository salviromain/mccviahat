# Emotional vs neutral prompts, accounting for node and day.


---

## The question

When the 70B model answers **emotional (E)** prompts, does the CPU throttle more than when it answers **neutral (N)** prompts?

"Throttling" is counted as **PTE**: processor throttle events, the moments when the CPU slows itself down because of power or heat limits.

---

## Step 1 — Turn each run into two numbers

Every recording (a "run") gets two measurements:

```
PTE mean rate  = total PTE events / run duration (seconds)
PTE / cycles   = total PTE events / total CPU cycles 
```

**Why two?**
- *Mean rate* is the most direct measure: how often throttling happens.
- *PTE / cycles* asks how much throttling happens **per unit of CPU work**. It is less sensitive to a run being faster, slower, longer or shorter, so it is a useful cross-check.

---

## Step 2 — Only compare runs from the same machine on the same day

We group runs by **node** (the physical machine) and **calendar date**. We keep only node/date pairs that have at least one E run **and** one N run.

**Why?** Throttling differs from machine to machine, and from day to day (temperature, load, etc.). Those differences can be bigger than the E/N difference we are looking for. Comparing E and N on the same machine on the same day cancels most of that out. It's like weighing two objects on the same scale, instead of on two different scales.

Result: **27 node/dates**, from **88 runs**, on **21 nodes** across **8 dates**. 10 node/dates have 1 E + 1 N run, 16 have 2 E + 2 N runs, and 1 has 2 E + 1 N. That gives **76 E−N comparisons** per measure.

---

## Step 3 — Compare every E run with every N run

Within each node/date, every E run is compared with every N run. With two runs per condition that gives **four** comparisons:

```
E1 vs N1,  E1 vs N2,  E2 vs N1,  E2 vs N2
difference = (E run / N run − 1) × 100   → "% higher in E than N"
```

With one run each there is one comparison. With 2 E runs and 1 N run there are two.

**Example (node 244, 2026-05-04, mean rate, in units of 10⁹ events/s):**
E1 = 1.185, E2 = 1.128, N1 = 1.079, N2 = 1.089

| | vs N1 | vs N2 |
|---|---|---|
| **E1** | +9.8% | +8.9% |
| **E2** | +4.5% | +3.6% |

Their average is **E about 6.7% higher** on this node and day.

Behind the scenes, the model works with `log(E / N)` instead of the ratio itself. Logs make "+10%" and "−10%" behave symmetrically, and they make averaging percentages fair. Every number in this document is converted back to a plain %.

---

## Step 4 — The four comparisons are not four independent results

The four comparisons reuse the same four runs. If E1 happened to come out high, *both* E1 comparisons look high. In fact the fourth comparison can always be computed from the other three:

```
(E1 vs N1) + (E2 vs N2) = (E1 vs N2) + (E2 vs N1)      (exactly, on the log scale)
```

Treating the four as four separate confirmations would overstate the evidence. So the model is told which comparisons **share a run** (Step 5), and the redundant fourth comparison adds no information. It is still shown in the plots and counts.

---

## Step 5 — Combine all comparisons into one overall answer

A simple average of all comparisons would over-count in several ways: comparisons that share a run, comparisons from the **same node/date**, and node/dates from the **same date** (e.g. a warm day) or the **same node**. So we describe each comparison as:

```
comparison = overall effect + node bump + date bump + node/date bump
           + E-run noise − N-run noise
```

- **overall effect** — the number we want: the typical E−N difference.
- **node bump** — some machines may react a bit more or less than average.
- **date bump** — some days may show a bit more or less than average.
- **node/date bump** — one machine on one particular day may sit higher or lower.
- **run noise** — each run comes out a bit higher or lower by chance. Comparisons that share a run share its noise; this is how the model knows they are linked.

The model finds the overall effect that best explains all comparisons, while allowing for all of these. (Technical name: a *mixed model with crossed random effects*.)

---

## Step 6 — Is the effect real, or could it be chance? (p-value)

We check by simulation:

1. Imagine a world where E and N are **truly identical** (overall effect = 0), but the node, date and noise variation stay as large as in our data.
2. Generate **19,999 fake datasets** from that world.
3. For each one, measure how strongly it seems to show an effect.
4. Count how often a fake dataset looks **at least as convincing as our real data**.

```
p = (number of fake datasets at least as convincing + 1) / (19,999 + 1)
```

A small p means pure chance rarely produces a result like ours.

**Correcting for testing two measures (Bonferroni correction).** If you test two things, you get two chances to be fooled by luck. Bonferroni's rule: multiply each p-value by the number of tests (here 2), capped at 1. A corrected p below **0.05** is the usual threshold for "unlikely to be chance".

---

## Step 7 — How big is the effect, and how sure are we? (intervals)

Again by simulation: generate **1,999 datasets** from the fitted model and re-estimate the effect each time. The spread of those estimates gives:

- **95% confidence interval (CI)** — the range where the **average** E−N difference most plausibly lies.
- **New-pair range** — where the difference for **one new machine on one new day** would likely fall. This is wider, because it also includes the node, date and noise variation. A single new comparison can easily go either way even when the average is clearly positive.

---

## Step 8 — Robustness checks

- **Drop one date at a time** (8 refits) and **drop one node at a time** (21 refits). If the result flipped when one date or machine was removed, it would depend on that one group.
- **Count directions**: in how many comparisons, node/dates and dates is E higher?
- **Q–Q plot**: checks whether the leftover noise looks roughly bell-shaped, as the model assumes.

---

## Results

| | PTE mean rate | PTE / cycles |
|---|---|---|
| **Overall: E higher than N by** | **+2.6%** | **+2.7%** |
| 95% CI for the average | +0.3% to +5.0% | −0.1% to +5.3% |
| Corrected p (Bonferroni) | **0.059** | **0.112** |
| Comparisons where E > N | 53 of 76 | 53 of 76 |
| Node/dates where E > N on average | 22 of 27 | 21 of 27 |
| Dates where E > N on average | 6 of 8 | 6 of 8 |
| Range for one new comparison (new node, new date) | −8.2% to +14.6% | −9.6% to +16.6% |
| Drop-one-date estimates | +2.2% to +3.3% (8/8 positive) | +2.1% to +3.5% (8/8 positive) |
| Drop-one-node estimates | +2.2% to +2.9% (21/21 positive) | +2.2% to +3.0% (21/21 positive) |

### In plain words

1. **Emotional prompts produced about 2.5–3% more CPU throttling than neutral prompts** on the same machine and day. Both measures agree.
2. **It is no longer clearly beyond chance.** After correcting for two measures, p = 0.059 and 0.112, just above the 0.05 threshold. Before the newest batch (2026-10-25) was added, the effect was larger (about +4%) and passed the threshold. On that day, 2 of 5 nodes showed N higher than E.
3. **The direction is consistent.** E was higher in 22 of 27 node/dates and on 6 of 8 dates. Removing any single date or machine never changed the direction, and the size stayed between about +2.1% and +3.5%.
4. **Most of the variation is run-to-run noise.** About 85% of the spread in a single comparison comes from one run simply coming out higher or lower than another. Node, date and node/date add little. This is why a single comparison can easily go either way.

### Caveats
- Matching removes machine and day differences, but not everything (e.g. which condition ran first, setup details).
- These two measures were chosen after looking at the data, so treat the result as exploratory.
- The best confirmation would be **new recordings** on new node/date pairs.
