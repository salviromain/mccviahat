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

Result: **22 matched pairs**, from **69 runs**, on **17 nodes** across **7 dates**. 10 pairs have 1 E + 1 N run; 12 pairs have 2 E + 2 N runs.

---

## Step 3 — Average within each pair

If a pair has two E runs, we average them; same for N:

```
E average = (E run 1 + E run 2) / 2
N average = (N run 1 + N run 2) / 2
```

**Why?** So that every node/date pair counts **once**. A pair with two runs per condition is not two independent experiments: same machine, same day.

---

## Step 4 — Compare E with N as a ratio

```
ratio      = E average / N average
difference = (ratio − 1) × 100   → "% higher in E than N"
```

**Example (node 244, 2026-05-04, mean rate):**
E average = 1.157 × 10⁹, N average = 1.084 × 10⁹
ratio = 1.067 → **E is 6.7% higher**.


Behind the scenes, the model works with `log(ratio)` instead of the ratio itself. Logs make "+10%" and "−10%" behave symmetrically, and they make averaging percentages fair. Every number in this document is converted back to a plain %.

---

## Step 5 — Combine the 22 differences into one overall answer

A simple approach would be averaging the 22 percentages. The problem: pairs from the **same date** may resemble each other (e.g. a warm day), and so may pairs from the **same node**. A plain average would count six pairs from one date as six independent confirmations, which overstates the evidence.

So we describe each pair's difference as:

```
pair difference = overall effect + node bump + date bump + leftover noise
```

- **overall effect** — the number we want: the typical E−N difference.
- **node bump** — some machines may react a bit more or less than average.
- **date bump** — some days may show a bit more or less than average.
- **leftover noise** — everything else (measurement noise, chance).

The model finds the overall effect that best explains all 22 pairs, while allowing for these bumps. (Technical name: a *mixed model with crossed random effects*.)

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

- **Drop one date at a time** (7 refits) and **drop one node at a time** (17 refits). If the result flipped when one date or machine was removed, it would depend on that one group.
- **Count directions**: in how many pairs and dates is E higher?
- **Q–Q plot**: checks whether the leftover noise looks roughly bell-shaped, as the model assumes.

---

## Results

| | PTE mean rate | PTE / cycles |
|---|---|---|
| **Overall: E higher than N by** | **+3.6%** | **+3.9%** |
| 95% CI for the average | +1.1% to +6.4% | +1.0% to +6.8% |
| Corrected p (Bonferroni) | **0.023** | **0.035** |
| Pairs where E > N | 20 of 22 | 19 of 22 |
| Dates where E > N on average | 6 of 7 | 6 of 7 |
| Range for one new node/date | −4.9% to +13.0% | −5.6% to +14.4% |
| Drop-one-date estimates | +2.9% to +4.3% (7/7 positive) | +3.0% to +4.7% (7/7 positive) |
| Drop-one-node estimates | +2.9% to +4.0% (17/17 positive) | +3.0% to +4.3% (17/17 positive) |

### In plain words

1. **Emotional prompts produced about 3.5–4% more CPU throttling than neutral prompts** on the same machine and day. Both measures agree.
2. **It is unlikely to be chance.** Both corrected p-values (0.023 and 0.035) are below the 0.05 threshold.
3. **It is consistent.** E was higher in 19–20 of the 22 pairs and on 6 of 7 dates. Removing any single date or machine never changed the direction, and the size stayed between about +3% and +4.7%.

4. **Machines differ more than days.** Most of the pair-to-pair variation comes from which node was used (about 45–60%), not which date (about 10%)!

### Caveats
- Matching removes machine and day differences, but not everything (e.g. which condition ran first, setup details).
- The best confirmation would be **new recordings** on new node/date pairs.
