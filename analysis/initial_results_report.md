# Initial Analysis Report: Hardware Substrate Signatures of Emotional vs. Neutral LLM Prompts

**Date:** 11 February 2026  
**Author:** Romain Salvi  
**Status:** Pilot study — single run per condition, 20 prompts each

---

## 1  Executive Summary

We ran Llama-2-7B (CPU-only, llama.cpp) on a CloudLab bare-metal node under two
prompt conditions — 20 neutral (WikiHow) and 20 emotional (Creepypasta) — while
recording 16 hardware/software events at 1 ms resolution.

| Analysis | Result | Meaning |
|---|---|---|
| Raw PERMANOVA | F = 7.00, **p = 0.0001** | The two conditions produce clearly different interrupt fingerprints |
| Residualised PERMANOVA | F = 1.23, **p = 0.19** | After removing confounders (CPU load, request duration, etc.), the difference **vanishes** |
| PERMDISP | F = 24.66, p = 0.0004 | The groups also differ in *variability*, not just averages |
| Mann-Whitney U | 84/162 features at p < 0.05 | Many individual features differ (uncorrected) |

**Bottom line:** There *is* a strong raw signal separating the two conditions — but
it is largely **explained by confounders** (CPU utilisation, request duration, CPU
frequency) rather than emotional content *per se*. This is an instructive null result
for the pilot, not a failure.

---

## 2  Experimental Setup

| Component | Detail |
|---|---|
| **Hardware** | CloudLab bare-metal, 16 cores pinned |
| **Model** | Llama-2-7B-Chat GGUF, CPU-only via llama.cpp in Docker |
| **Prompts** | 20 neutral (WikiHow, mean 641 tokens) + 20 emotional (Creepypasta, mean 626 tokens) |
| **Generation** | 50 tokens per prompt, fixed length |
| **Run duration** | ~506 s (8.4 min) per condition, all 20/20 prompts completed |
| **Perf events** | 16 events at 1 ms: IRQ entry/exit, softirqs, TLB flush, context-switches, cpu-migrations, page-faults, cpu-clock, power/thermal counters |
| **System metrics** | /proc/interrupts, softirqs, PSI, net, disk, CPU freq — sampled at 200 ms |

Prompt length is well balanced (641 vs. 626 tokens, p = 0.42). No data quality
issues in either run.

---

## 3  Feature Engineering

Each prompt becomes one observation (n = 40 total). From the 1 ms perf time series
in each request window, we extract **160 features** across five categories:

| Category | Count | What it captures |
|---|---|---|
| **Frequency** | 32 | Event rates (events/second) and relative shares |
| **Temporal Structure** | 32 | Regularity vs. burstiness (entropy, Fano factor) |
| **Burst Statistics** | 80 | Properties of above-threshold activity bursts |
| **Cross-Modal Coupling** | 18 | Interaction between event pairs (cross-correlation, mutual information) |
| **Complexity** | 32 | Information-theoretic complexity (Lempel-Ziv, permutation entropy) |

### Notable Raw Differences

| Event | Neutral (events/s) | Emotional | Ratio E/N |
|---|---|---|---|
| context-switches | 157,782 | 98,221 | **0.62** |
| cpu-migrations | 27,817 | 18,601 | **0.67** |
| page-faults | 24,447 | 34,298 | **1.40** |
| softirq entry | 10,897 | 9,716 | **0.89** |
| cpu-clock | 63,996 | 63,999 | **1.00** |

Almost all events are lower in emotional, except page-faults. This pattern is
consistent with shorter request windows in emotional (mean 13.4s vs. 17.6s) —
which is exactly why request duration is a confounder.

---

## 4  Confounders

We identified five confounders — variables that mechanically drive feature differences
independently of emotional content:

| Confounder | Balanced? | Why it matters |
|---|---|---|
| Prompt length (tokens) | ✓ p = 0.42 | More tokens → longer eval → different CPU profile |
| Temporal order | ✓ p = 1.00 | CPU warms up over 8 min |
| CPU utilisation | **⚠ p = 0.0008** | System load differs between conditions |
| CPU frequency | **⚠ p = 0.0011** | Frequency scaling differs |
| Request duration | **⚠ p = 0.0004** | Neutral takes longer (17.6s vs. 13.4s per request) |

Three of five are significantly imbalanced. Together, they explain **57% of feature
variance** on average. The neutral prompts take ~4 seconds longer per request despite
similar token counts — likely due to different text structures affecting inference speed.

---

## 5  Statistical Results

### Univariate Screening (Mann-Whitney U)

162 features tested; **84 significant at p < 0.05** (uncorrected). Top features are
hardware IRQ complexity and rate — both lower in emotional, with large effect sizes
(|r| > 0.7). However, these do **not** control for confounders and are uncorrected
for multiple comparisons.

### Multivariate (PERMANOVA)

| | pseudo-F | p-value | Interpretation |
|---|---|---|---|
| **Raw** (no confounder control) | 7.00 | **0.0001** | Strong separation |
| **Residualised** (confounders removed) | 1.23 | **0.19** | No significant effect |

The pseudo-F drops from 7.00 to 1.23 after removing confounder variance — the
confounders explain most of the apparent condition difference.

### Dispersion Check (PERMDISP)

PERMANOVA assumes groups have equal spread. Both raw and residualised data violate
this (p = 0.0004 and p = 0.04) — the emotional condition produces more *variable*
hardware signatures, likely because Creepypasta prompts are more heterogeneous in
style than WikiHow.

---

## 6  Interpretation

### What Went Right

1. **The pipeline works.** 16 events at 1 ms for 8+ minutes, no gaps, no data issues,
   20/20 prompts completed in both conditions.
2. **Real signal exists.** The raw PERMANOVA (p = 0.0001) confirms that the interrupt
   substrate *is* measurably different between conditions.
3. **Sound methodology.** Confounders identified, balance-checked, residualised,
   dispersion assumptions verified.

### The Challenge

The hardware differences are largely attributable to *how long and how hard* the model
works on each prompt type, not the emotional content itself. We want to isolate the
effect of prompt content on the substrate, but it's entangled with different inference
dynamics (speed, CPU load, thermal state).

### The Key Question

**Is there a substrate signature of emotional processing *beyond* the mechanical
effects of different computational loads?** The pilot can't answer this yet — we need
replications, better prompt matching, and a fixed per-process CPU tracker.

---

## 7  Known Issues & Next Steps

### Instrumentation Bug

The per-process CPU tracker (`proc_sample.csv`) is recording the wrong PID — likely
the Docker wrapper instead of the actual llama.cpp process. **Must fix before next
run** (use `docker top` or `pgrep` to find the real PID).

### Next Steps

1. **Fix PID tracking** in `substrate_collector.py`
2. **Run replications** — ≥ 10 runs per condition (~9 min each, reset server between runs)
3. **Better prompt matching** — match on processing time, not just token count
4. **Interleave conditions** within runs to control temporal drift
5. **Analysis extensions** — classifier with cross-validation, time-resolved analysis

---

## 8  What to Tell Your Professor

> "The pipeline is working — we collected 16 perf events at 1 ms resolution for
> 20 prompts per condition over 8-minute runs with no data quality issues. Raw
> multivariate analysis shows highly significant separation between conditions
> (PERMANOVA F = 7.0, p = 0.0001), confirming that different prompt types produce
> measurably different hardware interrupt fingerprints.
>
> However, after controlling for confounders — primarily request duration, CPU
> utilisation, and CPU frequency — the effect becomes non-significant (F = 1.2,
> p = 0.19). The current signal is mostly explained by *how long* the model works
> on each prompt type, not the emotional content itself.
>
> Next steps: fix the per-process CPU tracker (wrong PID), collect ≥ 10 replications
> per condition, and improve prompt matching on processing time. The methodology
> is solid and ready to scale."

---

## Appendix: Key Numbers

| Metric | Value |
|---|---|
| Perf events | 16 (14 non-zero), 1 ms resolution |
| Prompts per condition | 20 (all completed) |
| Run duration | 506 s each |
| Feature matrix | 40 × 160 |
| Confounders | 5 (mean R² = 0.567) |
| Raw PERMANOVA | F = 7.00, p = 0.0001 |
| Residualised PERMANOVA | F = 1.23, p = 0.19 |
| PERMDISP (raw / residualised) | p = 0.0004 / p = 0.04 |
| Top univariate feature | IRQ complexity, p < 0.0001, r = −0.77 |
