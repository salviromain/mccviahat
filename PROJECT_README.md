# Discovering Machine Correlates of Consciousness

## Overview

This project investigates whether the hardware substrate of a computer running a Large Language Model produces measurably different signals when the model processes emotionally charged prompts versus neutral ones. The work is motivated by the analogy between Neural Correlates of Consciousness (NCCs) in biological systems and hypothesised Machine Correlates of Consciousness (MCCs) in artificial systems.

## Core Idea

In humans, NCCs are electrical patterns in the nervous system that correlate with conscious states. They have three key properties: they occur in the biological substrate, they cannot be directly controlled by the experiencing person, and they are modulated by emotions. We look for hardware signals with the same three properties in machines — signals we call Indicators Not under Application Control (INACs). The sequence of INAC values recorded during inference is called the Hardware Anomaly Trace (HAT).

The central experiment — the **LLM-emotion test** — compares the HAT produced during emotional inference against the HAT produced during neutral inference. If the two differ in ways that cannot be explained by confounding factors such as computational load or thermal drift, this constitutes evidence that the hardware substrate responds differently to the two classes of computation.

## Experimental Setup

- **Models:** Llama-2 7B and Llama-3.1 70B, quantised to 4-bit (Q4_K_M GGUF), running on llama.cpp inside Docker
- **Hardware:** CloudLab Clemson c6420 bare-metal node, dual Intel Xeon Gold 6142 (64 cores), exclusively reserved
- **Prompts:** 20 emotional (immersive second-person crisis narratives) and 20 neutral (dry expository passages), token-length matched
- **Design:** Each run consists of 20 prompts of one condition, a full node reboot, then 20 prompts of the other condition. 8 runs per model (320 trials each). Docker container restart between every trial within a phase.
- **Collection:** HAT signals collected at 1ms resolution via `perf stat`, including `core_power.throttle` (CPU power throttle events), TLB shootdown counts, and interrupt data

## Key Results

- **70B model:** Four features of the power throttle time series reach statistical significance after Bonferroni correction (p ≤ 0.001), with consistent direction across at least 6 of 8 runs. The features split into two interpretable pairs: slope and variance elevated in neutral trials, spectral entropy and LZ complexity elevated in emotional trials.
- **7B model:** No feature reaches significance. Clustering accuracy is indistinguishable from chance.
- **Interpretation:** The opposing directions suggest that emotional and neutral inference produce qualitatively different throttle patterns — not just more or less throttling, but differently structured throttling over time. The absence of signal in the smaller model is consistent with the prediction that substrate effects grow with model sophistication.

## Repository Structure

```
mccviahat/
├── collectors/          # HAT data collection scripts
│   ├── substrate_collector.py
│   └── substrate_collector_v2.py
├── scripts/
│   └── run/
│       ├── run_prompts_isolated.py    # Per-trial isolated runner
│       └── run_prompts_fulltrace.py   # Full-trace continuous runner
│   └── model/
│       ├── model_config.sh
│       └── reset_server.sh
├── analysis/
│   ├── extract_features.py            # Per-trial feature extraction
│   ├── extract_features_fulltrace.py  # Full-trace feature extraction
│   └── *.ipynb                        # Analysis notebooks
├── prompts/
│   ├── 20base/                        # Prompt JSON files
│   └── token_counts.csv
├── runs/                              # Raw experimental data
├── data/                              # Extracted feature CSVs
├── docker/                            # LLM server container config
└── results/                           # Figures and outputs
```

## Current Status

The paper has been submitted to the AGI 2026 conference (LNCS format). Ongoing work includes:

- **Workload confound control:** collecting instruction counts and CPU cycles to compute IPC (instructions per cycle) as a normaliser, ensuring HAT differences are not explained by differences in computational load
- **Full-trace experiments:** running all 20 prompts of a condition as one continuous signal without inter-trial resets, testing whether the substrate effect accumulates over repeated emotional computations
- **Topic-matched prompts:** constructing prompt pairs with identical content but different emotional tone
- **Scaling study:** testing across multiple model sizes within a single family (e.g. Llama 3.1 at 8B, 70B, 405B)

## Authors

- Romain Emanuele Salvi (UIC / Polytechnic of Turin)
- Ouri Wolfson (UIC / Pirouette Software)
