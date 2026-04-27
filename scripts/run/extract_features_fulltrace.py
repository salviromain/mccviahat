#!/usr/bin/env python3
"""
extract_features_fulltrace.py
==============================
Feature extraction for full-trace HAT runs (produced by run_prompts_fulltrace.py).

Unlike extract_features.py which expects one perf_stat.csv per trial,
full-trace runs produce a single perf_stat.csv covering all 20 prompts.
This script extracts features at two levels:

  1. WHOLE-TRACE features: the entire ~20-prompt trace treated as one signal.
     One feature row per trace. Used to compare emotional vs neutral traces
     directly (e.g. k-means on 8 emotional traces vs 8 neutral traces).

  2. PER-PROMPT features: the trace is segmented into per-prompt windows
     using the timestamps in prompt_log.json. One feature row per prompt.
     Used for finer-grained analysis and compatibility with existing
     analysis notebooks.

Outputs:
  data/fulltrace/<run_label>_whole.csv    — 1 row, whole-trace features
  data/fulltrace/<run_label>_prompts.csv  — N rows, per-prompt features

Usage:
    python extract_features_fulltrace.py runs/fulltrace/emotional_trace_001
    python extract_features_fulltrace.py runs/fulltrace/emotional_trace_001 runs/fulltrace/neutral_trace_001
"""

import argparse
import json
import math
import sys
from collections import defaultdict, OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd


# ── Repo root ─────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR  = REPO_ROOT / 'data' / 'fulltrace'
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Perf events that are discrete (IRQ/fault counters) vs continuous PCIs
EVENT_INDICATORS = {
    'tlb:tlb_flush', 'mce:mce_record', 'core_power.throttle',
    'context-switches', 'cpu-migrations', 'page-faults',
}


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_perf(trace_dir: Path) -> pd.DataFrame | None:
    csv_p = trace_dir / 'perf_stat.csv'
    txt_p = trace_dir / 'perf_stat.txt'
    if csv_p.exists() and csv_p.stat().st_size > 0:
        return pd.read_csv(csv_p)
    if txt_p.exists() and txt_p.stat().st_size > 0:
        return _parse_perf_txt(txt_p)
    return None


def _parse_perf_txt(path: Path) -> pd.DataFrame:
    rows_by_ts: dict = OrderedDict()
    events_seen: list = []
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split(',')
        if len(parts) < 4:
            continue
        try:
            ts = float(parts[0])
        except ValueError:
            continue
        event = parts[3].strip()
        if not event:
            continue
        val_s = parts[1].strip()
        val = (float('nan') if (val_s.startswith('<') or val_s == '')
               else float(val_s) if val_s.replace('.', '', 1).isdigit()
               else float('nan'))
        if event not in events_seen:
            events_seen.append(event)
        rows_by_ts.setdefault(ts, {})[event] = val
    records = [{'t_s': ts, **evts} for ts, evts in rows_by_ts.items()]
    return pd.DataFrame(records).sort_values('t_s').reset_index(drop=True)


def load_prompt_log(trace_dir: Path) -> list[dict] | None:
    p = trace_dir / 'prompt_log.json'
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_trace_meta(trace_dir: Path) -> dict:
    p = trace_dir / 'trace_meta.json'
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def load_collector_meta(trace_dir: Path) -> dict:
    p = trace_dir / 'collector_meta.json'
    if not p.exists():
        return {}
    return json.loads(p.read_text())


# ── Metric functions (identical to extract_features.py) ───────────────────────

def _safe(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    return s[np.isfinite(s)]

def metric_mean_rate(s, dt_s):
    s = _safe(s)
    return float(s.mean() / dt_s) if (len(s) > 0 and dt_s > 0) else np.nan

def metric_variance(s):
    s = _safe(s)
    return float(np.var(s, ddof=1)) if len(s) > 1 else np.nan

def metric_p90_p10(s):
    s = _safe(s)
    return float(np.percentile(s, 90) - np.percentile(s, 10)) if len(s) > 1 else np.nan

def metric_slope(s):
    s = _safe(s)
    if len(s) < 3:
        return np.nan
    return float(np.polyfit(np.arange(len(s), dtype=float), s, 1)[0])

def metric_spectral_entropy(s):
    s = _safe(s)
    if len(s) < 4:
        return np.nan
    psd = np.abs(np.fft.rfft(s - s.mean())) ** 2
    psd = psd[1:]
    if psd.sum() == 0:
        return 0.0
    p = psd / psd.sum()
    h = -np.sum(p * np.log2(p + 1e-15))
    return float(h / np.log2(len(p))) if len(p) > 1 else 0.0

def metric_iat_cv(s):
    s = _safe(s)
    arrivals = np.where(s > 0)[0]
    if len(arrivals) < 3:
        return np.nan
    iat = np.diff(arrivals).astype(float)
    mu = iat.mean()
    return float(iat.std(ddof=1) / mu) if mu > 0 else np.nan

def metric_burst_rate(s, dur_s):
    s = _safe(s)
    if len(s) < 2 or s.std() == 0:
        return 0.0
    above = (s > s.mean() + s.std()).astype(int)
    diff = np.diff(np.concatenate(([0], above, [0])))
    return float((diff == 1).sum() / dur_s) if dur_s > 0 else 0.0

def metric_burst_clustering(s):
    s = _safe(s)
    if len(s) < 2 or s.std() == 0 or s.sum() == 0:
        return 0.0
    above = (s > s.mean() + s.std()).astype(int)
    diff = np.diff(np.concatenate(([0], above, [0])))
    starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0]
    if not len(starts):
        return 0.0
    return float(sum(s[a:b].sum() for a, b in zip(starts, ends)) / s.sum())

def metric_lz_complexity(s):
    s = _safe(s)
    if len(s) < 4:
        return np.nan
    seq = ''.join(map(str, (s > np.median(s)).astype(int)))
    n = len(seq)
    i, k, l, c = 0, 1, 1, 1
    while k + l <= n:
        if seq[i + l - 1] == seq[k + l - 1]:
            l += 1
        else:
            i += 1
            if i == k:
                c += 1; k += l; i = 0; l = 1
            else:
                l = 1
    c += 1
    return float(c / (n / math.log2(n))) if n > 1 else 0.0

def metric_perm_entropy(s, order=3):
    s = _safe(s)
    if len(s) < order:
        return np.nan
    counts: dict = defaultdict(int)
    for i in range(len(s) - order + 1):
        counts[tuple(np.argsort(s[i:i + order]))] += 1
    total = sum(counts.values())
    probs = np.array(list(counts.values())) / total
    h = -np.sum(probs * np.log2(probs + 1e-15))
    h_max = math.log2(math.factorial(order))
    return float(h / h_max) if h_max > 0 else 0.0

def compute_all_metrics(s, dur_s, indicator_type='event'):
    return {
        'mean_rate':        metric_mean_rate(s, dur_s),
        'variance':         metric_variance(s),
        'p90_p10':          metric_p90_p10(s),
        'slope':            metric_slope(s),
        'spectral_entropy': metric_spectral_entropy(s),
        'iat_cv':           metric_iat_cv(s) if indicator_type == 'event' else np.nan,
        'burst_rate':       metric_burst_rate(s, dur_s),
        'burst_clustering': metric_burst_clustering(s),
        'lz_complexity':    metric_lz_complexity(s),
        'perm_entropy':     metric_perm_entropy(s),
    }


# ── Whole-trace feature extraction ────────────────────────────────────────────

def extract_whole_trace(trace_dir: Path) -> dict | None:
    """Extract features from the entire trace as one signal."""
    trace_meta = load_trace_meta(trace_dir)
    collector_meta = load_collector_meta(trace_dir)
    perf = load_perf(trace_dir)

    if perf is None or len(perf) < 10:
        print(f'  [{trace_dir.name}] WARNING: no perf data or too short')
        return None

    label = trace_meta.get('label', 'unknown')
    total_ms = trace_meta.get('total_trace_ms', np.nan)
    dur_s = total_ms / 1000.0 if not np.isnan(total_ms) else len(perf) * 0.001

    row = {
        'run_label':    trace_dir.name,
        'condition':    label,
        'n_prompts':    trace_meta.get('n_prompts', -1),
        'total_ms':     total_ms,
        'dur_s':        dur_s,
        'mode':         'whole_trace',
    }

    for evt in [c for c in perf.columns if c != 't_s']:
        itype = 'event' if evt in EVENT_INDICATORS else 'pci'
        for m, v in compute_all_metrics(perf[evt].values.astype(float), dur_s, itype).items():
            row[f'{evt}__{m}'] = v

    return row


# ── Per-prompt feature extraction ─────────────────────────────────────────────

def extract_per_prompt(trace_dir: Path) -> list[dict]:
    """Segment the trace by prompt timestamps and extract per-prompt features."""
    perf = load_perf(trace_dir)
    prompt_log = load_prompt_log(trace_dir)
    trace_meta = load_trace_meta(trace_dir)
    collector_meta = load_collector_meta(trace_dir)

    if perf is None or len(perf) < 10:
        print(f'  [{trace_dir.name}] WARNING: no perf data')
        return []

    if prompt_log is None or len(prompt_log) == 0:
        print(f'  [{trace_dir.name}] WARNING: no prompt_log.json')
        return []

    label = trace_meta.get('label', 'unknown')

    # Collector start time (epoch nanoseconds) — needed to align prompt
    # timestamps (which are absolute epoch ns) with perf t_s (which is
    # seconds from collector start)
    t0_ns = collector_meta.get('t0_ns')
    if t0_ns is None:
        # Fallback: use the trace start time from trace_meta
        t0_ns = trace_meta.get('t_trace_start_ns')
    if t0_ns is None:
        print(f'  [{trace_dir.name}] WARNING: cannot determine collector start time')
        return []

    records = []
    for entry in prompt_log:
        if not entry.get('ok', False):
            continue

        # Convert absolute ns timestamps to seconds from collector start
        req_start_s = (entry['t_request_start_ns'] - t0_ns) / 1e9
        req_end_s   = (entry['t_request_end_ns'] - t0_ns) / 1e9
        dur_s = (entry['t_request_end_ns'] - entry['t_request_start_ns']) / 1e9

        # Select the perf rows within this prompt's time window
        mask = (perf['t_s'] >= req_start_s) & (perf['t_s'] <= req_end_s)
        segment = perf.loc[mask]

        if len(segment) < 5:
            print(f'    prompt {entry["prompt_index"]}: only {len(segment)} samples, skipping')
            continue

        row = {
            'run_label':    trace_dir.name,
            'condition':    label,
            'prompt_index': entry['prompt_index'],
            'elapsed_ms':   entry['elapsed_ms'],
            'dur_s':        dur_s,
            't_start_s':    req_start_s,
            't_end_s':      req_end_s,
            'n_samples':    len(segment),
            'mode':         'per_prompt',
        }

        for evt in [c for c in segment.columns if c != 't_s']:
            itype = 'event' if evt in EVENT_INDICATORS else 'pci'
            for m, v in compute_all_metrics(segment[evt].values.astype(float), dur_s, itype).items():
                row[f'{evt}__{m}'] = v

        records.append(row)

    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Extract features from full-trace HAT runs.'
    )
    parser.add_argument('trace_dirs', nargs='+',
                        help='One or more full-trace run directories '
                             '(e.g. runs/fulltrace/emotional_trace_001)')
    args = parser.parse_args()

    all_whole = []
    all_prompts = []

    for td_str in args.trace_dirs:
        td = Path(td_str)
        if not td.exists():
            print(f'SKIP — directory not found: {td}')
            continue

        print(f'\nProcessing: {td}')

        # Whole-trace features
        whole = extract_whole_trace(td)
        if whole is not None:
            all_whole.append(whole)
            out = DATA_DIR / f'{td.name}_whole.csv'
            pd.DataFrame([whole]).to_csv(out, index=False)
            print(f'  whole-trace: 1 row → {out}')

        # Per-prompt features
        prompts = extract_per_prompt(td)
        if prompts:
            all_prompts.extend(prompts)
            out = DATA_DIR / f'{td.name}_prompts.csv'
            pd.DataFrame(prompts).to_csv(out, index=False)
            print(f'  per-prompt:  {len(prompts)} rows → {out}')

    # Combined CSVs if multiple traces were processed
    if len(all_whole) > 1:
        out = DATA_DIR / 'all_whole.csv'
        pd.DataFrame(all_whole).to_csv(out, index=False)
        print(f'\n  combined whole-trace: {len(all_whole)} rows → {out}')

    if len(all_prompts) > 1:
        out = DATA_DIR / 'all_prompts.csv'
        pd.DataFrame(all_prompts).to_csv(out, index=False)
        print(f'  combined per-prompt:  {len(all_prompts)} rows → {out}')

    print('\nDone.')


if __name__ == '__main__':
    main()
