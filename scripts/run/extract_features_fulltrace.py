#!/usr/bin/env python3
"""
extract_features_fulltrace.py
==============================
Feature extraction for full-trace HAT runs (produced by run_prompts_fulltrace.py).

Unlike extract_features.py which expects one perf_stat.csv per trial,
full-trace runs produce a single perf_stat.csv covering all 20 prompts.
LZ complexity is omitted because its computation is slow on long traces.
This script extracts features at two levels:

  1. WHOLE-TRACE features: the entire ~20-prompt trace treated as one signal.
     One feature row per trace. Used to compare emotional vs neutral traces
     directly (e.g. k-means on 8 emotional traces vs 8 neutral traces).

  2. PER-PROMPT features: the trace is segmented into per-prompt windows
     using the timestamps in prompt_log.json. One feature row per prompt.
     Used for finer-grained analysis and compatibility with existing
     analysis notebooks.

How to run (from the project root):
    # Extract all run folders directly inside runsNEW/fulltrace:
    .mccvenv/bin/python scripts/run/extract_features_fulltrace.py

    # Explicitly select the parent directory:
    .mccvenv/bin/python scripts/run/extract_features_fulltrace.py runsNEW/fulltrace

    # Extract just one run:
    .mccvenv/bin/python scripts/run/extract_features_fulltrace.py runsNEW/fulltrace/229_J_1

    # Save this batch in a separate output directory:
    .mccvenv/bin/python scripts/run/extract_features_fulltrace.py --output-dir data/fulltraceNEW

    # Recompute only whole-trace features from both experiment collections:
    .mccvenv/bin/python scripts/run/extract_features_fulltrace.py runsNEW/fulltrace runs/fulltrace --whole-only --output-dir dataNEW

Inputs:
    With no arguments, reads <project root>/runsNEW/fulltrace.
    You may also supply one or more run folders or parent directories.
    Each run contains perf_stat.csv (or perf_stat.txt), trace_meta.json,
    prompt_log.json, and optionally collector_meta.json. Condition labels
    come from trace_meta.json (for example, joy or neutral).

Where feature files are saved:
    Default: <project root>/data/fulltrace/ (independent of working directory).
    <run_label>_whole.csv   — one whole-trace feature row per run.
    <run_label>_prompts.csv — one row per successfully segmented prompt.
    all_whole.csv          — combined output when more than one whole row exists.
    all_prompts.csv        — combined output when more than one prompt row exists.
    Example: runsNEW/fulltrace/229_J_1 produces
      data/fulltrace/229_J_1_whole.csv
      data/fulltrace/229_J_1_prompts.csv
    --output-dir changes the destination for every CSV; relative paths are
    relative to your current working directory. The directory is created
    automatically. Existing CSVs with the same names are overwritten.
    --whole-only skips per-prompt extraction and writes only whole-trace CSVs.

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
DEFAULT_TRACE_DIR = REPO_ROOT / 'runsNEW' / 'fulltrace'

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
    return float(s.sum() / dt_s) if (len(s) > 0 and dt_s > 0) else np.nan

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

def discover_trace_dirs(paths: list[Path]) -> list[Path]:
    """Expand parent directories and deduplicate runs before extraction."""
    traces = []
    seen = set()
    for path in paths:
        if not path.is_dir():
            raise ValueError(f'Directory not found: {path}')
        def is_trace(candidate):
            return candidate.is_dir() and any(
                (candidate / name).is_file()
                for name in ('perf_stat.csv', 'perf_stat.txt')
            )
        candidates = [path] if is_trace(path) else sorted(
            child for child in path.iterdir() if is_trace(child)
        )
        if not candidates:
            raise ValueError(f'No full-trace runs found in: {path}')
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                traces.append(candidate)
    names = [trace.name for trace in traces]
    if len(names) != len(set(names)):
        raise ValueError('Trace folder names must be unique to avoid overwriting outputs.')
    return traces


def main():
    parser = argparse.ArgumentParser(
        description='Extract features from full-trace HAT runs.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('trace_dirs', nargs='*', type=Path,
                        default=[DEFAULT_TRACE_DIR],
                        help='Run folders or parent directories. Default: runsNEW/fulltrace')
    parser.add_argument('--output-dir', type=Path, default=DATA_DIR,
                        help='Output directory. Default: <project root>/data/fulltrace')
    parser.add_argument('--whole-only', action='store_true',
                        help='Extract only whole-trace features; skip per-prompt features.')
    args = parser.parse_args()
    try:
        trace_dirs = discover_trace_dirs(args.trace_dirs)
    except ValueError as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Found {len(trace_dirs)} trace directories')
    print(f'Feature CSV destination: {args.output_dir.resolve()}')

    all_whole = []
    all_prompts = []

    for td in trace_dirs:

        print(f'\nProcessing: {td}')

        # Whole-trace features
        whole = extract_whole_trace(td)
        if whole is not None:
            all_whole.append(whole)
            out = args.output_dir / f'{td.name}_whole.csv'
            pd.DataFrame([whole]).to_csv(out, index=False)
            print(f'  whole-trace: 1 row → {out}')

        # Per-prompt features
        prompts = [] if args.whole_only else extract_per_prompt(td)
        if prompts:
            all_prompts.extend(prompts)
            out = args.output_dir / f'{td.name}_prompts.csv'
            pd.DataFrame(prompts).to_csv(out, index=False)
            print(f'  per-prompt:  {len(prompts)} rows → {out}')

    # Combined CSVs if multiple traces were processed
    if len(all_whole) > 1:
        out = args.output_dir / 'all_whole.csv'
        pd.DataFrame(all_whole).to_csv(out, index=False)
        print(f'\n  combined whole-trace: {len(all_whole)} rows → {out}')

    if len(all_prompts) > 1:
        out = args.output_dir / 'all_prompts.csv'
        pd.DataFrame(all_prompts).to_csv(out, index=False)
        print(f'  combined per-prompt:  {len(all_prompts)} rows → {out}')

    print('\nDone.')


if __name__ == '__main__':
    main()
