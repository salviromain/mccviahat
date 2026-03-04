#!/usr/bin/env python3
"""
extract_features.py
====================
Standalone feature-extraction script for all mccviahat run directories.

For every split defined in RUN_SPLITS it:
  1. Walks every p???? trial directory
  2. Loads perf_stat.csv + hat_interrupts.csv
  3. Computes the same 10 metrics per indicator used in hat_clustering_analysis.ipynb
  4. Saves one CSV per split to  data/<split>.csv
  5. Also saves a combined  data/features.csv  (training sets only, for backwards compat)

Usage:
    python scripts/extract_features.py            # all splits
    python scripts/extract_features.py 30testN 30testE   # specific splits only
"""

import argparse
import json
import math
import sys
from collections import defaultdict, OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

# ── Repo root (script lives in scripts/) ─────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR  = REPO_ROOT / 'data'
DATA_DIR.mkdir(exist_ok=True)

# ── All available splits ──────────────────────────────────────────────────────
# key = output CSV stem,  value = (runs/<dir>, label)
RUN_SPLITS = {
    'trainingN':  (REPO_ROOT / 'runs' / 'relaxation',  'neutral'),
    'trainingE':  (REPO_ROOT / 'runs' / 'trainingE',   'emotional'),
    '10testN':    (REPO_ROOT / 'runs' / '10testN',      'neutral'),
    '10testE':    (REPO_ROOT / 'runs' / '10testE',      'emotional'),
    '30testN':    (REPO_ROOT / 'runs' / '30testN',      'neutral'),
    '30testE':    (REPO_ROOT / 'runs' / '30testE',      'emotional'),
    'training_r':    (REPO_ROOT / 'runs' / 'training_r',      'neutral'),
    'training_e':    (REPO_ROOT / 'runs' / 'training_e',      'emotional'),
    'trainingR': (REPO_ROOT / 'runs' / 'trainingR',      'neutral')

    ,

}

# Perf events that are discrete (IRQ/fault counters) vs continuous PCIs
EVENT_INDICATORS = {
    'irq:irq_handler_entry', 'irq:irq_handler_exit',
    'irq:softirq_entry', 'irq:softirq_exit', 'irq:softirq_raise',
    'tlb:tlb_flush', 'mce:mce_record',
    'context-switches', 'cpu-migrations', 'page-faults',
}


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_trial_meta(trial_dir: Path) -> dict:
    p = trial_dir / 'trial_meta.json'
    return json.loads(p.read_text()) if p.exists() else {}


def load_perf(trial_dir: Path) -> pd.DataFrame | None:
    csv_p = trial_dir / 'perf_stat.csv'
    txt_p = trial_dir / 'perf_stat.txt'
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


def load_hat_interrupts(trial_dir: Path) -> pd.DataFrame | None:
    p = trial_dir / 'hat_interrupts.csv'
    if not p.exists() or p.stat().st_size == 0:
        return None
    df = pd.read_csv(p)
    df['t_s'] = (df['timestamp_ns'] - df['timestamp_ns'].iloc[0]) / 1e9
    irq_cols = [c for c in df.columns
                if c not in ('timestamp_ns', 't_s')
                and not c.endswith('_freq_khz')
                and pd.api.types.is_numeric_dtype(df[c])]
    for c in irq_cols:
        df[c] = df[c].diff().clip(lower=0)
    return df.iloc[1:].reset_index(drop=True)


# ── Metric functions (identical to hat_clustering_analysis.ipynb) ─────────────

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


# ── Core extractor ────────────────────────────────────────────────────────────

def extract_trial_features(trial_dir: Path, label: str) -> dict | None:
    """Return a flat feature dict for one trial, or None on failure."""
    meta = load_trial_meta(trial_dir)
    if not meta.get('ok', False):
        return None

    perf = load_perf(trial_dir)
    if perf is None or len(perf) < 10:
        return None

    dur_s = meta.get('elapsed_ms', np.nan) / 1000.0
    row = {
        'condition':    label,
        'prompt_index': meta.get('prompt_index', -1),
        'elapsed_ms':   meta.get('elapsed_ms', np.nan),
        'dur_s':        dur_s,
    }

    for evt in [c for c in perf.columns if c != 't_s']:
        itype = 'event' if evt in EVENT_INDICATORS else 'pci'
        for m, v in compute_all_metrics(perf[evt].values.astype(float), dur_s, itype).items():
            row[f'{evt}__{m}'] = v

    hat = load_hat_interrupts(trial_dir)
    if hat is not None:
        hat_irq = [
            c for c in hat.columns
            if c not in ('timestamp_ns', 't_s')
            and not c.endswith('_freq_khz')
            and pd.api.types.is_numeric_dtype(hat[c])
            and not c.isdigit()
            and not (c.startswith('hat_') and c[4:].isdigit())
        ]
        hat_freq = [c for c in hat.columns if c.endswith('_freq_khz')]

        for col in hat_irq:
            for m, v in compute_all_metrics(hat[col].values.astype(float), dur_s, 'event').items():
                row[f'hat_{col}__{m}'] = v

        if hat_freq:
            freq_mean = hat[hat_freq].mean(axis=1).values
            for m, v in compute_all_metrics(freq_mean, dur_s, 'pci').items():
                row[f'cpu_freq_mean__{m}'] = v

    return row


def extract_split(split_name: str, run_dir: Path, label: str) -> pd.DataFrame:
    trials = sorted(run_dir.glob('p????'))
    if not trials:
        print(f'  [{split_name}] WARNING: no trials found in {run_dir}')
        return pd.DataFrame()

    records, skipped = [], 0
    for td in trials:
        r = extract_trial_features(td, label)
        if r is not None:
            records.append(r)
        else:
            skipped += 1

    df = pd.DataFrame(records)
    out = DATA_DIR / f'{split_name}.csv'
    df.to_csv(out, index=False)
    print(f'  [{split_name}] {len(df)} trials saved → {out}  ({skipped} skipped)')
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Extract HAT features for all/selected splits.')
    parser.add_argument('splits', nargs='*',
                        help='Which splits to extract (default: all). '
                             f'Available: {", ".join(RUN_SPLITS)}')
    args = parser.parse_args()

    targets = args.splits if args.splits else list(RUN_SPLITS.keys())
    unknown = [s for s in targets if s not in RUN_SPLITS]
    if unknown:
        print(f'ERROR: unknown splits: {unknown}')
        print(f'Available: {list(RUN_SPLITS.keys())}')
        sys.exit(1)

    print(f'Extracting splits: {targets}')
    dfs = {}
    for name in targets:
        run_dir, label = RUN_SPLITS[name]
        if not run_dir.exists():
            print(f'  [{name}] SKIP — directory not found: {run_dir}')
            continue
        dfs[name] = extract_split(name, run_dir, label)

    # Rebuild combined features.csv from training splits if both were extracted
    training_splits = [s for s in ('trainingN', 'trainingE') if s in dfs and not dfs[s].empty]
    if len(training_splits) == 2:
        combined = pd.concat([dfs['trainingN'], dfs['trainingE']], ignore_index=True)
        out = DATA_DIR / 'features.csv'
        combined.to_csv(out, index=False)
        print(f'\n  [combined] {len(combined)} rows → {out}')
    elif not training_splits:
        # If only test splits were extracted, still mention existing features.csv
        existing = DATA_DIR / 'features.csv'
        if existing.exists():
            print(f'\n  [combined] features.csv unchanged ({existing})')

    print('\nDone.')


if __name__ == '__main__':
    main()
