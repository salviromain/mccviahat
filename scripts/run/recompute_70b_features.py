#!/usr/bin/env python3
"""Recompute runs/70b isolated-trial features into dataNEW/70b.

Run from any directory with .mccvenv/bin/python scripts/run/recompute_70b_features.py.
Pair emotional/neutral folders by their shared suffix; use the existing trial
extractor with LZ complexity omitted, as requested. Save one CSV per paired run.
"""
import hashlib
import json
from pathlib import Path

import pandas as pd
from extract_features import REPO_ROOT, extract_trial_features


def main():
    source = REPO_ROOT / 'runs' / '70b'
    output = REPO_ROOT / 'dataNEW' / '70b'
    output.mkdir(parents=True, exist_ok=True)
    folders = sorted(p for p in source.iterdir() if p.is_dir())
    suffixes = sorted({p.name[len('emotional'):] for p in folders if p.name.startswith('emotional')})
    expected = {f'{cond}{suffix}' for suffix in suffixes for cond in ('emotional', 'neutral')}
    if {p.name for p in folders} != expected:
        raise ValueError('Unexpected or unmatched condition folders')
    manifest, combined = [], []
    for suffix in suffixes:
        run = 'run' + (suffix or '_unsuffixed')
        records = []
        for condition in ('emotional', 'neutral'):
            folder = source / f'{condition}{suffix}'
            for trial in sorted(folder.glob('p????')):
                entry = {'source': str(trial.relative_to(REPO_ROOT)), 'run': run, 'inputs': {}}
                for name in ('trial_meta.json', 'perf_stat.csv', 'perf_stat.txt', 'hat_interrupts.csv'):
                    path = trial / name
                    if path.exists():
                        with path.open('rb') as f:
                            digest = hashlib.file_digest(f, 'sha256').hexdigest()
                        entry['inputs'][name] = {'bytes': path.stat().st_size, 'sha256': digest}
                row = extract_trial_features(trial, condition, include_lz=False)
                entry['extracted'] = row is not None
                manifest.append(entry)
                if row is None:
                    print(f'SKIPPED {trial}', flush=True)
                    continue
                if row['condition'] != condition:
                    raise ValueError(f'Condition mismatch: {trial}')
                row.update(run=run, source_trial=entry['source'])
                records.append(row)
                print(f'Extracted {entry["source"]}', flush=True)
        frame = pd.DataFrame(records)
        frame.to_csv(output / f'{run}.csv', index=False)
        combined.append(frame)
        print(f'Saved {run}: {len(frame)} rows', flush=True)
    pd.concat(combined, ignore_index=True).to_csv(output / 'all_trials.csv', index=False)
    (output / 'input_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Done: {sum(len(d) for d in combined)} trials; {len(combined)} paired runs.', flush=True)


if __name__ == '__main__':
    main()
