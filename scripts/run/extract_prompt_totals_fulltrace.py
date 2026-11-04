#!/usr/bin/env python3
"""Per-prompt counter totals for full-trace recordings.

For every recording, sums the perf counters (throttle, cycles, instructions)
over each prompt's time window, so per-prompt throttle per cycle can be
related to per-prompt input length. Raw data are read-only.

Time alignment: perf's t_s counts from the collector start (collector_meta
t0_ns); prompt windows come from prompt_log.json, relative to the trace start
(trace_meta t_trace_start_ns). A perf row is the count for the interval ending
at t_s, so it is assigned to the prompt whose window (start, end] contains t_s.
The first ~2 s of prompt 0 precede the collector start and are not covered;
`covered_s` records the counted time.

Usage:
    python scripts/run/extract_prompt_totals_fulltrace.py \
        --out data/features/recomputed/fulltrace_prompt_totals.csv
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "analysis/lib"))
from extract_features_fulltrace import load_perf  # noqa: E402
from fulltrace_node_day import load_recordings  # noqa: E402

COUNTERS = ["core_power.throttle", "cycles", "instructions"]
CONDITIONS = ("emotional", "neutral", "mechanical", "mechanicalprime")


def prompt_totals(run_dir: Path) -> pd.DataFrame:
    trace = json.loads((run_dir / "trace_meta.json").read_text())
    collector = json.loads((run_dir / "collector_meta.json").read_text())
    prompts = json.loads((run_dir / "prompt_log.json").read_text())
    perf = load_perf(run_dir)
    if perf is None or not set(COUNTERS) <= set(perf.columns):
        raise ValueError(f"{run_dir}: missing perf counters")
    shift_s = (collector["t0_ns"] - trace["t_trace_start_ns"]) / 1e9  # perf t_s = trace time - shift
    t = perf.t_s.to_numpy(float)
    previous = np.concatenate([[0.0], t[:-1]])
    rows = []
    for p in prompts:
        start = p["t_offset_from_trace_start_ms"] / 1000 - shift_s
        end = start + p["elapsed_ms"] / 1000
        inside = (t > start) & (t <= end)
        covered = float(np.clip(t[inside], None, end).max() - max(start, 0.0)) if inside.any() else 0.0
        row = dict(prompt_index=int(p["prompt_index"]), ok=bool(p["ok"]), elapsed_s=p["elapsed_ms"] / 1000,
                   covered_s=covered, n_samples=int(inside.sum()))
        for counter in COUNTERS:
            row[counter] = float(np.nansum(perf.loc[inside, counter].to_numpy(float)))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "data/features/recomputed/fulltrace_prompt_totals.csv")
    args = parser.parse_args()
    traces, _, _, _ = load_recordings(ROOT, "America/Chicago", conditions=CONDITIONS)
    frames = []
    for i, r in enumerate(traces.itertuples(index=False), 1):
        run_dir = ROOT / "data/raw/fulltrace" / r.batch / r.run_label
        frame = prompt_totals(run_dir)
        frames.append(frame.assign(batch=r.batch, run_label=r.run_label, condition=r.condition,
                                   node=r.node, day=r.day))
        if i % 25 == 0:
            print(f"{i}/{len(traces)} recordings", flush=True)
    out = pd.concat(frames, ignore_index=True)
    out["throttle_per_cycle"] = out["core_power.throttle"] / out["cycles"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} prompt rows from {len(traces)} recordings to {args.out}")


if __name__ == "__main__":
    main()
