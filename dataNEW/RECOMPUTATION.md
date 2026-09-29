# Fulltrace feature recomputation

Completed: 2026-09-29T10:57:31-05:00
Branch: `newfeatures`; starting commit: `5d75a41`.

## Scope and command

Recomputed all 67 experiments: 27 from `runsNEW/fulltrace` and 40 from
`runs/fulltrace`. Outputs are directly in `dataNEW/` (created for this batch).

```bash
.mccvenv/bin/python -u scripts/run/extract_features_fulltrace.py runsNEW/fulltrace runs/fulltrace --whole-only --output-dir dataNEW > dataNEW/extraction.log 2>&1
```

Added `--whole-only` to the extractor to skip per-prompt extraction; existing
feature definitions and default behavior are unchanged. Added the command to
`RUNNING.md`. Preserved the pre-existing `.gitignore` modification. No raw inputs
or existing `data/` outputs were changed.

## Outputs and verification

- 67 individual `<run_label>_whole.csv` files, each containing one row.
- `all_whole.csv`: 67 rows and 42 columns, with unique run labels.
- Conditions: 27 joy, 20 emotional, 20 neutral; 20 prompts per experiment.
- Every discovered source folder is represented exactly once.
- Every individual CSV matches its row in the combined CSV.
- All rows have mode `whole_trace`; no per-prompt CSVs were generated.
- Extraction exited successfully, with no warnings; no infinite numeric values.
- `input_manifest.json` records source paths, output names, input sizes, and SHA-256 hashes.
- `extraction.log` contains the complete extractor output.

LZ complexity remains omitted by the existing fulltrace implementation.
Missing values are retained according to the existing metric definitions
(e.g. IAT CV is undefined for non-event indicators or too few arrivals):

```json
{
  "instructions__iat_cv": 67,
  "cycles__iat_cv": 67
}
```

Environment: Python 3.12.12, NumPy 2.4.2, pandas 3.0.1.
