# Data organization

Reorganized on October 17, 2026.

**All existing raw data, processed data, results, and ZIP archives were kept.**
Files were moved, not recomputed, merged, or edited. Code changes update paths;
feature formulas and statistical methods remain unchanged.

## Where things are now

```text
data/
  raw/                 Original recordings, logs, and collection metadata
    isolated/7b/
    isolated/70b/
    fulltrace/batch_001_en/
    fulltrace/batch_002_joy/
    fulltrace/batch_003_enj/
    fulltrace/batch_004_enm/
  features/            Existing processed CSVs
    legacy/            Everything previously under data/
    recomputed/        Feature data previously under dataNEW/
  catalog/             Recording index, batch index, path mapping, and audit
  provenance/          Original fulltrace recomputation notes and manifest
  archives/            Original ZIPs and a backup of the original project files
results/               Existing analysis tables, fitted models, plots, presentations
```

`legacy` and `recomputed` identify the two previous storage groups. They do not
mean that their feature definitions are interchangeable. All existing versions,
individual CSVs, and combined CSVs remain separate.

## Old location → new location

| Old location | New location |
|---|---|
| `runs/7b` | `data/raw/isolated/7b` |
| `runs/70b` | `data/raw/isolated/70b` |
| `runs/fulltrace` | `data/raw/fulltrace/batch_001_en` |
| `runsNEW/fulltrace` | `data/raw/fulltrace/batch_002_joy` |
| `runsNEW/fulltrace_2nd_batch` | `data/raw/fulltrace/batch_003_enj` |
| `runsMECHA/fulltrace` | `data/raw/fulltrace/batch_004_enm` |
| `data/70b`, `data/clemsonc6420`, `data/longer70b`, `data/shorter7b` | Corresponding folders in `data/features/legacy/isolated/` |
| `data/fulltrace` | `data/features/legacy/fulltrace` |
| `data/fulltrace_prompts` | `data/features/legacy/fulltrace_prompts` |
| `dataNEW/70b` | `data/features/recomputed/isolated/70b` |
| `dataNEW/70b/analysis` | `results/isolated_70b_recomputed` |
| `dataNEW/fulltrace_0ld_batch` | `data/features/recomputed/fulltrace/batch_001_en` |
| `dataNEW/fulltrace_1st_batch` | `data/features/recomputed/fulltrace/batch_002_joy` |
| `dataNEW/fulltrace_2nd_batch` | `data/features/recomputed/fulltrace/batch_003_enj` |
| `dataNEW/fulltraceMECHAbatch` | `data/features/recomputed/fulltrace/batch_004_enm` |
| `dataNEW/analysis_*`, `dataNEW/batch_corrected_analysisFTNEJ` | Same folder names under `results/` |
| `output/` | `results/` |
| `dataNEW/RECOMPUTATION.md`, `dataNEW/input_manifest.json` | `data/provenance/fulltrace_recomputation/` |
| `runsMECHA.zip`, `dataNEW/fulltraceMECHAbatch.zip` | `data/archives/` |

The full mapping is in [path_mapping.csv](data/catalog/path_mapping.csv).
Original run-folder names and the files inside them were kept unchanged.

## What changed in the project files

- Scripts and notebook source paths now point to the locations above.
- Notebook names, saved outputs, execution metadata, parameters, and batch
  selections were kept. Your existing edits were preserved.
- Classification explicitly loads batches 001 and 002: exactly the same 67
  input files recorded in the saved classification metadata. It does not add
  later batches. The current NEJ notebook still selects batches 002 and 003.
- Both collection runners accept `--output-root` to select a raw-data parent.
  Their new defaults are documented in [RUNNING.md](RUNNING.md).
- The fulltrace extractor's default input and output now both refer to batch
  002, with its outputs under `data/features/recomputed/fulltrace/`.
- `hat_viz.py` had default paths to missing recordings. Its defaults now select
  existing 7B examples: `emotional1/p0007` and `neutral1/p0006`. Its plotting
  calculations and style are unchanged.
- The main README, running instructions, and Git ignore paths were updated.
  Previously tracked files remain eligible for tracking at their new paths.

The exact list of 22 edited project files is in
[changed_project_files.json](data/catalog/reorganization/changed_project_files.json).
The new catalogs, verification script, and this guide are additions.

## Finding a recording or following an old reference

[recordings.csv](data/catalog/recordings.csv) indexes **985 recordings**:
880 isolated trials and 105 full traces. It includes current and original paths,
condition, model size, and metadata labels. Fulltrace node IDs and repetitions
come from the existing folder labels; unknown isolated-trial node IDs are blank.
These fields describe provenance, not a new statistical grouping scheme.

[batches.csv](data/catalog/batches.csv) describes the four fulltrace collections:

| Batch | Contents |
|---|---|
| `batch_001_en` | 20 emotional + 20 neutral |
| `batch_002_joy` | 27 joy |
| `batch_003_enj` | 10 emotional + 10 neutral + 3 joy |
| `batch_004_enm` | 5 emotional + 5 neutral + 5 mechanical |

Old paths inside existing CSVs, logs, manifests, historical READMEs, and saved
notebook outputs were deliberately left untouched. Use the mapping above or
[original_inventory.json](data/catalog/reorganization/original_inventory.json)
to resolve them. The inventory maps every original file to its current path.

Existing irregularities were preserved: 40 older 7B trials are inside nested
condition folders, and 15 joy metadata run labels differ from their folder
names. The catalog includes the nested trials;
[metadata_label_discrepancies.csv](data/catalog/metadata_label_discrepancies.csv)
records the label differences. No original labels were corrected or guessed.

## What was verified

- All **9,334 original files** are accounted for.
- All **9,288 moved files** match their original sizes and SHA-256 checksums.
  This includes 8,970 raw files, 252 feature-folder files (248 CSVs), 58 result
  files, the original archives, and historical provenance files.
- The other 46 original files stayed in place: 22 received the documented path
  or documentation changes, and 24 remain byte-for-byte unchanged.
- All nine notebooks load the intended inputs. Eight completed full test runs.
  Test outputs went to temporary folders; existing results and notebook outputs
  were not overwritten.
- The 7B notebook has a **pre-existing `NameError: p_ms is not defined`**. The
  original backup fails at the same point. Its data loading passes and matches
  the original. The unrelated analysis error was left unchanged.
- Classification inputs match all 67 saved input hashes, and node splits match
  the saved assignments. Sample isolated/fulltrace feature calculations match
  the original code exactly. The raw plot and script CLI checks passed.
- CloudLab hardware collection was not executed locally.

Detailed check results are under [data/catalog/reorganization/](data/catalog/reorganization/).
To repeat the original-file preservation check without changing anything:

```bash
python3 scripts/verify_data_reorganization.py
```

A backup of the scripts, notebooks, and primary project documentation **as they
were immediately before this work**, including your existing edits, is in
`data/archives/pre_reorganization_project_files.zip`.

For future extraction, use a fresh output directory when you want to retain an
existing feature version. The extractors retain their existing overwrite
behavior when explicitly rerun against the same output filenames.
