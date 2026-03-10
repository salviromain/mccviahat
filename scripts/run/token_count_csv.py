#!/usr/bin/env python3
"""
token_count_csv.py

For each prompt JSON file, count the Llama tokenizer tokens in the
`instructions` field and write a CSV with columns:

    condition, prompt_index, n_tokens, title

The CSV is written next to the JSON file (same directory) with the
same stem but a _token_counts.csv suffix.  A combined CSV covering
all input files is written to --output (default: token_counts.csv in
the current directory).

When multiple files are provided a combined CSV is also written to
--combined (default: <first_file_dir>/combined_token_counts.csv).
This combined file can be joined onto the feature CSV in the notebook
on (condition, prompt_index) to use n_tokens as a confounder.

Usage
-----
    # Single file — produces prompts/20base/independentE_token_counts.csv
    python scripts/run/token_count_csv.py prompts/20base/independentE.json

    # Both independent sets — also produces combined_token_counts.csv
    python scripts/run/token_count_csv.py \\
        prompts/20base/independentE.json \\
        prompts/20base/independentN.json

    # Custom output paths
    python scripts/run/token_count_csv.py \\
        prompts/20base/independentE.json \\
        prompts/20base/independentN.json \\
        --output token_counts.csv \\
        --combined data/clemsonc6420/token_counts_combined.csv

    # Custom tokenizer
    python scripts/run/token_count_csv.py prompts/20base/*.json \\
        --model meta-llama/Llama-3-8B

Output
------
The 'condition' column is inferred from the filename stem:
    independentE  →  emotional
    independentN  →  neutral
    (anything else kept as-is)

'prompt_index' is the 0-based position of the entry in the JSON array
— matching the prompt_index column used in the data CSVs.
"""

import argparse
import csv
import json
import sys
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

CONDITION_MAP = {
    "independente": "emotional",
    "independentn": "neutral",
    "training_e":   "emotional",
    "training_r":   "neutral",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Count Llama tokens per prompt and write a CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="JSON prompt files (arrays of objects with an 'instructions' key).",
    )
    p.add_argument(
        "--model",
        default="meta-llama/Llama-2-7b-hf",
        help="HuggingFace model ID or local path. Default: meta-llama/Llama-2-7b-hf",
    )
    p.add_argument(
        "--no-bos",
        dest="add_bos",
        action="store_false",
        default=True,
        help="Do not prepend the BOS token (default: BOS is added).",
    )
    p.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("token_counts.csv"),
        help="Path for the combined CSV. Default: token_counts.csv",
    )
    p.add_argument(
        "--combined", "-c",
        type=Path,
        default=None,
        help="Path for the combined CSV written alongside the data CSVs "
             "(joinable on condition+prompt_index). "
             "Default: <first_file_parent>/combined_token_counts.csv "
             "when more than one file is given.",
    )
    p.add_argument(
        "--no-per-file",
        dest="per_file",
        action="store_false",
        default=True,
        help="Skip writing individual per-file CSVs.",
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Tokeniser
# ──────────────────────────────────────────────────────────────────────────────

def load_tokenizer(model_id: str):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        sys.exit(
            "ERROR: `transformers` is not installed.\n"
            "       pip install transformers"
        )
    print(f"Loading tokenizer: {model_id} …", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(model_id)
    print(f"  → {tok.__class__.__name__}", file=sys.stderr)
    return tok


# ──────────────────────────────────────────────────────────────────────────────
# Core
# ──────────────────────────────────────────────────────────────────────────────

def infer_condition(stem: str) -> str:
    return CONDITION_MAP.get(stem.lower(), stem)


def process_file(path: Path, tokenizer, add_bos: bool) -> list[dict]:
    """Return rows [{condition, prompt_index, n_tokens, title}] for one file."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        sys.exit(f"ERROR reading {path}: {exc}")

    if not isinstance(data, list):
        sys.exit(f"ERROR: {path} is not a JSON array at the top level.")

    condition = infer_condition(path.stem)
    rows = []
    for prompt_index, obj in enumerate(data):
        if "instructions" not in obj:
            print(
                f"  WARN: entry {prompt_index} in {path.name} has no "
                "'instructions' field — skipped.",
                file=sys.stderr,
            )
            continue
        text     = obj["instructions"]
        n_tokens = len(tokenizer.encode(text, add_special_tokens=add_bos))
        rows.append(
            {
                "condition":    condition,
                "prompt_index": prompt_index,
                "n_tokens":     n_tokens,
                "title":        obj.get("title", ""),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["condition", "prompt_index",
                                                  "n_tokens", "title"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written → {path}  ({len(rows)} rows)", file=sys.stderr)


def print_summary(rows: list[dict], label: str) -> None:
    import statistics as st
    counts = [r["n_tokens"] for r in rows]
    if not counts:
        return
    print(
        f"  {label}: n={len(counts)}  "
        f"min={min(counts)}  max={max(counts)}  "
        f"mean={st.mean(counts):.1f}  "
        f"median={st.median(counts):.1f}  "
        f"total={sum(counts)}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    tok  = load_tokenizer(args.model)

    all_rows: list[dict] = []

    for path in args.files:
        if not path.exists():
            print(f"WARN: {path} not found — skipped.", file=sys.stderr)
            continue

        rows = process_file(path, tok, args.add_bos)
        all_rows.extend(rows)

        if args.per_file:
            out = path.with_name(path.stem + "_token_counts.csv")
            write_csv(out, rows)

        print_summary(rows, path.name)

    if not all_rows:
        sys.exit("No data found.")

    # Always write the flat combined output (--output)
    write_csv(args.output, all_rows)

    # When multiple files are given, also write the data-dir combined CSV
    # (joinable onto the feature CSVs on condition + prompt_index).
    existing_files = [p for p in args.files if p.exists()]
    if len(existing_files) > 1:
        combined_path = args.combined or (
            existing_files[0].parent / "combined_token_counts.csv"
        )
        write_csv(combined_path, all_rows)

    # Grand summary split by condition
    print("\n── Grand summary ──────────────────────────────────")
    from itertools import groupby
    for cond, grp in groupby(sorted(all_rows, key=lambda r: r["condition"]),
                              key=lambda r: r["condition"]):
        print_summary(list(grp), cond)
    print()


if __name__ == "__main__":
    main()
