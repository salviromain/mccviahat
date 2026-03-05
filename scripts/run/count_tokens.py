#!/usr/bin/env python3
"""
count_tokens.py

Count the number of Llama tokenizer tokens required for the `instructions`
field of every JSON object in one or more prompt files.

Usage
-----
    python scripts/count_tokens.py prompts/training_e.json
    python scripts/count_tokens.py prompts/training_e.json prompts/training_n.json
    python scripts/count_tokens.py prompts/20base/training_e.json --model meta-llama/Llama-3-8B
    python scripts/count_tokens.py prompts/*.json --no-bos

Output columns (tab-separated, printed to stdout):
    file  id  title  n_tokens

A summary (min/max/mean/total) is printed at the end for each file.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Count Llama tokenizer tokens for each `instructions` field.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="One or more JSON files (arrays of objects with an 'instructions' key).",
    )
    p.add_argument(
        "--model",
        default="meta-llama/Llama-2-7b-hf",
        help="HuggingFace model ID (or local path) whose tokenizer to use. "
             "Default: meta-llama/Llama-2-7b-hf",
    )
    p.add_argument(
        "--no-bos",
        dest="add_bos",
        action="store_false",
        default=True,
        help="Do not prepend the BOS token (the tokenizer adds it by default).",
    )
    p.add_argument(
        "--tsv",
        action="store_true",
        help="Print results as plain TSV (no header, no summary).",
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Core
# ──────────────────────────────────────────────────────────────────────────────

def load_tokenizer(model_id: str):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        sys.exit(
            "ERROR: `transformers` is not installed.\n"
            "  pip install transformers"
        )
    print(f"Loading tokenizer: {model_id} …", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(model_id)
    print(f"  → {tok.__class__.__name__}", file=sys.stderr)
    return tok


def count_tokens_in_file(path: Path, tokenizer, add_bos: bool) -> list[dict]:
    """Return a list of dicts {file, id, title, n_tokens} for every entry."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        sys.exit(f"ERROR reading {path}: {exc}")

    if not isinstance(data, list):
        sys.exit(f"ERROR: {path} is not a JSON array at the top level.")

    rows = []
    for i, obj in enumerate(data):
        if "instructions" not in obj:
            print(
                f"  WARN: entry {i} in {path.name} has no 'instructions' field — skipped.",
                file=sys.stderr,
            )
            continue

        text = obj["instructions"]
        ids  = tokenizer.encode(text, add_special_tokens=add_bos)
        rows.append(
            {
                "file":     path.name,
                "id":       obj.get("id", i),
                "title":    obj.get("title", ""),
                "n_tokens": len(ids),
            }
        )
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Output helpers
# ──────────────────────────────────────────────────────────────────────────────

def print_table(rows: list[dict], tsv: bool) -> None:
    if tsv:
        for r in rows:
            print(f"{r['file']}\t{r['id']}\t{r['title']}\t{r['n_tokens']}")
        return

    # Pretty-print with aligned columns
    col_widths = {
        "file":     max(len(r["file"])  for r in rows),
        "id":       max(len(str(r["id"])) for r in rows),
        "title":    max(len(r["title"]) for r in rows),
        "n_tokens": len("n_tokens"),
    }
    col_widths = {k: max(v, len(k)) for k, v in col_widths.items()}
    sep = "  "

    header = (
        f"{'file':<{col_widths['file']}}{sep}"
        f"{'id':>{col_widths['id']}}{sep}"
        f"{'n_tokens':>{col_widths['n_tokens']}}{sep}"
        f"{'title':<{col_widths['title']}}"
    )
    print(header)
    print("─" * len(header))
    for r in rows:
        print(
            f"{r['file']:<{col_widths['file']}}{sep}"
            f"{str(r['id']):>{col_widths['id']}}{sep}"
            f"{r['n_tokens']:>{col_widths['n_tokens']}}{sep}"
            f"{r['title']:<{col_widths['title']}}"
        )


def print_summary(rows: list[dict], file_path: Path) -> None:
    counts = [r["n_tokens"] for r in rows]
    if not counts:
        return
    print(
        f"\n  {file_path.name}: "
        f"n={len(counts)}  "
        f"min={min(counts)}  max={max(counts)}  "
        f"mean={statistics.mean(counts):.1f}  "
        f"median={statistics.median(counts):.1f}  "
        f"total={sum(counts)}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args  = parse_args()
    tok   = load_tokenizer(args.model)

    all_rows: list[dict] = []
    per_file_rows: list[tuple[Path, list[dict]]] = []

    for path in args.files:
        if not path.exists():
            print(f"WARN: {path} not found — skipped.", file=sys.stderr)
            continue
        rows = count_tokens_in_file(path, tok, args.add_bos)
        per_file_rows.append((path, rows))
        all_rows.extend(rows)

    if not all_rows:
        sys.exit("No data found.")

    print()
    print_table(all_rows, tsv=args.tsv)

    if not args.tsv:
        print("\n── Summary ────────────────────────────────────────")
        for path, rows in per_file_rows:
            print_summary(rows, path)

        if len(per_file_rows) > 1:
            grand_counts = [r["n_tokens"] for r in all_rows]
            print(
                f"\n  ALL FILES: "
                f"n={len(grand_counts)}  "
                f"min={min(grand_counts)}  max={max(grand_counts)}  "
                f"mean={statistics.mean(grand_counts):.1f}  "
                f"median={statistics.median(grand_counts):.1f}  "
                f"total={sum(grand_counts)}"
            )
        print()


if __name__ == "__main__":
    main()
