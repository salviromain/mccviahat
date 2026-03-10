#!/usr/bin/env python3
"""
mix_prompts.py

Interleave emotional and neutral prompts from independentE.json and
independentN.json into a single JSON file suitable for a randomised
mixed-condition run.

The output array preserves all original fields and adds:
    condition     : "emotional" | "neutral"   (inferred from source file)
    mixed_index   : 0-based position in the output array
    source_index  : original 0-based position within the source file

Mixing strategies (--strategy):
    interleave  (default) — E N E N … (round-robin, trimmed to shorter list)
    shuffle               — truly random order, seeded by --seed
    block                 — all E then all N (or N then E with --neutral-first)

Usage
-----
    # Default interleaved mix, written to prompts/20base/mixed_independent.json
    python scripts/run/mix_prompts.py \\
        prompts/20base/independentE.json \\
        prompts/20base/independentN.json

    # Shuffled, reproducible
    python scripts/run/mix_prompts.py \\
        prompts/20base/independentE.json \\
        prompts/20base/independentN.json \\
        --strategy shuffle --seed 42

    # Explicit output path
    python scripts/run/mix_prompts.py \\
        prompts/20base/independentE.json \\
        prompts/20base/independentN.json \\
        --output prompts/20base/mixed_independent.json
"""

import argparse
import json
import random
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
        description="Mix emotional and neutral prompt JSONs into one file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "emotional",
        type=Path,
        help="Path to the emotional prompts JSON (e.g. independentE.json).",
    )
    p.add_argument(
        "neutral",
        type=Path,
        help="Path to the neutral prompts JSON (e.g. independentN.json).",
    )
    p.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output path. Default: <emotional_dir>/mixed_independent.json",
    )
    p.add_argument(
        "--strategy",
        choices=["interleave", "shuffle", "block"],
        default="interleave",
        help="How to arrange the prompts (default: interleave).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by the 'shuffle' strategy (default: 42).",
    )
    p.add_argument(
        "--neutral-first",
        action="store_true",
        help="For 'block' strategy: put neutral prompts first.",
    )
    p.add_argument(
        "--no-trim",
        action="store_true",
        help="For 'interleave': keep all prompts from the longer list "
             "(the shorter list wraps around). Default: trim to shorter list.",
    )
    p.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the output JSON (indent=2). Default: compact.",
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def infer_condition(stem: str) -> str:
    return CONDITION_MAP.get(stem.lower(), stem)


def load_prompts(path: Path, condition: str) -> list[dict]:
    """Load a JSON array and tag each entry with condition + source_index."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        sys.exit(f"ERROR reading {path}: {exc}")
    if not isinstance(data, list):
        sys.exit(f"ERROR: {path} must be a JSON array at the top level.")

    tagged = []
    for i, obj in enumerate(data):
        entry = dict(obj)          # shallow copy to avoid mutating original
        entry["condition"]    = condition
        entry["source_index"] = i
        tagged.append(entry)
    return tagged


def mix_interleave(e_list: list, n_list: list, no_trim: bool) -> list:
    if no_trim:
        from itertools import zip_longest, chain
        pairs = zip_longest(e_list, n_list)
        return [item for pair in pairs for item in pair if item is not None]
    else:
        return [item for pair in zip(e_list, n_list) for item in pair]


def mix_shuffle(e_list: list, n_list: list, seed: int) -> list:
    combined = e_list + n_list
    rng = random.Random(seed)
    rng.shuffle(combined)
    return combined


def mix_block(e_list: list, n_list: list, neutral_first: bool) -> list:
    if neutral_first:
        return n_list + e_list
    return e_list + n_list


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    e_cond = infer_condition(args.emotional.stem)
    n_cond = infer_condition(args.neutral.stem)

    e_prompts = load_prompts(args.emotional, e_cond)
    n_prompts = load_prompts(args.neutral, n_cond)

    print(f"Loaded  {len(e_prompts):3d} {e_cond} prompts  ← {args.emotional.name}",
          file=sys.stderr)
    print(f"Loaded  {len(n_prompts):3d} {n_cond} prompts  ← {args.neutral.name}",
          file=sys.stderr)

    if args.strategy == "interleave":
        mixed = mix_interleave(e_prompts, n_prompts, args.no_trim)
    elif args.strategy == "shuffle":
        mixed = mix_shuffle(e_prompts, n_prompts, args.seed)
    else:  # block
        mixed = mix_block(e_prompts, n_prompts, args.neutral_first)

    # Stamp the final mixed_index
    for i, entry in enumerate(mixed):
        entry["mixed_index"] = i

    # Stats
    n_e = sum(1 for e in mixed if e["condition"] == e_cond)
    n_n = sum(1 for e in mixed if e["condition"] == n_cond)
    print(f"Output  {len(mixed):3d} prompts total  "
          f"({n_e} {e_cond}, {n_n} {n_cond})  strategy={args.strategy}",
          file=sys.stderr)

    # Output path
    out = args.output or (args.emotional.parent / "mixed_independent.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    indent = 2 if args.pretty else None
    out.write_text(
        json.dumps(mixed, indent=indent, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Written → {out}", file=sys.stderr)

    # Quick preview
    print("\nFirst 6 entries (condition, mixed_index, title):", file=sys.stderr)
    for entry in mixed[:6]:
        print(f"  [{entry['mixed_index']:2d}] {entry['condition']:<10s}  "
              f"{entry.get('title', '(no title)')}", file=sys.stderr)
    if len(mixed) > 6:
        print(f"  … and {len(mixed) - 6} more", file=sys.stderr)


if __name__ == "__main__":
    main()
