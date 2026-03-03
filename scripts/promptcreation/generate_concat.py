import json
import itertools
import os
from pathlib import Path


# Input and output paths (robust, relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INFILE = PROJECT_ROOT / 'prompts' / '5base'/ 'training_relax.json'
OUTFILE = Path(__file__).parent / 'Perm_training_relax.json'

# ── Tokenizer (same as token_counts.ipynb) ────────────────────────────────────
try:
    from transformers import AutoTokenizer
    _tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    def count_tokens(text: str) -> int:
        return len(_tokenizer.encode(text, add_special_tokens=False))
    print("Tokenizer loaded: meta-llama/Llama-2-7b-hf")
except Exception as e:
    _tokenizer = None
    def count_tokens(text: str) -> int:
        return None
    print(f"Tokenizer unavailable ({e}), token counts will be skipped.")

# Load prompts
with open(INFILE, 'r') as f:
    prompts = json.load(f)
print(f"Loaded {len(prompts)} prompts from {INFILE}")
if len(prompts) != 5:
    raise ValueError(f"Expected 5 prompts, got {len(prompts)}. Check the input file.")

# Get all permutations of the 5 prompts
perms = list(itertools.permutations(prompts, 5))
if len(perms) != 120:
    raise RuntimeError(f"Expected 120 permutations, got {len(perms)}. Check input prompts.")

# Build concatenated objects — report token count for first permutation immediately
concat_objs = []
for idx, perm in enumerate(perms, 1):
    concat_text = '\n\n'.join(p['instructions'].strip() for p in perm)
    titles = [p['title'] for p in perm]
    obj = {
        'id': idx,
        'titles': titles,
        'instructions': concat_text
    }
    concat_objs.append(obj)

    # Report token count right after the first permutation is built
    if idx == 1:
        n_tokens = count_tokens(concat_text)
        n_chars  = len(concat_text)
        n_words  = len(concat_text.split())
        print(f"\nFirst permutation (id=1, titles={titles}):")
        print(f"  chars={n_chars}  words={n_words}  tokens={n_tokens if n_tokens is not None else 'N/A'}")

# Write output
with open(OUTFILE, 'w') as f:
    json.dump(concat_objs, f, indent=2)

print(f"\nWrote {len(concat_objs)} permutations to {OUTFILE}")
