import json
import itertools
import os
from pathlib import Path


# Input and output paths (robust, relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INFILE = PROJECT_ROOT / 'prompts' / 'neutral_test.json'
OUTFILE = Path(__file__).parent / 'Perm_test_n.json'

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

# Build concatenated objects
concat_objs = []
for idx, perm in enumerate(perms, 1):
    concat_text = '\n\n'.join(p['instructions'].strip() for p in perm)
    titles = [p['title'] for p in perm]
    concat_objs.append({
        'id': idx,
        'titles': titles,
        'instructions': concat_text
    })

# Write output
with open(OUTFILE, 'w') as f:
    json.dump(concat_objs, f, indent=2)

# Show stats
lengths = [len(obj['instructions']) for obj in concat_objs]
try:
    import tiktoken
    enc = tiktoken.get_encoding('cl100k_base')
    tokens = [len(enc.encode(obj['instructions'])) for obj in concat_objs]
    print(f"Token stats: min={min(tokens)}, max={max(tokens)}, mean={sum(tokens)/len(tokens):.1f}")
except ImportError:
    tokens = None
    print("tiktoken not installed, skipping token count.")

print(f"String length stats: min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.1f}")
print(f"Wrote {len(concat_objs)} permutations to {OUTFILE}")
