#!/usr/bin/env python3
"""Check that all pre-reorganization files still exist at their recorded locations.

Read-only: compares size and SHA-256 with the migration inventory. Only the
explicitly listed project files may differ; moved data/results must match exactly.
Run: python3 scripts/verify_data_reorganization.py
"""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    audit = root / 'data' / 'catalog' / 'reorganization'
    inventory = json.loads((audit / 'original_inventory.json').read_text())
    allowed = set(json.loads((audit / 'changed_project_files.json').read_text()))
    unchanged, changed, errors = 0, [], []
    moved = 0
    for row in inventory:
        path = root / row['new_path']
        if not path.is_file():
            errors.append(f"Missing: {row['new_path']}")
            continue
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        matches = path.stat().st_size == row['bytes'] and digest.hexdigest() == row['sha256']
        is_move = row['old_path'] != row['new_path']
        moved += is_move
        if matches:
            unchanged += 1
        elif row['old_path'] in allowed and not is_move:
            changed.append(row['old_path'])
        else:
            errors.append(f"Content changed: {row['new_path']}")
    if set(changed) != allowed:
        errors.append('Changed project-file list differs from the inventory comparison.')
    print(f"Original files accounted for: {len(inventory)}")
    print(f"Moved files verified: {moved}")
    print(f"Files with identical contents: {unchanged}")
    print(f"Project files changed for paths/documentation: {len(changed)}")
    for error in errors:
        print(error)
    if errors:
        raise SystemExit(1)
    print('PASS: all original files are present; every moved file is byte-for-byte unchanged.')


if __name__ == '__main__':
    main()
