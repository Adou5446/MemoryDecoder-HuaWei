#!/usr/bin/env python3
"""
Check dstore file to see if shift was applied.
"""

import sys
import os
sys.path.insert(0, '/root/miniconda3/envs/memtest/lib/python3.9/site-packages')

from datasets import Dataset
import pickle

def main():
    print("=" * 80)
    print("Checking dstore file for shift")
    print("=" * 80)

    # Load train_vals (ground truth)
    with open('./dstore/qwen2.5-7B/wikitext/train_vals.pkl', 'rb') as f:
        train_vals = pickle.load(f)

    # Load dstore file (rank0)
    dstore_path = './dstore/qwen2.5-7B/wikitext/dstore_qwen2.5_train_3584_rank0.arrow'
    dstore = Dataset.from_file(dstore_path)

    print(f"\nDstore size: {len(dstore):,}")
    print(f"Train_vals size: {len(train_vals):,}")

    print(f"\nChecking first 20 records:")
    print(f"If shift was applied: dstore[i].vals should equal train_vals[i+1]")
    print(f"If no shift: dstore[i].vals should equal train_vals[i]")
    print()

    matches_with_shift = 0
    matches_no_shift = 0

    for i in range(min(20, len(dstore))):
        dstore_val = int(dstore[i]['vals'])
        train_val_at_i = int(train_vals[i])
        train_val_at_i_plus_1 = int(train_vals[i + 1]) if i + 1 < len(train_vals) else -1

        if dstore_val == train_val_at_i_plus_1:
            matches_with_shift += 1
            status = "WITH SHIFT ✓"
        elif dstore_val == train_val_at_i:
            matches_no_shift += 1
            status = "NO SHIFT ✓"
        else:
            status = "NEITHER ❌"

        print(f"dstore[{i}].vals = {dstore_val}, train_vals[{i}] = {train_val_at_i}, train_vals[{i+1}] = {train_val_at_i_plus_1} → {status}")

    print(f"\nMatches with shift (dstore[i] = train_vals[i+1]): {matches_with_shift}/20")
    print(f"Matches no shift (dstore[i] = train_vals[i]): {matches_no_shift}/20")

    if matches_with_shift > matches_no_shift:
        print("\n✓ Dstore HAS shift applied: dstore[i].vals = token at position i+1")
    else:
        print("\n✓ Dstore has NO shift: dstore[i].vals = token at position i")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
