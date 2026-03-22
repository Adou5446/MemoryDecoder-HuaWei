#!/usr/bin/env python3
"""More accurate diagnosis."""

import glob
import os
import re
import pickle

import torch
from datasets import Dataset

def main():
    with open("./dstore/qwen2.5-7B/wikitext/train_vals.pkl", 'rb') as f:
        vals = pickle.load(f)
    
    print(f"train_vals[:24] = {vals[:24].tolist()}")
    print()
    
    # Load all KNN rank files
    knn_files = glob.glob("./dstore/qwen2.5-7B/wikitext/knn_qwen2.5_train_3584_rank*.arrow")
    knn_files.sort(key=lambda x: int(re.search(r'rank(\d+)', x).group(1)))
    
    knn_data = []
    for f in knn_files:
        ds = Dataset.from_file(f)
        knn_data.append(ds)
    
    # Let's do a proper check: compare knn_rank0[:100] labels with orig[0:100]
    print("=" * 60)
    print("knn_rank0[:100] labels vs orig[:100]:")
    print("=" * 60)
    
    mismatch = 0
    for i in range(100):
        expected = int(vals[i])
        actual = int(knn_data[0][i]['label'])
        if expected != actual:
            mismatch += 1
            if mismatch <= 10:
                print(f"  MISMATCH at {i}: expected {expected}, got {actual}")
    
    print(f"\nTotal mismatches in knn_rank0[:100]: {mismatch}/100")
    
    if mismatch == 0:
        print("  -> knn_rank0 is IN ORDER! No interleaving needed for first 100!")
    
    # Now check around batch boundary
    print("\n" + "=" * 60)
    print("Check around batch boundary (16000):")
    print("=" * 60)
    
    print("\nknn_rank0[15990:16010] labels:")
    for i in range(15990, 16010):
        print(f"  [{i}]: {knn_data[0][i]['label']}")
    
    print("\nCompare with orig (find where these labels appear consecutively):")
    
    # Find sequence
    seq = [int(knn_data[0][i]['label']) for i in range(15990, 16000)]
    print(f"Looking for sequence: {seq}")
    
    found = False
    for start in range(500):
        match = True
        for j in range(10):
            if int(vals[start + j]) != seq[j]:
                match = False
                break
        if match:
            print(f"  FOUND! orig[{start}:{start+10}] matches")
            found = True
            # Now check next 10
            print(f"\n  Now checking orig[{start+10}:{start+20}] vs knn_rank0[16000:16010]:")
            for j in range(10):
                orig_label = int(vals[start + 10 + j])
                knn_label = int(knn_data[0][16000 + j]['label'])
                match = "✓" if orig_label == knn_label else "✗"
                print(f"    orig[{start+10+j}]={orig_label}, knn_rank0[{16000+j}]={knn_label} {match}")
            break
    
    if not found:
        print("  Sequence not found in first 500")
    
    # Key insight: knn files might already be in correct order!
    # Let's do a bigger check
    print("\n" + "=" * 60)
    print("BIG CHECK: Compare first N samples from each knn rank with orig")
    print("=" * 60)
    
    N = 1000
    for r in range(8):
        mismatches = 0
        for i in range(N):
            if int(knn_data[r][i]['label']) != int(vals[i]):
                mismatches += 1
        print(f"  knn_rank{r}[:{N}] mismatches: {mismatches}/{N} ({100*mismatches/N:.1f}%)")
    
    # Wait! Maybe each knn file has the SAME data?
    print("\n" + "=" * 60)
    print("Check if knn files have the same content:")
    print("=" * 60)
    
    for r in range(1, 8):
        mismatches = 0
        for i in range(1000):
            if int(knn_data[r][i]['label']) != int(knn_data[0][i]['label']):
                mismatches += 1
        if mismatches == 0:
            print(f"  knn_rank{r} == knn_rank0 for first 1000!")
        else:
            print(f"  knn_rank{r} differs from knn_rank0 at {mismatches}/1000 positions")
    
    # Check total samples
    print("\n" + "=" * 60)
    print("Sample count check:")
    print("=" * 60)
    print(f"  train_vals total: {len(vals):,}")
    for r in range(8):
        print(f"  knn_rank{r}: {len(knn_data[r]):,}")
    
    # If all knn files have the same data and it matches orig,
    # we just need to take ONE file and use it!
    print("\n" + "=" * 60)
    print("VERDICT:")
    print("=" * 60)
    
    # Check if knn_rank0 matches orig completely (or at least first portion)
    check_size = min(len(knn_data[0]), 100000)
    mismatches = 0
    for i in range(check_size):
        if int(knn_data[0][i]['label']) != int(vals[i]):
            mismatches += 1
    
    print(f"  knn_rank0[:{check_size}] vs orig[:{check_size}]: {mismatches} mismatches")
    
    if mismatches == 0:
        print("\n  *** ALL KNN FILES HAVE IDENTICAL DATA IN CORRECT ORDER! ***")
        print("  Solution: Just use ONE knn file for training!")
        print(f"  cp ./dstore/qwen2.5-7B/wikitext/knn_qwen2.5_train_3584_rank0.arrow ./dstore/qwen2.5-7B/wikitext/knn_qwen2.5_train_3584.arrow")
    else:
        print(f"\n  There are {mismatches} mismatches. Need more investigation.")

if __name__ == "__main__":
    main()
