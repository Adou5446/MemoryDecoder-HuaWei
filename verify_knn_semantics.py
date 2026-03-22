#!/usr/bin/env python3
"""
Verify KNN data semantics for next-token prediction.
Check what knn[i].prob and knn[i].label actually represent.
"""

import sys
import os
sys.path.insert(0, '/root/miniconda3/envs/memtest/lib/python3.9/site-packages')

from datasets import Dataset, load_from_disk
import pickle
import torch

def main():
    print("=" * 80)
    print("KNN Semantics Verification for Next-Token Prediction")
    print("=" * 80)

    # Load training dataset
    print("\n1. Loading training dataset...")
    train_ds = load_from_disk('/data/processed_data/wikitext-qwen')['train']
    print(f"   Training dataset: {len(train_ds):,} samples")

    # Load train_vals (ground truth labels for all tokens)
    print("\n2. Loading train_vals.pkl (ground truth labels)...")
    with open('./dstore/qwen2.5-7B/wikitext/train_vals.pkl', 'rb') as f:
        train_vals = pickle.load(f)
    print(f"   Train vals: {len(train_vals):,} tokens")

    # Load merged KNN file
    print("\n3. Loading merged KNN file...")
    knn_path = './dstore/qwen2.5-7B/wikitext/knn_qwen2.5_train_3584.arrow'
    if os.path.isdir(knn_path):
        knn_ds = load_from_disk(knn_path)
    else:
        knn_ds = Dataset.from_file(knn_path)
    print(f"   KNN dataset: {len(knn_ds):,} records")

    # Get first sample
    print("\n4. Analyzing first sample for next-token prediction...")
    first_sample = train_ds[0]
    dstore_range = first_sample['dstore_range']
    start, end = int(dstore_range[0]), int(dstore_range[1])

    print(f"   Sample 0: dstore_range=[{start}, {end}]")
    print(f"   This sample has {end - start + 1} tokens")

    # Get ground truth tokens for this sample
    gt_tokens = train_vals[start:start+10]  # First 10 tokens
    print(f"\n   Ground truth tokens at positions {start} to {start+9}:")
    for i, token in enumerate(gt_tokens):
        print(f"     Position {start+i}: token_id = {int(token)}")

    # Get KNN data for these positions
    print(f"\n   KNN data for positions {start} to {start+9}:")
    for i in range(10):
        knn_record = knn_ds[start + i]
        knn_label = int(knn_record['label'])
        print(f"     knn[{start+i}].label = {knn_label}")

    # Now check: what does knn[i].prob predict?
    print("\n5. Understanding knn[i].prob semantics...")
    print("   Question: Does knn[i].prob predict token at position i or i+1?")
    print()

    # Get knn[0].prob and see which token it should predict
    knn_0 = knn_ds[start]
    token_ids = knn_0['token_id']
    probs = knn_0['prob']

    # Build full distribution
    full_prob = torch.zeros(152064)  # vocab size
    for tid, p in zip(token_ids, probs):
        full_prob[tid] = float(p)

    # Get top-5 predictions from knn[0].prob
    top5_probs, top5_ids = torch.topk(full_prob, k=5)

    print(f"   knn[{start}].prob top-5 predictions:")
    for prob, tid in zip(top5_probs, top5_ids):
        print(f"     token_id {int(tid)}: prob = {prob:.4f}")

    print(f"\n   Ground truth token at position {start}: {int(gt_tokens[0])}")
    print(f"   Ground truth token at position {start+1}: {int(gt_tokens[1])}")

    # Check which one is in top predictions
    token_at_i = int(gt_tokens[0])
    token_at_i_plus_1 = int(gt_tokens[1])

    prob_at_i = float(full_prob[token_at_i])
    prob_at_i_plus_1 = float(full_prob[token_at_i_plus_1])

    print(f"\n   knn[{start}].prob[token at position {start}] = {prob_at_i:.6f}")
    print(f"   knn[{start}].prob[token at position {start+1}] = {prob_at_i_plus_1:.6f}")

    if prob_at_i > prob_at_i_plus_1:
        print(f"\n   ⚠️  knn[{start}].prob gives HIGHER probability to token at position {start}")
        print(f"       This suggests knn[i].prob predicts token AT position i (not next!)")
    else:
        print(f"\n   ✓ knn[{start}].prob gives HIGHER probability to token at position {start+1}")
        print(f"       This suggests knn[i].prob predicts token at position i+1 (next token)")

    # Check a few more positions
    print("\n6. Checking multiple positions...")
    for pos_offset in [0, 1, 2, 5, 10]:
        pos = start + pos_offset
        if pos >= len(knn_ds) - 1:
            break

        knn_record = knn_ds[pos]
        token_ids = knn_record['token_id']
        probs = knn_record['prob']

        full_prob = torch.zeros(152064)
        for tid, p in zip(token_ids, probs):
            full_prob[tid] = float(p)

        token_at_pos = int(train_vals[pos])
        token_at_next = int(train_vals[pos + 1])

        prob_at_pos = float(full_prob[token_at_pos])
        prob_at_next = float(full_prob[token_at_next])

        winner = "CURRENT" if prob_at_pos > prob_at_next else "NEXT"
        print(f"   Position {pos}: prob[current]={prob_at_pos:.6f}, prob[next]={prob_at_next:.6f} → {winner}")

    print("\n" + "=" * 80)
    print("Verification complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
