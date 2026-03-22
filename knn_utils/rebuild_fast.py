#!/usr/bin/env python3
"""
Fast reconstruction of KNN datastore using streaming approach.

Key insight: instead of random access, read each file once and
place batches at correct positions using the formula.

B = 16000
knn_rank[r] contains batches [r, r+8, r+16, ...] from original

So knn_rank[r].batch[N] goes to orig[(r + N*8)*B : (r + N*8)*B + B]

We'll:
1. Pre-allocate output file
2. For each knn file, read batch by batch
3. Write each batch to correct position in output
"""

import argparse
import glob
import os
import re
import pickle

import pyarrow as pa
from datasets import Dataset, concatenate_datasets
from loguru import logger
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Fast Reconstruction")
    parser.add_argument("--knn_pattern", type=str,
                        default="./dstore/qwen2.5-7B/wikitext/knn_qwen2.5_train_3584_rank*.arrow")
    parser.add_argument("--vals_path", type=str,
                        default="./dstore/qwen2.5-7B/wikitext/train_vals.pkl")
    parser.add_argument("--output_path", type=str,
                        default="./dstore/qwen2.5-7B/wikitext/knn_qwen2.5_train_3584.arrow")
    parser.add_argument("--batch_size", type=int, default=16000)
    parser.add_argument("--num_ranks", type=int, default=8)
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Fast Reconstruction using Batch Placement")
    logger.info("=" * 60)
    
    B = args.batch_size
    
    # Load vals
    with open(args.vals_path, 'rb') as f:
        import torch
        vals = pickle.load(f)
    total_samples = len(vals)
    logger.info(f"Total samples: {total_samples:,}")
    
    # Calculate total batches needed
    total_batches = (total_samples + B - 1) // B
    logger.info(f"Total batches needed: {total_batches}")
    
    # Load KNN files info
    knn_files = sorted(glob.glob(args.knn_pattern),
                       key=lambda x: int(re.search(r'rank(\d+)', x).group(1)))
    
    # Read all batches from all files first
    # batches_by_orig_batch_idx[orig_batch_idx] = list of samples
    batches = {}
    
    for r in range(args.num_ranks):
        logger.info(f"Reading knn_rank{r}...")
        ds = Dataset.from_file(knn_files[r])
        
        # Calculate how many batches this file has
        num_batches_in_file = (len(ds) + B - 1) // B
        
        # Read batch by batch
        for N in tqdm(range(num_batches_in_file), desc=f"rank{r}"):
            start = N * B
            end = min((N + 1) * B, len(ds))
            
            # This batch goes to original position (r + N*8)*B
            orig_batch_idx = r + N * args.num_ranks
            
            # Skip if beyond what we need
            if orig_batch_idx >= total_batches:
                continue
            
            # Read the batch
            batch_data = ds.select(range(start, end))
            batches[orig_batch_idx] = batch_data
        
        del ds
    
    logger.info(f"Collected {len(batches)} batches")
    
    # Verify first few batches
    logger.info("\nVerification:")
    for orig_batch_idx in [0, 1, 8, 9, 16, 17]:
        if orig_batch_idx in batches:
            batch = batches[orig_batch_idx]
            expected_label = int(vals[orig_batch_idx * B])
            actual_label = int(batch[0]['label'])
            match = "✓" if expected_label == actual_label else "✗"
            logger.info(f"  orig_batch[{orig_batch_idx}][0]: expected={expected_label}, actual={actual_label} {match}")
    
    # Concatenate batches in order
    logger.info("\nConcatenating batches in order...")
    all_datasets = []
    for i in tqdm(range(total_batches)):
        if i in batches:
            # Cut to exact size if needed
            start = i * B
            end = min((i + 1) * B, total_samples)
            actual_size = end - start
            
            batch = batches[i]
            if len(batch) > actual_size:
                batch = batch.select(range(actual_size))
            
            all_datasets.append(batch)
            del batches[i]
    
    result = concatenate_datasets(all_datasets)
    logger.info(f"Result: {len(result):,} samples")
    
    # Final verification
    logger.info("\nFinal verification:")
    errors = 0
    for i in range(min(1000, len(result))):
        if int(result[i]['label']) != int(vals[i]):
            errors += 1
            if errors <= 5:
                logger.warning(f"  Mismatch at {i}: expected {vals[i]}, got {result[i]['label']}")
    logger.info(f"  First 1000: {errors} errors")
    
    errors = 0
    for i in range(min(1000, len(result))):
        idx = len(result) - 1000 + i
        if int(result[idx]['label']) != int(vals[idx]):
            errors += 1
    logger.info(f"  Last 1000: {errors} errors")
    
    # Save
    logger.info(f"\nSaving to {args.output_path}...")
    
    # Remove existing
    if os.path.exists(args.output_path):
        import shutil
        if os.path.isdir(args.output_path):
            shutil.rmtree(args.output_path)
        else:
            os.remove(args.output_path)
    
    result.save_to_disk(args.output_path)
    
    logger.info("=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
