#!/usr/bin/env python3
"""
Interleave 8 rank-specific KNN Arrow files into correct order.

Because DistributedSampler distributes data interleaved:
- Rank0: samples 0, 8, 16...
- Rank1: samples 1, 9, 17...

We need to interleave them back: [r0[0], [r1[0], ..., [r7[0], [r0[1], [r1[1], ...
"""

import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
from datasets import Dataset, concatenate_datasets
from loguru import logger
from tqdm import tqdm


def find_rank_files(input_pattern):
    """Find all rank-specific Arrow files matching the pattern."""
    files = glob.glob(input_pattern)
    
    if not files:
        raise ValueError(f"No files found matching pattern: {input_pattern}")
    
    files.sort(key=lambda x: int(re.search(r'rank(\d+)', x).group(1)))
    
    logger.info(f"Found {len(files)} rank files:")
    for i, f in enumerate(files):
        ds = Dataset.from_file(f)
        logger.info(f"  Rank {i}: {os.path.basename(f)} - {len(ds):,} samples")
    
    return files


def interleave_files(rank_files, num_ranks=8):
    """
    Interleave rank files back to original order.
    
    rank0: [0, 8, 16...
    rank1: [1, 9, 17...
    -> interleaved: [0, 1, 2, ..., 7, 8, 9, ...]
    """
    # Load all rank datasets
    rank_datasets = []
    for f in rank_files:
        ds = Dataset.from_file(f)
        rank_datasets.append(ds)
    
    # Check lengths
    lengths = [len(ds) for ds in rank_datasets]
    total = sum(lengths)
    logger.info(f"Total samples across all ranks: {total:,}")
    logger.info(f"Lengths: {lengths}")
    
    # Get max length (for iteration)
    max_len = max(lengths)
    
    # Interleave
    logger.info("Interleaving data...")
    
    all_data = {
        'id_cnt': [],
        'token_id': [],
        'prob': [],
        'label': []
    }
    
    for i in tqdm(range(max_len), desc="Interleaving"):
        for rank_idx in range(num_ranks):
            if i < lengths[rank_idx]:
                sample = rank_datasets[rank_idx][i]
                all_data['id_cnt'].append(sample['id_cnt'])
                all_data['token_id'].append(sample['token_id'])
                all_data['prob'].append(sample['prob'])
                all_data['label'].append(sample['label'])
    
    logger.info(f"Interleaved {len(all_data['id_cnt']:,} samples")
    
    # Create dataset
    logger.info("Creating interleaved dataset...")
    result = Dataset.from_dict(all_data)
    
    logger.info(f"Result size: {len(result):,}")
    
    return result


def save_result(dataset, output_path):
    """Save dataset to disk."""
    logger.info(f"Saving to: {output_path}")
    
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    dataset.save_to_disk(output_path)
    
    logger.info("Save completed!")


def main():
    parser = argparse.ArgumentParser(description="Interleave rank-specific KNN Arrow files")
    parser.add_argument("--input_pattern", type=str, 
                        default="./dstore/qwen2.5-7B/wikitext/knn_qwen2.5_train_3584_rank*.arrow",
                        help="Glob pattern for rank-specific files")
    parser.add_argument("--output_path", type=str,
                        default="./dstore/qwen2.5-7B/wikitext/knn_qwen2.5_train_3584_fixed.arrow",
                        help="Output path")
    parser.add_argument("--num_ranks", type=int, default=8,
                        help="Number of ranks")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Interleaving KNN Rank Files")
    logger.info("=" * 60)
    logger.info(f"Input pattern: {args.input_pattern}")
    logger.info(f"Output path: {args.output_path}")
    logger.info("=" * 60)
    
    files = find_rank_files(args.input_pattern)
    
    if len(files) != args.num_ranks:
        raise ValueError(f"Expected {args.num_ranks} files, found {len(files)}")
    
    result = interleave_files(files, args.num_ranks)
    
    save_result(result, args.output_path)
    
    logger.info("=" * 60)
    logger.info("Interleave completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
