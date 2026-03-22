#!/usr/bin/env python3
"""
Merge 8 rank-specific KNN Arrow files into a single file.
"""

import argparse
import glob
import os
import re
from pathlib import Path

from datasets import Dataset, concatenate_datasets
from loguru import logger


def find_rank_files(input_pattern):
    """Find all rank-specific Arrow files matching the pattern."""
    files = glob.glob(input_pattern)
    
    if not files:
        raise ValueError(f"No files found matching pattern: {input_pattern}")
    
    files.sort(key=lambda x: int(re.search(r'rank(\d+)', x).group(1)))
    
    logger.info(f"Found {len(files)} rank files:")
    for i, f in enumerate(files):
        logger.info(f"  Rank {i}: {os.path.basename(f)}")
    
    return files


def load_and_merge(files):
    """Load and concatenate multiple Arrow files."""
    datasets_list = []
    total_samples = 0
    
    for i, file_path in enumerate(files):
        logger.info(f"Loading file {i+1}/{len(files)}: {os.path.basename(file_path)}")
        
        ds = Dataset.from_file(file_path)
        datasets_list.append(ds)
        total_samples += len(ds)
        
        logger.info(f"  Loaded {len(ds):,} samples")
    
    logger.info(f"Total samples to merge: {total_samples:,}")
    logger.info("Concatenating datasets...")
    
    merged = concatenate_datasets(datasets_list)
    
    logger.info(f"Merged dataset size: {len(merged):,}")
    
    return merged


def save_arrow(dataset, output_path):
    """Save dataset to Arrow file."""
    logger.info(f"Saving merged dataset to: {output_path}")
    
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    dataset.save_to_disk(output_path)
    
    logger.info("Save completed!")


def main():
    parser = argparse.ArgumentParser(description="Merge rank-specific KNN Arrow files")
    parser.add_argument("--input_pattern", type=str, 
                        default="./dstore/qwen2.5-7B/wikitext/knn_qwen2.5_train_3584_rank*.arrow",
                        help="Glob pattern for rank-specific files")
    parser.add_argument("--output_path", type=str,
                        default="./dstore/qwen2.5-7B/wikitext/knn_qwen2.5_train_3584.arrow",
                        help="Output merged file path")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Merging KNN Rank Files")
    logger.info("=" * 60)
    logger.info(f"Input pattern: {args.input_pattern}")
    logger.info(f"Output path: {args.output_path}")
    logger.info("=" * 60)
    
    files = find_rank_files(args.input_pattern)
    
    if len(files) != 8:
        logger.warning(f"Expected 8 files, but found {len(files)}")
    
    merged = load_and_merge(files)
    
    save_arrow(merged, args.output_path)
    
    logger.info("=" * 60)
    logger.info("Merge completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
