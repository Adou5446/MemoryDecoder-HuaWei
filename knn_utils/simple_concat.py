#!/usr/bin/env python3
"""
Reorder KNN rank files back to original dstore order.

During saveKNNMulti, accelerate distributes batches across world_size processes:
  batch 0 (dstore[0..B-1])         -> rank 0
  batch 1 (dstore[B..2B-1])        -> rank 1
  ...
  batch W-1 (dstore[(W-1)B..WB-1]) -> rank W-1
  batch W   (dstore[WB..WB+B-1])   -> rank 0
  ...

Within each chunk c of size B*W, the records in dstore order are:
  rank0[c*B .. c*B+B-1], rank1[c*B .. c*B+B-1], ..., rank{W-1}[c*B .. c*B+B-1]
"""

import argparse
import glob
import os
import re
import shutil

from datasets import Dataset, concatenate_datasets
from loguru import logger
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pattern", type=str,
                        default="./dstore/qwen2.5-7B/wikitext/knn_qwen2.5_train_3584_rank*.arrow")
    parser.add_argument("--dstore_pattern", type=str,
                        default="./dstore/qwen2.5-7B/wikitext/dstore_qwen2.5_train_3584_rank*.arrow")
    parser.add_argument("--output_path", type=str,
                        default="./dstore/qwen2.5-7B/wikitext/knn_qwen2.5_train_3584.arrow")
    parser.add_argument("--batch_size", type=int, default=16000,
                        help="batch_size used in saveKNNMulti (BATCH_SIZE_KNN in save_pipeline.sh)")
    parser.add_argument("--save_every", type=int, default=50,
                        help="Flush to disk every N chunks")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Reorder KNN Rank Files to Original Dstore Order")
    logger.info("=" * 60)

    knn_files = sorted(glob.glob(args.input_pattern),
                       key=lambda x: int(re.search(r'rank(\d+)', x).group(1)))
    W = len(knn_files)
    B = args.batch_size
    chunk_size = B * W
    logger.info(f"world_size={W}, batch_size={B}, chunk_size={chunk_size:,}")

    knn_datasets = []
    for f in knn_files:
        ds = Dataset.from_file(f)
        logger.info(f"  {os.path.basename(f)}: {len(ds):,} records")
        knn_datasets.append(ds)

    dstore_files = sorted(glob.glob(args.dstore_pattern),
                          key=lambda x: int(re.search(r'rank(\d+)', x).group(1)))
    total = sum(len(Dataset.from_file(f)) for f in dstore_files)
    logger.info(f"Total dstore records: {total:,}")

    n_chunks = (total + chunk_size - 1) // chunk_size
    logger.info(f"Total chunks: {n_chunks:,} (save every {args.save_every})")

    if os.path.exists(args.output_path):
        if os.path.isdir(args.output_path):
            shutil.rmtree(args.output_path)
        else:
            os.remove(args.output_path)

    tmp_dir = args.output_path + "_tmp_shards"
    os.makedirs(tmp_dir, exist_ok=True)
    shard_paths = []
    shard_datasets = []
    written = 0

    for c in tqdm(range(n_chunks), desc="Reordering", unit="chunk"):
        j_start = c * B
        parts = []
        for r in range(W):
            if j_start >= len(knn_datasets[r]):
                break
            j_end = min(j_start + B, len(knn_datasets[r]))
            parts.append(knn_datasets[r].select(range(j_start, j_end)))

        if not parts:
            break

        chunk_ds = concatenate_datasets(parts)
        remaining = total - written
        if len(chunk_ds) > remaining:
            chunk_ds = chunk_ds.select(range(remaining))

        shard_datasets.append(chunk_ds)
        written += len(chunk_ds)

        if len(shard_datasets) >= args.save_every:
            shard_path = os.path.join(tmp_dir, f"shard_{len(shard_paths):06d}")
            concatenate_datasets(shard_datasets).save_to_disk(shard_path)
            shard_paths.append(shard_path)
            shard_datasets = []

    if shard_datasets:
        shard_path = os.path.join(tmp_dir, f"shard_{len(shard_paths):06d}")
        concatenate_datasets(shard_datasets).save_to_disk(shard_path)
        shard_paths.append(shard_path)

    logger.info(f"Merging {len(shard_paths)} shards...")
    result = concatenate_datasets([Dataset.load_from_disk(p) for p in shard_paths])
    logger.info(f"Final: {len(result):,} records")
    result.save_to_disk(args.output_path)
    shutil.rmtree(tmp_dir)
    logger.info(f"Done! {len(result):,} records -> {args.output_path}")


if __name__ == "__main__":
    main()
