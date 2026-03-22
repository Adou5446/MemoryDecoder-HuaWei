"""
Add labels, attention_mask, dstore_range columns to old dataset.
Old dataset was created with stride=1024, block_size=1024, no -100 masking.
Each sample has 1024 tokens, contributing 1023 dstore entries (vals[:,1:]).
dstore_range[i] = (i*1023, (i+1)*1023)
"""
import os
from datasets import load_from_disk
from loguru import logger

OLD_DATASET_PATH = "/data/processed_data/wikitext-gpt2"
OUTPUT_PATH = "/data/processed_data/wikitext-gpt2-fixed"

def main():
    logger.info(f"Loading old dataset from {OLD_DATASET_PATH}")
    ds = load_from_disk(OLD_DATASET_PATH)
    logger.info(f"Loaded: {ds}")

    for split_name in ds.keys():
        split = ds[split_name]
        n = len(split)
        logger.info(f"Processing split '{split_name}' with {n} samples")

        block_size = len(split[0]["input_ids"])
        tokens_per_sample = block_size - 1  # 1023

        # labels = input_ids (stride=block_size, trg_len=block_size, no -100 masking)
        labels_col = [list(x) for x in split["input_ids"]]

        # attention_mask = all 1s (no padding)
        attn_col = [[1] * block_size for _ in range(n)]

        # dstore_range: each sample i maps to dstore entries [i*1023, (i+1)*1023)
        dstore_range_col = [(i * tokens_per_sample, (i + 1) * tokens_per_sample) for i in range(n)]

        split = split.add_column("labels", labels_col)
        split = split.add_column("attention_mask", attn_col)
        split = split.add_column("dstore_range", dstore_range_col)
        ds[split_name] = split

        logger.info(f"  columns = {split.column_names}")
        logger.info(f"  dstore_range[0] = {split[0]['dstore_range']}, dstore_range[-1] = {split[-1]['dstore_range']}")
        logger.info(f"  total dstore entries = {n * tokens_per_sample}")

    logger.info(f"Saving to {OUTPUT_PATH}")
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    ds.save_to_disk(OUTPUT_PATH)
    logger.info("Done!")

if __name__ == "__main__":
    main()
