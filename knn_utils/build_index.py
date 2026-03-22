"""
Build a FAISS index from an Arrow file containing keys and values.
Only requires the path to the datastore Arrow file.
"""

import os
import time
from loguru import logger
import argparse
import pickle
import numpy as np
import faiss
import re
import datasets
from datasets import Dataset
from tqdm import tqdm

def find_all_rank_files(dstore_path):
    """
    Find all rank-specific Arrow files in the same directory.
    
    Args:
        dstore_path: Path to a single dstore file (e.g., dstore_qwen2.5_train_3584_rank0.arrow)
        
    Returns:
        List of all rank files sorted by rank index
    """
    dstore_dir = os.path.dirname(dstore_path)
    filename = os.path.basename(dstore_path)
    
    # Extract base filename pattern (without rank suffix)
    # e.g., dstore_qwen2.5_train_3584_rank0.arrow -> dstore_qwen2.5_train_3584
    base_pattern = re.sub(r'_rank\d+\.arrow$', '', filename)
    
    # Find all files matching the pattern
    all_files = []
    for f in os.listdir(dstore_dir):
        match = re.match(rf'^{re.escape(base_pattern)}_rank(\d+)\.arrow$', f)
        if match:
            rank = int(match.group(1))
            all_files.append((rank, os.path.join(dstore_dir, f)))
    
    # Sort by rank index
    all_files.sort(key=lambda x: x[0])
    
    if not all_files:
        raise ValueError(f"No rank files found matching pattern: {base_pattern}_rank*.arrow in {dstore_dir}")
    
    logger.info(f"Found {len(all_files)} rank files: {[f[1] for f in all_files]}")
    return [f[1] for f in all_files]

def load_and_concatenate_dstore(rank_files):
    """
    Load and concatenate multiple Arrow files.
    
    Args:
        rank_files: List of Arrow file paths
        
    Returns:
        Concatenated dataset
    """
    logger.info(f"Loading {len(rank_files)} rank files...")
    
    datasets_list = []
    total_samples = 0
    
    for i, file_path in enumerate(rank_files):
        logger.info(f"Loading rank file {i+1}/{len(rank_files)}: {file_path}")
        ds = Dataset.from_file(file_path)
        datasets_list.append(ds)
        total_samples += len(ds)
        logger.info(f"  Loaded {len(ds)} samples")
    
    logger.info(f"Concatenating {total_samples} total samples...")
    concatenated = datasets.concatenate_datasets(datasets_list)
    logger.info(f"Concatenated dataset size: {len(concatenated)}")
    
    return concatenated

def parse_dstore_path(dstore_path):
    """
    Parse the dstore path to extract model_type, eval_subset, and dimension.
    
    Example:
    /fs-computility/plm/shared/jqcao/projects/neuralKNN/dstore/Qwen2.5-7B/reviews/dstore_qwen2_train_3584.arrow
    /fs-computility/plm/shared/jqcao/projects/neuralKNN/dstore/Qwen2.5-7B/reviews/dstore_qwen2_train_3584_rank0.arrow
    """
    # Extract directory and filename
    dstore_dir = os.path.dirname(dstore_path)
    filename = os.path.basename(dstore_path)
    
    # Remove rank suffix if present (e.g., _rank0, _rank1, etc.)
    filename_without_rank = re.sub(r'_rank\d+\.arrow$', '.arrow', filename)
    
    # Extract dimension from filename (the number before .arrow)
    dimension_match = re.search(r'_(\d+)\.arrow$', filename_without_rank)
    if not dimension_match:
        raise ValueError(f"Could not extract dimension from filename: {filename}")
    dimension = int(dimension_match.group(1))
    
    # Extract eval_subset from filename (usually between model and dimension)
    # Format typically: dstore_model_subset_dimension.arrow or dstore_model_subset_dimension_rankX.arrow
    parts = filename_without_rank.split('_')
    if len(parts) < 3:
        raise ValueError(f"Unexpected filename format: {filename}")
    eval_subset = parts[-2]  # Second to last part (before dimension)
    
    # Get model_type from the parent directory structure
    # The parent directory of the dstore directory is typically the model name
    model_dir = os.path.basename(dstore_dir)
    parent_dir = os.path.basename(os.path.dirname(dstore_dir))
    
    # Remove model size suffix (e.g., 'qwen2.5-7B' -> 'qwen2.5')
    model_type = parent_dir.split('-')[0] if '-' in parent_dir else parent_dir
    
    return {
        "dstore_dir": dstore_dir,
        "model_type": model_type,
        "eval_subset": eval_subset,
        "dimension": dimension
    }

def get_index_path(dstore_info):
    """Generate the path for the FAISS index file."""
    index_path = os.path.join(
        dstore_info["dstore_dir"], 
        f"{dstore_info['eval_subset']}_{dstore_info['dimension']}.index"
    )
    return index_path

def select_continuous_chunks(dataset, total_sample_size, num_chunks=10, seed=42, log_every=20):
    """
    Select continuous chunks of data from non-overlapping regions of the dataset.
    
    Args:
        dataset: The dataset to sample from
        total_sample_size: Total approximate number of samples to select
        num_chunks: Number of continuous chunks to select
        seed: Random seed for reproducibility
        
    Returns:
        Numpy array of selected samples
    """
    logger.info(f"Selecting ~{total_sample_size} samples in {num_chunks} continuous chunks")
    rng = np.random.default_rng(seed)
    dataset_size = len(dataset)
    
    # Calculate samples per chunk (approximate)
    num_chunks = max(1, min(num_chunks, total_sample_size))
    samples_per_chunk = max(1, total_sample_size // num_chunks)
    
    # Dataset is too small for chunking
    if dataset_size <= total_sample_size:
        logger.warning(f"Dataset size ({dataset_size}) is smaller than requested sample size, using all data")
        return np.asarray(dataset[:]['keys'], dtype=np.float32)
    
    # Divide dataset into non-overlapping regions
    region_size = dataset_size // num_chunks
    
    all_samples = []
    total_selected = 0
    
    logger.info(f"Selecting {num_chunks} chunks with ~{samples_per_chunk} samples each")
    for i in range(num_chunks):
        # Calculate region boundaries
        region_start = i * region_size
        region_end = (i + 1) * region_size if i < num_chunks - 1 else dataset_size
        
        # Calculate valid starting range within this region
        # The starting point should allow the chunk to fit within the region
        max_start = max(region_start, region_end - samples_per_chunk - 1)
        
        # If the region is smaller than samples_per_chunk, use the whole region
        if max_start <= region_start:
            start_idx = region_start
            end_idx = region_end
        else:
            # Randomly select a starting point within valid range
            start_idx = int(rng.integers(region_start, max_start + 1))
            end_idx = min(start_idx + samples_per_chunk, region_end)
        
        chunk_size = end_idx - start_idx
        
        if i == 0 or i == num_chunks - 1 or (i + 1) % log_every == 0:
            logger.info(
                f"Chunk {i+1}/{num_chunks}: Region [{region_start}:{region_end}], "
                f"Selected [{start_idx}:{end_idx}] ({chunk_size} samples)"
            )
        # Direct contiguous slicing is much faster than dataset.select(range(...))
        chunk_data = np.asarray(dataset[start_idx:end_idx]['keys'], dtype=np.float32)
        all_samples.append(chunk_data)
        total_selected += chunk_size
        
    # Concatenate all chunks
    logger.info(f"Selected {total_selected} samples in total across {num_chunks} non-overlapping chunks")
    return np.vstack(all_samples)

def build_index(
    dstore_path,
    num_keys_to_add_at_a_time=100000,
    ncentroids=4096,
    seed=42,
    code_size=32,
    probe=8,
    num_threads=None,
    train_sample_size=1_000_000,
    train_num_chunks=1000,
    faiss_verbose=False,
):
    """
    Build a FAISS index from an Arrow file containing keys and values.
    
    Args:
        dstore_path: Path to the Arrow file containing the datastore
        num_keys_to_add_at_a_time: Number of keys to add at a time
        ncentroids: Number of centroids for IVFPQ
        seed: Random seed
        code_size: Code size for PQ
        probe: Number of probes for query
    """
    # Parse the dstore path to extract necessary information
    dstore_info = parse_dstore_path(dstore_path)
    logger.info(f"Parsed dstore path: {dstore_info}")
    
    dimension = dstore_info["dimension"]
    
    # Find all rank files and load them
    rank_files = find_all_rank_files(dstore_path)
    logger.info('Loading Dataset...')
    if len(rank_files) == 1:
        # Fast path for single-card output: avoid concatenate wrapper overhead
        dstore = Dataset.from_file(rank_files[0])
        logger.info(f"Loaded single rank file: {rank_files[0]} ({len(dstore):,} samples)")
    else:
        dstore = load_and_concatenate_dstore(rank_files)
    
    # Set format to numpy for proper array conversion
    dstore.set_format(type='numpy', columns=['keys', 'vals'])
    
    # Save dstore.vals to a separate pickle file for later use
    vals_path = os.path.join(dstore_info["dstore_dir"], f"{dstore_info['eval_subset']}_vals.pkl")
    
    # Select only vals column
    vals_dataset = dstore.select_columns(['vals'])
    vals_dataset.set_format(type='torch', columns=['vals'])
    
    # Note that since datasets version 4.0.0, we can't use direct column selecting since the implementation of lazy columns, see pr https://github.com/huggingface/datasets/pull/7614
    vals_tensor = vals_dataset[:]['vals']
    logger.info(f"Saved val tensor shape: {vals_tensor.shape}")
    with open(vals_path, 'wb') as f:
        pickle.dump(vals_tensor, f)
    
    logger.info('Building index...')
    index_name = get_index_path(dstore_info)
    logger.info(f"Index will be saved to: {index_name}")

    # Set number of threads to maximize CPU utilization
    available_cores = os.cpu_count() or 32
    if num_threads is None:
        num_threads = min(120, available_cores)
    num_threads = max(1, int(num_threads))
    
    # Set environment variables for optimal performance
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    os.environ['NUMEXPR_MAX_THREADS'] = str(num_threads)
    
    # Set FAISS threads
    faiss.omp_set_num_threads(num_threads)
    logger.info(f'Using {num_threads} FAISS threads (host cores visible: {available_cores})')
    
    # Verify thread settings
    logger.info(f"FAISS OMP threads (before index creation): {faiss.omp_get_max_threads()}")
    logger.info(f"Environment OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'not set')}")
    
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFPQ(quantizer, dimension, ncentroids, code_size, 8)
    index.nprobe = probe
    
    # Verify thread settings after index creation
    logger.info(f"FAISS OMP threads (after index creation): {faiss.omp_get_max_threads()}")
    
    # Verbose FAISS prints massive logs and can slow down long add stages.
    index.verbose = bool(faiss_verbose)
    quantizer.verbose = bool(faiss_verbose)
    
    logger.info('=' * 60)
    logger.info('Training Index (K-means Clustering)')
    logger.info('=' * 60)
    logger.info(f'Index type: IVFPQ')
    logger.info(f'Dimension: {dimension}')
    logger.info(f'Number of centroids: {ncentroids}')
    logger.info(f'Code size: {code_size}')
    
    sample_size = min(int(train_sample_size), len(dstore))
    chunk_count = min(int(train_num_chunks), sample_size)
    logger.info(f'Training data size: {sample_size:,} vectors')
    logger.info('=' * 60)
    logger.info('Starting training... (this may take several minutes)')
    logger.info('FAISS will print iteration progress below:')
    logger.info('=' * 60)
    
    train_data = select_continuous_chunks(
        dstore,
        total_sample_size=sample_size,
        num_chunks=chunk_count,
        seed=seed,
    )

    logger.info(f'End Selecting data. Start Training...')

    start = time.time()
    index.train(train_data)
    
    elapsed = time.time() - start
    logger.info('=' * 60)
    logger.info(f'Training completed in {elapsed:.2f} seconds ({elapsed/60:.1f} minutes)')
    logger.info(f'Training speed: {sample_size/elapsed:,.0f} vectors/second')
    logger.info('=' * 60)
    
    logger.info('Adding Keys...')
    start_time = time.time()
    
    # Add keys in configurable batches. Larger batches reduce Python/HF Dataset overhead.
    batch_size = int(num_keys_to_add_at_a_time)
    if batch_size <= 0:
        raise ValueError(f"num_keys_to_add_at_a_time must be > 0, got {batch_size}")

    total_added = 0
    
    for batch_idx, start in enumerate(tqdm(range(0, len(dstore), batch_size))):
        end = min(len(dstore), start + batch_size)
        # Use direct slicing instead of select() for faster Arrow data access
        to_add = np.asarray(dstore[start:end]['keys'], dtype=np.float32)
        
        # Use add() for better multi-threading support
        index.add(to_add)
        total_added += len(to_add)
        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            speed = total_added / max(elapsed, 1e-6)
            logger.info(
                f"Added {total_added:,}/{len(dstore):,} vectors "
                f"({speed:,.0f} vec/s)"
            )
    
    faiss.write_index(index, index_name)
    logger.info(f'Added {total_added} keys in {time.time() - start_time:.2f} s')
    return index_name

def main():
    parser = argparse.ArgumentParser(description="Build a FAISS index from an Arrow datastore")
    
    # Only required parameter is the dstore_path
    parser.add_argument(
        "--dstore_path", 
        type=str, 
        required=True,
        help="Path to the Arrow file containing the datastore"
    )
    
    # Optional parameters
    parser.add_argument(
        "--num_keys_to_add_at_a_time", 
        type=int, 
        default=100_000,
        help="Number of keys to add at a time"
    )
    parser.add_argument(
        "--ncentroids", 
        type=int, 
        default=4096,
        help="Number of centroids for IVFPQ"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--code_size", 
        type=int, 
        default=32,
        help="Code size for PQ"
    )
    parser.add_argument(
        "--probe", 
        type=int, 
        default=8,
        help="Number of probes for query"
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=None,
        help="Number of FAISS CPU threads (default: min(120, visible CPU cores))"
    )
    parser.add_argument(
        "--train_sample_size",
        type=int,
        default=1_000_000,
        help="Number of vectors used for IVF/PQ training"
    )
    parser.add_argument(
        "--train_num_chunks",
        type=int,
        default=1000,
        help="Number of continuous chunks used to build training sample"
    )
    parser.add_argument(
        "--faiss_verbose",
        action="store_true",
        help="Enable verbose FAISS internals (slower, large logs)"
    )
    
    args = parser.parse_args()
    
    index_path = build_index(
        dstore_path=args.dstore_path,
        num_keys_to_add_at_a_time=args.num_keys_to_add_at_a_time,
        ncentroids=args.ncentroids,
        seed=args.seed,
        code_size=args.code_size,
        probe=args.probe,
        num_threads=args.num_threads,
        train_sample_size=args.train_sample_size,
        train_num_chunks=args.train_num_chunks,
        faiss_verbose=args.faiss_verbose,
    )
    
    logger.info(f"Index saved to: {index_path}")

if __name__ == "__main__":
    main()
