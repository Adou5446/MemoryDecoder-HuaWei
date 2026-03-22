import os
import re
import time
import pickle
import logging
import numpy as np
import torch
import torch.nn.functional as F
import faiss
import faiss.contrib.torch_utils
import pyarrow as pa

from pathlib import Path
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
from datasets import Dataset
import datasets
from transformers import AutoTokenizer
from loguru import logger

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
    
    # Find all files matching pattern
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

class KNNSearchMulti:
    def __init__(self, 
                 dstore_path,
                 val_path,
                 index_path,
                 output_path,
                 model_path,
                 k=1024,
                 knn_temp=1.0,
                 probe=32,
                 batch_size=32,
                 knn_gpu=True,
                 ignore_first=False,
                 threshold=1e-10):
        
        self.dstore_path = dstore_path
        self.val_path = val_path
        self.index_path = index_path
        self.output_path = output_path
        self.model_path = model_path
        self.k = k
        self.knn_temp = knn_temp
        self.probe = probe
        self.batch_size = batch_size
        self.knn_gpu = knn_gpu
        self.ignore_first = ignore_first
        
        self.threshold = threshold
        
        # Initialize accelerator
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.world_size = self.accelerator.num_processes
        self.process_index = self.accelerator.local_process_index
        
        # Modify output_path to include rank suffix for each process
        output_path_obj = Path(self.output_path)
        output_dir = output_path_obj.parent
        output_stem = output_path_obj.stem
        output_ext = output_path_obj.suffix
        self.output_path = str(output_dir / f"{output_stem}_rank{self.process_index}{output_ext}")
        logger.info(f"Process {self.process_index}: Output path: {self.output_path}")
        
        # Get vocab size from tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.vocab_size = len(tokenizer)
        logger.info(f"Vocab size: {self.vocab_size}")
        
        # Load FAISS index (each process loads it)
        self.reconstruct_index, self.index = self._load_faiss_index()

        # ALIGNMENT FIX: Each process loads ONLY its own rank file.
        # accelerator.prepare(dataloader) would interleave batches across processes
        # (process 0 gets batches [0,8,16,...], process 1 gets [1,9,17,...]),
        # which would break the train_memdec concat_start mapping that assumes
        # knn_rank{i} corresponds exactly to dstore_rank{i} tokens in order.
        all_rank_files = find_all_rank_files(self.dstore_path)
        if self.process_index >= len(all_rank_files):
            raise ValueError(
                f"Process {self.process_index} has no rank file "
                f"(only {len(all_rank_files)} rank files found)"
            )
        my_rank_file = all_rank_files[self.process_index]
        logger.info(f"Process {self.process_index}: Loading its own rank file: {my_rank_file}")
        dataset = Dataset.from_file(my_rank_file)
        logger.info(f"Process {self.process_index}: Loaded {len(dataset):,} samples")

        # Set format to torch for proper tensor conversion
        dataset.set_format(type='torch', columns=['keys', 'vals'])

        # vals must cover ALL 116M indices (FAISS returns global indices 0..N-1).
        # Keep on CPU: knns from FAISS are CPU tensors, indexing stays on CPU.
        if self.val_path is not None:
            with open(self.val_path, 'rb') as f:
                self.vals = pickle.load(f).cpu()
        else:
            # Fallback: only current rank's vals (will fail if knns exceed rank size)
            self.vals = dataset['vals'].cpu()

        # Each process runs its own DataLoader — do NOT call accelerator.prepare()
        # to avoid accelerate re-sharding our already-sharded data.
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=1,
            pin_memory=True,
            prefetch_factor=4
        )
        
        # Initialize Arrow writer for each process
        self._setup_arrow_writer()
    
    def _load_faiss_index(self):
        """Load FAISS index and optionally move to GPU"""
        logger.info(f"Process {self.process_index}: Loading FAISS index from {self.index_path}")

        # Use all available cores divided by number of processes.
        available_cores = os.cpu_count() or 144
        threads_per_process = max(1, available_cores // self.world_size)
        faiss.omp_set_num_threads(threads_per_process)
        logger.info(f"Process {self.process_index}: FAISS using {threads_per_process} threads")

        cpu_index = faiss.read_index(self.index_path, faiss.IO_FLAG_ONDISK_SAME_DIR)
        cpu_index.nprobe = self.probe

        # make_direct_map() is only needed for reconstruct() calls which we don't use.
        # Skipping it saves significant memory (116M-entry map per process).

        return cpu_index, cpu_index
    
    def _setup_arrow_writer(self):
        """Set up Arrow writer for streaming writes (each process writes to its own file)"""
        # Define schema for output Arrow file
        fields = [
            pa.field('id_cnt', pa.int32()),
            pa.field('token_id', pa.list_(pa.int32())),
            pa.field('prob', pa.list_(pa.float16())),  # Changed to float16
            pa.field('label', pa.int32())
        ]
        schema = pa.schema(fields)
        
        # Create output directory if needed
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create Arrow file and writer for this process
        self.arrow_file = pa.OSFile(self.output_path, 'wb')
        self.arrow_writer = pa.ipc.new_stream(self.arrow_file, schema)
        logger.info(f"Process {self.process_index}: Arrow writer initialized for {self.output_path}")

    def get_knns(self, queries, ignore_first=False):
        # FAISS CPU search always needs float32 CPU tensors.
        dists, knns = self.index.search(queries.cpu().to(torch.float32), self.k)
        # Keep results on CPU to avoid unnecessary NPU transfers.
        dists = dists.cpu()
        knns = knns.cpu()
        
        # If we need to ignore the first nearest neighbor
        # There seems to be a bug of faiss 1.11.0 cuvs, that the searched result isn't sorted by distance, please use faiss 1.12.0 w/o cuvs instead
        if ignore_first:
            return dists[:,1:], knns[:,1:]
        else:
            return dists, knns
    
    def knns_to_probs(self, knns, neg_dists):
        """Compute kNN probability distribution following the reference implementation"""
        # All tensors stay on CPU (FAISS outputs are CPU, vals is CPU).
        probs = torch.nn.functional.softmax(neg_dists / self.knn_temp, dim=-1).to(torch.float32)

        vals_at_knns = self.vals[knns]  # (batch, k)
        knn_probs = torch.full(size=(vals_at_knns.shape[:-1] + (self.vocab_size,)), fill_value=0.0) \
            .scatter_add(dim=-1, index=vals_at_knns.long(), src=probs)  # (batch, vocab)

        knn_probs = F.normalize(knn_probs, p=1, dim=-1)

        return knn_probs
    
    def sparsify_distribution(self, knn_probs):
        """Extract sparse representation of distribution with probs > threshold"""
        batch_size = knn_probs.shape[0]
        
        id_cnt_list = []
        token_id_list = []
        prob_list = []
        
        for b in range(batch_size):
            # Find indices where probability > threshold
            valid_mask = knn_probs[b] > self.threshold
            valid_ids = torch.nonzero(valid_mask).squeeze(-1)
            valid_probs = knn_probs[b][valid_ids]
            
            # Sort by probability (descending)
            sorted_indices = torch.argsort(valid_probs, descending=True)
            sorted_ids = valid_ids[sorted_indices]
            sorted_probs = valid_probs[sorted_indices]
            
            id_cnt_list.append(len(sorted_ids))
            token_id_list.append(sorted_ids.cpu())
            prob_list.append(sorted_probs.cpu().to(torch.float16))
        
        return id_cnt_list, token_id_list, prob_list
    
    def _save_step_data(self, id_cnt, token_id, prob, label):
        """Save data for current step using streaming Arrow format"""
        # Convert id_cnt to tensor (stay on CPU, no NPU needed for Arrow writing)
        id_cnt_tensor = torch.tensor(id_cnt)
        
        # Each process writes its own data directly (no gathering)
        batch_size = id_cnt_tensor.shape[0]
        
        # Create Arrow arrays from local data
        id_cnt_np = id_cnt_tensor.cpu().numpy()
        label_np = label.cpu().numpy()
        
        # Convert token_id and prob lists to numpy arrays
        token_id_list_np = [t.cpu().numpy() for t in token_id]
        prob_list_np = [p.cpu().numpy() for p in prob]
        
        pass  # progress logged every 500 batches in process()
        
        # Create Arrow arrays
        id_cnt_array = pa.array(id_cnt_np, type=pa.int32())
        token_id_array = pa.array(token_id_list_np, type=pa.list_(pa.int32()))
        prob_array = pa.array(prob_list_np, type=pa.list_(pa.float16()))
        label_array = pa.array(label_np, type=pa.int32())
        
        # Create batch and write
        batch = pa.RecordBatch.from_arrays(
            [id_cnt_array, token_id_array, prob_array, label_array],
            ['id_cnt', 'token_id', 'prob', 'label']
        )
        self.arrow_writer.write_batch(batch)
    
    def process(self):
        """Main processing loop"""
        logger.info(f"Process {self.process_index}: Starting kNN search and processing")

        for batch_idx, batch in enumerate(tqdm(self.dataloader, desc=f"Process {self.process_index}")):
            # Keep keys on CPU: FAISS is CPU-only, sending to NPU and back is wasteful.
            keys = batch['keys'].to(torch.float32)
            vals = batch['vals'].to(torch.int32)
            
            # Perform kNN search
            dists, knns = self.get_knns(keys, self.ignore_first)
            neg_dists = -dists
            
            # Compute probability distribution
            knn_probs = self.knns_to_probs(knns, neg_dists)
            
            # Sparsify distribution
            id_cnt, token_id, prob = self.sparsify_distribution(knn_probs)
            
            # Save step data
            self._save_step_data(id_cnt, token_id, prob, vals)

            if batch_idx % 50 == 0:
                logger.info(f"Process {self.process_index}: batch {batch_idx}/28564 done")
        
        # Close Arrow writer for each process
        self.arrow_writer.close()
        self.arrow_file.close()
        logger.info(f"Process {self.process_index}: Finished writing to {self.output_path}")

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Multi-GPU kNN Search and Distribution Processing')
    parser.add_argument('--dstore_path', type=str, required=True,
                        help='Path to input Arrow file with keys and vals')
    parser.add_argument('--val_path', type=str, required=False, default=None,
                        help='Path to input Arrow file with vals')
    parser.add_argument('--index_path', type=str, required=True,
                        help='Path to FAISS index file')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to output Arrow file')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model for tokenizer loading')
    parser.add_argument('--k', type=int, default=1024,
                        help='Number of nearest neighbors to search')
    parser.add_argument('--knn_temp', type=float, default=1.0,
                        help='Temperature for kNN probability computation')
    parser.add_argument('--threshold', type=float, default=0,
                        help='Temperature for kNN probability computation')
    parser.add_argument('--probe', type=int, default=32,
                        help='Number of probes for FAISS index')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing')
    parser.add_argument('--knn_gpu', action='store_true',
                        help='Use GPU for FAISS search')
    parser.add_argument('--ignore_first', type=bool, default=False,
                        help='whether to ignore the nearest neighbor')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # Create KNNSearchMulti instance
    knn_search = KNNSearchMulti(
        dstore_path=args.dstore_path,
        val_path=args.val_path,
        index_path=args.index_path,
        output_path=args.output_path,
        model_path=args.model_path,
        k=args.k,
        knn_temp=args.knn_temp,
        probe=args.probe,
        batch_size=args.batch_size,
        knn_gpu=args.knn_gpu,
        ignore_first=args.ignore_first,
        threshold=args.threshold
    )
    
    # Process the data
    knn_search.process()

if __name__ == "__main__":
    main()