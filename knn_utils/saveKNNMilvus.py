"""
Multi-process kNN search using Milvus.
Replaces saveKNNMulti.py (which used FAISS).

Each accelerate process:
  - Reads its data shard from the dstore (via accelerator.prepare(dataloader))
  - Queries the shared Milvus collection for K nearest neighbors
  - Writes results to its own rank Arrow file

Milvus server: http://10.140.158.149:8766
"""

import os
import re
import argparse
import shutil
import numpy as np
import torch
import torch.nn.functional as F
import pyarrow as pa

from pathlib import Path
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
from datasets import Dataset
import datasets
from transformers import AutoTokenizer
from loguru import logger
from pymilvus import MilvusClient


DEFAULT_MILVUS_URI = "http://10.140.158.149:8766"


# ──────────────────────────────────────────────────────────
# helpers (same as build_milvus_index.py)
# ──────────────────────────────────────────────────────────

def find_all_rank_files(dstore_path):
    dstore_dir   = os.path.dirname(dstore_path)
    filename     = os.path.basename(dstore_path)
    base_pattern = re.sub(r'_rank\d+\.arrow$', '', filename)

    all_files = []
    for f in os.listdir(dstore_dir):
        m = re.match(rf'^{re.escape(base_pattern)}_rank(\d+)\.arrow$', f)
        if m:
            all_files.append((int(m.group(1)), os.path.join(dstore_dir, f)))

    all_files.sort(key=lambda x: x[0])
    if not all_files:
        raise ValueError(f"No rank files found for {base_pattern}")
    logger.info(f"Found {len(all_files)} rank files")
    return [f[1] for f in all_files]


def load_and_concatenate_dstore(rank_files):
    ds_list = [Dataset.from_file(fp) for fp in rank_files]
    return datasets.concatenate_datasets(ds_list)


def get_milvus_output_path(output_path, output_dir=None):
    """
    Keep Milvus-generated KNN artifacts separate from the original preprocess
    outputs. By default:

        .../dstore/<model>/<subset>/<file>.arrow
            -> .../dastore_miluvs/<model>/<subset>/<file>.arrow
    """
    output_path = Path(output_path)

    if output_dir:
        out_dir = Path(output_dir)
    else:
        parts = list(output_path.parent.parts)
        if "dstore" in parts:
            parts[parts.index("dstore")] = "dastore_miluvs"
            out_dir = Path(parts[0]).joinpath(*parts[1:])
        else:
            out_dir = output_path.parent / "dastore_miluvs"

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / output_path.name


def prepare_auxiliary_files(dstore_path, output_dir):
    """
    train_memdec.py expects dstore rank files and dstore_metadata.json to live
    next to the KNN files. When we isolate Milvus KNN outputs into a separate
    directory, mirror the lightweight sidecars there and symlink the large
    dstore rank files to avoid duplicating storage.
    """
    dstore_dir = Path(dstore_path).parent.resolve()
    output_dir = Path(output_dir).resolve()

    if output_dir == dstore_dir:
        return

    rank_files = find_all_rank_files(dstore_path)
    for rank_file in rank_files:
        src = Path(rank_file).resolve()
        dst = output_dir / src.name
        if dst.exists() or dst.is_symlink():
            continue
        try:
            dst.symlink_to(src)
            logger.info(f"Linked dstore sidecar: {dst} -> {src}")
        except OSError:
            shutil.copy2(src, dst)
            logger.warning(f"Symlink failed, copied dstore file instead: {dst}")

    metadata_src = dstore_dir / "dstore_metadata.json"
    metadata_dst = output_dir / "dstore_metadata.json"
    if metadata_src.exists() and not metadata_dst.exists():
        shutil.copy2(metadata_src, metadata_dst)
        logger.info(f"Copied dstore metadata to: {metadata_dst}")


# ──────────────────────────────────────────────────────────
# main class
# ──────────────────────────────────────────────────────────

class KNNSearchMilvus:
    def __init__(
        self,
        dstore_path,
        output_path,
        model_path,
        collection_name,
        k=1024,
        knn_temp=1.0,
        nprobe=32,
        batch_size=1000,
        ignore_first=False,
        threshold=1e-10,
        milvus_uri=DEFAULT_MILVUS_URI,
        output_dir=None,
        search_sub_batch_size=64,
        log_every=10,
    ):
        self.dstore_path     = dstore_path
        self.collection_name = collection_name
        self.k               = k
        self.knn_temp        = knn_temp
        self.nprobe          = nprobe
        self.batch_size      = batch_size
        self.ignore_first    = ignore_first
        self.threshold       = threshold
        self.milvus_uri      = milvus_uri
        self.search_sub_batch_size = search_sub_batch_size
        self.log_every       = log_every

        # Accelerator
        self.accelerator   = Accelerator()
        self.device        = self.accelerator.device
        self.process_index = self.accelerator.local_process_index

        # Remap Milvus outputs into a separate artifact directory by default.
        remapped_output_path = get_milvus_output_path(output_path, output_dir)
        if self.process_index == 0:
            prepare_auxiliary_files(self.dstore_path, remapped_output_path.parent)
        self.accelerator.wait_for_everyone()

        # Per-process output path: add _rank{i} suffix
        p = Path(remapped_output_path)
        self.output_path = str(p.parent / f"{p.stem}_rank{self.process_index}{p.suffix}")
        logger.info(f"Process {self.process_index}: output → {self.output_path}")

        # Vocab size from tokenizer
        tokenizer       = AutoTokenizer.from_pretrained(model_path)
        self.vocab_size = len(tokenizer)
        logger.info(f"Process {self.process_index}: vocab_size={self.vocab_size}")

        # Milvus client (each process opens its own connection)
        self.client = MilvusClient(uri=self.milvus_uri)
        self.client.load_collection(self.collection_name)
        logger.info(f"Process {self.process_index}: connected to Milvus, "
                    f"collection='{self.collection_name}'")

        # Alignment fix: mirror saveKNNMulti.py and let each process handle
        # exactly one original dstore rank file. This preserves token order for
        # downstream train_memdec.py and avoids every process reopening all
        # datastore shards.
        all_rank_files = find_all_rank_files(self.dstore_path)
        if self.process_index >= len(all_rank_files):
            raise ValueError(
                f"Process {self.process_index} has no rank file "
                f"(only {len(all_rank_files)} rank files found)"
            )
        my_rank_file = all_rank_files[self.process_index]
        logger.info(
            f"Process {self.process_index}: loading its own rank file: {my_rank_file}"
        )
        dstore = Dataset.from_file(my_rank_file)
        logger.info(
            f"Process {self.process_index}: loaded {len(dstore):,} rows"
        )
        dstore.set_format(type='torch', columns=['keys', 'vals'])

        self.dataloader = DataLoader(
            dstore,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=1,
            prefetch_factor=4,
        )

        self._setup_arrow_writer()

    # ── Arrow writer ─────────────────────────────────────

    def _setup_arrow_writer(self):
        fields = [
            pa.field('id_cnt',   pa.int32()),
            pa.field('token_id', pa.list_(pa.int32())),
            pa.field('prob',     pa.list_(pa.float16())),
            pa.field('label',    pa.int32()),
        ]
        schema = pa.schema(fields)
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        self.arrow_file   = pa.OSFile(self.output_path, 'wb')
        self.arrow_writer = pa.ipc.new_stream(self.arrow_file, schema)

    # ── Milvus search ────────────────────────────────────

    def search_milvus(self, queries_np):
        """
        Args:
            queries_np: (batch, dim) float32 numpy
        Returns:
            dists   : (batch, k) float32  — L2 distances
            val_ids : (batch, k) int64    — token ids of neighbours
        """
        # Split into smaller sub-batches to keep Milvus response payloads
        # and Python-side deserialization manageable.
        sub_batch_size = max(1, int(self.search_sub_batch_size))
        all_dists  = []
        all_vals   = []
        fetch_k    = self.k + (1 if self.ignore_first else 0)

        for sub_start in range(0, len(queries_np), sub_batch_size):
            sub_end     = min(len(queries_np), sub_start + sub_batch_size)
            sub_queries = queries_np[sub_start:sub_end].astype(np.float32).tolist()

            results = self.client.search(
                collection_name=self.collection_name,
                data=sub_queries,
                limit=fetch_k,
                output_fields=["val"],
                search_params={"nprobe": self.nprobe},
            )

            for hits in results:
                if self.ignore_first:
                    hits = hits[1:]
                dists_i = [h['distance']       for h in hits]
                vals_i  = [h['entity']['val']   for h in hits]

                # Pad if fewer than k results returned
                while len(dists_i) < self.k:
                    dists_i.append(float('inf'))
                    vals_i.append(0)

                all_dists.append(dists_i[:self.k])
                all_vals.append(vals_i[:self.k])

        return (np.array(all_dists, dtype=np.float32),
                np.array(all_vals,  dtype=np.int64))

    # ── Probability computation ───────────────────────────

    def knns_to_probs(self, val_ids_np, neg_dists_np):
        """
        val_ids_np  : (batch, k) int64  — token ids of neighbours
        neg_dists_np: (batch, k) float32 — negative L2 distances
        Returns     : (batch, vocab_size) float32 probability distribution
        """
        # Keep step-3 post-processing on CPU. Milvus already did the ANN search;
        # these scatter/sort ops do not benefit from NPU here and are more
        # stable on CPU, matching the FAISS implementation path.
        neg_dists = torch.tensor(neg_dists_np, dtype=torch.float32)
        val_ids   = torch.tensor(val_ids_np,   dtype=torch.int64)

        probs = F.softmax(neg_dists / self.knn_temp, dim=-1)

        knn_probs = torch.zeros(
            val_ids.shape[0], self.vocab_size, dtype=torch.float32
        ).scatter_add(dim=-1, index=val_ids, src=probs)

        knn_probs = F.normalize(knn_probs, p=1, dim=-1)
        return knn_probs

    def sparsify_distribution(self, knn_probs):
        id_cnt_list, token_id_list, prob_list = [], [], []
        for b in range(knn_probs.shape[0]):
            valid_mask  = knn_probs[b] > self.threshold
            valid_ids   = torch.nonzero(valid_mask).squeeze(-1)
            valid_probs = knn_probs[b][valid_ids]

            sorted_idx  = torch.argsort(valid_probs, descending=True)
            sorted_ids  = valid_ids[sorted_idx]
            sorted_probs = valid_probs[sorted_idx]

            id_cnt_list.append(len(sorted_ids))
            token_id_list.append(sorted_ids)
            prob_list.append(sorted_probs.to(torch.float16))
        return id_cnt_list, token_id_list, prob_list

    # ── Arrow writing ─────────────────────────────────────

    def _save_step_data(self, id_cnt, token_id, prob, label):
        id_cnt_np   = np.array(id_cnt, dtype=np.int32)
        label_np    = label.cpu().numpy()
        token_id_np = [t.cpu().numpy() for t in token_id]
        prob_np     = [p.cpu().numpy() for p in prob]

        batch = pa.RecordBatch.from_arrays(
            [
                pa.array(id_cnt_np,   type=pa.int32()),
                pa.array(token_id_np, type=pa.list_(pa.int32())),
                pa.array(prob_np,     type=pa.list_(pa.float16())),
                pa.array(label_np,    type=pa.int32()),
            ],
            ['id_cnt', 'token_id', 'prob', 'label'],
        )
        self.arrow_writer.write_batch(batch)

    # ── Main loop ─────────────────────────────────────────

    def process(self):
        logger.info(f"Process {self.process_index}: starting Milvus kNN search")

        total_batches = len(self.dataloader)
        log_every = max(1, int(self.log_every))

        for batch_idx, batch in enumerate(
            tqdm(self.dataloader, desc=f"Process {self.process_index}"),
            start=1,
        ):
            keys_np = batch['keys'].to(torch.float32).numpy()   # (batch, dim)
            vals    = batch['vals'].to(torch.int32)

            dists, val_ids = self.search_milvus(keys_np)
            neg_dists      = -dists

            knn_probs              = self.knns_to_probs(val_ids, neg_dists)
            id_cnt, token_id, prob = self.sparsify_distribution(knn_probs)

            self._save_step_data(id_cnt, token_id, prob, vals)

            if batch_idx == 1 or batch_idx % log_every == 0:
                logger.info(
                    f"Process {self.process_index}: batch {batch_idx}/{total_batches} "
                    f"(rows~{batch_idx * len(keys_np):,})"
                )
                if hasattr(self.arrow_file, "flush"):
                    self.arrow_file.flush()

        self.arrow_writer.close()
        self.arrow_file.close()
        logger.info(f"Process {self.process_index}: done → {self.output_path}")


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Milvus kNN Search (multi-process)')
    parser.add_argument('--dstore_path',     type=str,   required=True)
    parser.add_argument('--output_path',     type=str,   required=True)
    parser.add_argument('--model_path',      type=str,   required=True)
    parser.add_argument('--collection_name', type=str,   required=True,
                        help='Milvus collection name (created by build_milvus_index.py)')
    parser.add_argument('--k',               type=int,   default=1024)
    parser.add_argument('--knn_temp',        type=float, default=1.0)
    parser.add_argument('--nprobe',          type=int,   default=32,
                        help='IVF search probe count (≈ FAISS nprobe)')
    parser.add_argument('--batch_size',      type=int,   default=1000,
                        help='DataLoader batch size per process')
    parser.add_argument('--ignore_first',    type=bool,  default=False,
                        help='Skip the nearest neighbour (self-match)')
    parser.add_argument('--threshold',       type=float, default=0)
    parser.add_argument('--milvus_uri',      type=str,   default=DEFAULT_MILVUS_URI)
    parser.add_argument(
        '--search_sub_batch_size',
        type=int,
        default=64,
        help='Milvus query sub-batch size inside each process batch.',
    )
    parser.add_argument(
        '--log_every',
        type=int,
        default=10,
        help='Emit a log line every N process batches.',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory for Milvus KNN artifacts. Defaults to mirroring the '
             'output path under dastore_miluvs/.',
    )
    return parser.parse_args()


def main():
    args    = parse_args()
    searcher = KNNSearchMilvus(
        dstore_path=args.dstore_path,
        output_path=args.output_path,
        model_path=args.model_path,
        collection_name=args.collection_name,
        k=args.k,
        knn_temp=args.knn_temp,
        nprobe=args.nprobe,
        batch_size=args.batch_size,
        ignore_first=args.ignore_first,
        threshold=args.threshold,
        milvus_uri=args.milvus_uri,
        output_dir=args.output_dir,
        search_sub_batch_size=args.search_sub_batch_size,
        log_every=args.log_every,
    )
    searcher.process()


if __name__ == "__main__":
    main()
