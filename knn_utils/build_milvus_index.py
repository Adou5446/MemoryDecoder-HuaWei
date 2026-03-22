"""
Build a Milvus IVF_PQ collection from Arrow dstore files.
Replaces build_index.py (which used FAISS).

Milvus server: http://10.140.158.149:8766
Collection schema:
    id     INT64  (primary, sequential 0..N-1)
    vector FLOAT_VECTOR(dim)
    val    INT64  (next-token id)
"""

import os
import re
import pickle
import argparse
import json
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from loguru import logger
from pymilvus import MilvusClient, DataType
DEFAULT_MILVUS_URI = "http://10.140.158.149:8766"


# ──────────────────────────────────────────────────────────
# helpers shared with saveKNNMilvus.py
# ──────────────────────────────────────────────────────────

def find_all_rank_files(dstore_path):
    dstore_dir = os.path.dirname(dstore_path)
    filename   = os.path.basename(dstore_path)
    base_pattern = re.sub(r'_rank\d+\.arrow$', '', filename)

    all_files = []
    for f in os.listdir(dstore_dir):
        m = re.match(rf'^{re.escape(base_pattern)}_rank(\d+)\.arrow$', f)
        if m:
            all_files.append((int(m.group(1)), os.path.join(dstore_dir, f)))

    all_files.sort(key=lambda x: x[0])
    if not all_files:
        raise ValueError(f"No rank files found for pattern {base_pattern}_rank*.arrow in {dstore_dir}")

    logger.info(f"Found {len(all_files)} rank files")
    return [f[1] for f in all_files]



def parse_dstore_path(dstore_path):
    filename = os.path.basename(dstore_path)
    filename_no_rank = re.sub(r'_rank\d+\.arrow$', '.arrow', filename)

    dim_match = re.search(r'_(\d+)\.arrow$', filename_no_rank)
    if not dim_match:
        raise ValueError(f"Cannot extract dimension from filename: {filename}")
    dimension = int(dim_match.group(1))

    parts = filename_no_rank.split('_')
    eval_subset = parts[-2]   # e.g. "train"

    return {
        "dstore_dir":  os.path.dirname(dstore_path),
        "eval_subset": eval_subset,
        "dimension":   dimension,
    }


def get_milvus_output_dir(dstore_dir, output_dir=None):
    """
    Keep Milvus-generated local artifacts separate from the original preprocess
    outputs. By default:

        .../dstore/<model>/<subset>  ->  .../dastore_miluvs/<model>/<subset>
    """
    if output_dir:
        out_dir = Path(output_dir)
    else:
        dstore_path = Path(dstore_dir)
        parts = list(dstore_path.parts)
        if "dstore" in parts:
            parts[parts.index("dstore")] = "dastore_miluvs"
            out_dir = Path(parts[0]).joinpath(*parts[1:])
        else:
            out_dir = dstore_path / "dastore_miluvs"

    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir)


def build_schema(dimension):
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="id",     datatype=DataType.INT64,        is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimension)
    schema.add_field(field_name="val",    datatype=DataType.INT64)
    return schema


def build_index_params(client, nlist, m, nbits):
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="IVF_PQ",
        metric_type="L2",
        params={"nlist": nlist, "m": m, "nbits": nbits},
    )
    return index_params


def estimate_safe_rpc_batch_size(dimension, rpc_message_limit_mb=48.0):
    """
    Keep each insert RPC comfortably below the default gRPC 64 MiB message limit.
    A float32 vector with `dimension` dims uses roughly dimension * 4 bytes before
    protobuf framing and scalar fields, so we reserve some headroom.
    """
    limit_bytes = int(float(rpc_message_limit_mb) * 1024 * 1024)
    per_row_bytes = (int(dimension) * 4) + 32
    return max(1, limit_bytes // max(per_row_bytes, 1))


def collect_rank_jobs(rank_files):
    rank_jobs = []
    global_id = 0

    for rank_idx, rank_file in enumerate(rank_files):
        ds = Dataset.from_file(rank_file)
        rank_total = len(ds)
        rank_jobs.append(
            {
                "rank_idx": rank_idx,
                "rank_file": rank_file,
                "start_id": global_id,
                "row_count": rank_total,
            }
        )
        global_id += rank_total

    return rank_jobs, global_id


def assign_rank_jobs(rank_jobs, num_workers):
    num_workers = max(1, min(int(num_workers), len(rank_jobs)))
    worker_jobs = [[] for _ in range(num_workers)]
    worker_loads = [0] * num_workers

    # Greedy assignment keeps per-worker total rows roughly balanced.
    for job in sorted(rank_jobs, key=lambda x: x["row_count"], reverse=True):
        worker_idx = min(range(num_workers), key=lambda idx: worker_loads[idx])
        worker_jobs[worker_idx].append(job)
        worker_loads[worker_idx] += job["row_count"]

    return worker_jobs, worker_loads


def insert_batch(client, collection_name, ids, keys, vals):
    data = [
        {"id": int(row_id), "vector": vector, "val": int(token_id)}
        for row_id, vector, token_id in zip(ids.tolist(), keys.tolist(), vals.tolist())
    ]
    client.insert(collection_name=collection_name, data=data)


def insert_rank_jobs_worker(
    worker_idx,
    jobs,
    collection_name,
    dimension,
    insert_batch_size,
    rpc_message_limit_mb,
    milvus_uri,
    log_every,
):
    logger.info(
        f"Insert worker {worker_idx}: starting {len(jobs)} rank file(s), "
        f"rows={sum(job['row_count'] for job in jobs):,}"
    )
    client = MilvusClient(uri=milvus_uri)
    total_inserted = 0
    rpc_batch_size = min(
        int(insert_batch_size),
        estimate_safe_rpc_batch_size(dimension, rpc_message_limit_mb),
    )
    logger.info(
        f"Insert worker {worker_idx}: rpc_batch_size={rpc_batch_size:,} "
        f"(rpc_message_limit_mb={rpc_message_limit_mb})"
    )

    for job in jobs:
        rank_idx = job["rank_idx"]
        rank_file = job["rank_file"]
        rank_total = job["row_count"]
        start_id = job["start_id"]

        logger.info(
            f"Insert worker {worker_idx}: rank{rank_idx} rows={rank_total:,} "
            f"start_id={start_id:,} file={rank_file}"
        )

        ds = Dataset.from_file(rank_file)
        ds.set_format(type="numpy", columns=["keys", "vals"])
        num_batches = math.ceil(rank_total / insert_batch_size)

        for batch_idx, batch_start in enumerate(range(0, rank_total, insert_batch_size), start=1):
            batch_end = min(rank_total, batch_start + insert_batch_size)
            batch = ds[batch_start:batch_end]
            keys = np.asarray(batch["keys"], dtype=np.float32)
            vals = np.asarray(batch["vals"], dtype=np.int64)
            ids = np.arange(start_id + batch_start, start_id + batch_end, dtype=np.int64)
            try:
                for rpc_start in range(0, len(ids), rpc_batch_size):
                    rpc_end = min(len(ids), rpc_start + rpc_batch_size)
                    insert_batch(
                        client,
                        collection_name,
                        ids[rpc_start:rpc_end],
                        keys[rpc_start:rpc_end],
                        vals[rpc_start:rpc_end],
                    )
            except Exception as e:
                raise RuntimeError(
                    f"Insert worker {worker_idx} failed on rank{rank_idx} "
                    f"outer_batch={batch_idx}/{num_batches}, "
                    f"rows={batch_start:,}:{batch_end:,}, "
                    f"rpc_batch_size={rpc_batch_size:,}: {type(e).__name__}: {e}"
                ) from None
            total_inserted += len(ids)

            if batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == num_batches:
                logger.info(
                    f"Insert worker {worker_idx}: rank{rank_idx} "
                    f"batch {batch_idx}/{num_batches} "
                    f"rows {batch_end:,}/{rank_total:,}"
                )

        del ds

    logger.info(f"Insert worker {worker_idx}: done, inserted {total_inserted:,} rows")
    return total_inserted


def save_vals_pickle(rank_jobs, output_dir, eval_subset):
    logger.info("Saving vals.pkl for compatibility...")
    vals_chunks = []
    for job in rank_jobs:
        ds = Dataset.from_file(job["rank_file"])
        ds.set_format(type="numpy", columns=["vals"])
        vals_chunks.append(np.asarray(ds[:]["vals"], dtype=np.int64))

    vals_tensor = torch.from_numpy(np.concatenate(vals_chunks, axis=0))
    vals_path = os.path.join(output_dir, f"{eval_subset}_vals.pkl")
    with open(vals_path, "wb") as f:
        pickle.dump(vals_tensor, f)
    logger.info(f"vals.pkl saved to: {vals_path}")
    return vals_path


def save_build_metadata(output_dir, metadata):
    metadata_path = os.path.join(output_dir, "milvus_build_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    logger.info(f"Build metadata saved to: {metadata_path}")
    return metadata_path


# ──────────────────────────────────────────────────────────
# main function
# ──────────────────────────────────────────────────────────

def build_milvus_index(
    dstore_path,
    collection_name,
    nlist=4096,
    m=64,
    nbits=8,
    insert_batch_size=20000,
    milvus_uri=DEFAULT_MILVUS_URI,
    output_dir=None,
    num_insert_workers=1,
    build_index_after_insert=True,
    log_every=20,
    rpc_message_limit_mb=48.0,
):
    """
    Create a Milvus IVF_PQ collection and insert all token embeddings.

    Args:
        dstore_path:       Path to any one rank Arrow file (rank0 is fine).
        collection_name:   Milvus collection name.
        nlist:             IVF number of cluster centroids.
        m:                 PQ sub-quantizers count (must divide dimension).
        nbits:             Bits per sub-quantizer index (8 is standard).
        insert_batch_size: Vectors per insert call.
        milvus_uri:        Milvus server URI.
        output_dir:        Directory for Milvus local artifacts. Defaults to a
                           sibling directory rooted at dastore_miluvs/.
        num_insert_workers:Number of parallel insert workers.
        build_index_after_insert:
                           If True, insert all rows first, then build IVF_PQ.
        log_every:         Insert progress logging interval in batches.
        rpc_message_limit_mb:
                           Estimated per-RPC payload target used to sub-batch
                           insert requests below the gRPC size limit.
    """
    dstore_info = parse_dstore_path(dstore_path)
    dimension   = dstore_info["dimension"]
    milvus_output_dir = get_milvus_output_dir(dstore_info["dstore_dir"], output_dir)

    # Validate m divides dimension
    if dimension % m != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by m ({m}). "
                         f"Choose m from: {[x for x in range(1, dimension+1) if dimension%x==0][:20]}...")

    logger.info(f"Connecting to Milvus: {milvus_uri}")
    client = MilvusClient(uri=milvus_uri)
    logger.info(f"Milvus local artifacts will be written to: {milvus_output_dir}")

    rank_files = find_all_rank_files(dstore_path)
    rank_jobs, total = collect_rank_jobs(rank_files)
    logger.info(
        f"Prepared {len(rank_jobs)} rank job(s), total rows={total:,}, "
        f"insert_batch_size={insert_batch_size:,}, num_insert_workers={num_insert_workers}, "
        f"rpc_message_limit_mb={rpc_message_limit_mb}"
    )

    # Drop existing collection
    if collection_name in client.list_collections():
        logger.info(f"Dropping existing collection '{collection_name}'")
        client.drop_collection(collection_name)

    schema = build_schema(dimension)
    index_params = build_index_params(client, nlist, m, nbits)

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=None if build_index_after_insert else index_params,
    )
    logger.info(
        f"Created collection '{collection_name}' | dim={dimension} "
        f"nlist={nlist} m={m} build_index_after_insert={build_index_after_insert}"
    )

    worker_jobs, worker_loads = assign_rank_jobs(rank_jobs, num_insert_workers)
    for worker_idx, (jobs, rows) in enumerate(zip(worker_jobs, worker_loads)):
        logger.info(
            f"Insert worker plan {worker_idx}: ranks={[job['rank_idx'] for job in jobs]} "
            f"rows={rows:,}"
        )

    if len(worker_jobs) == 1:
        inserted = insert_rank_jobs_worker(
            worker_idx=0,
            jobs=worker_jobs[0],
            collection_name=collection_name,
            dimension=dimension,
            insert_batch_size=insert_batch_size,
            rpc_message_limit_mb=rpc_message_limit_mb,
            milvus_uri=milvus_uri,
            log_every=log_every,
        )
        logger.info(f"Single insert worker finished, inserted {inserted:,} rows")
    else:
        with ProcessPoolExecutor(
            max_workers=len(worker_jobs),
            mp_context=mp.get_context("spawn"),
        ) as executor:
            futures = [
                executor.submit(
                    insert_rank_jobs_worker,
                    worker_idx,
                    jobs,
                    collection_name,
                    dimension,
                    insert_batch_size,
                    rpc_message_limit_mb,
                    milvus_uri,
                    log_every,
                )
                for worker_idx, jobs in enumerate(worker_jobs)
                if jobs
            ]
            inserted = 0
            for future in as_completed(futures):
                inserted += future.result()
        logger.info(f"Parallel insert workers finished, inserted {inserted:,} rows")

    logger.info("Flushing collection after insert...")
    client.flush(collection_name)

    if build_index_after_insert:
        logger.info("Creating IVF_PQ index after data import...")
        client.create_index(collection_name=collection_name, index_params=index_params)

    # Load collection to make it searchable.
    client.load_collection(collection_name)
    logger.info(f"Collection '{collection_name}' loaded and ready. Total: {total:,} vectors.")

    vals_path = save_vals_pickle(rank_jobs, milvus_output_dir, dstore_info["eval_subset"])
    save_build_metadata(
        milvus_output_dir,
        {
            "collection_name": collection_name,
            "dstore_path": dstore_path,
            "milvus_uri": milvus_uri,
            "dimension": dimension,
            "nlist": nlist,
            "m": m,
            "nbits": nbits,
            "insert_batch_size": insert_batch_size,
            "num_insert_workers": len(worker_jobs),
            "build_index_after_insert": build_index_after_insert,
            "rpc_message_limit_mb": rpc_message_limit_mb,
            "total_vectors": total,
            "vals_path": vals_path,
            "rank_files": rank_files,
        },
    )

    return collection_name


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build Milvus IVF_PQ index from Arrow dstore")
    parser.add_argument("--dstore_path",       type=str, required=True,
                        help="Path to rank0 Arrow dstore file")
    parser.add_argument("--collection_name",   type=str, required=True,
                        help="Milvus collection name")
    parser.add_argument("--nlist",             type=int, default=4096,
                        help="IVF centroid count (nlist * 39 <= num_vectors required)")
    parser.add_argument("--m",                 type=int, default=64,
                        help="PQ sub-quantizers, must divide dimension")
    parser.add_argument("--nbits",             type=int, default=8)
    parser.add_argument("--insert_batch_size", type=int, default=20000)
    parser.add_argument("--milvus_uri",        type=str, default=DEFAULT_MILVUS_URI)
    parser.add_argument("--num_insert_workers", type=int, default=1,
                        help="Number of parallel insert workers")
    parser.add_argument("--log_every",         type=int, default=20,
                        help="Log insert progress every N batches")
    parser.add_argument("--build_index_after_insert", action="store_true",
                        help="Insert all rows first, then build IVF_PQ index")
    parser.add_argument("--rpc_message_limit_mb", type=float, default=48.0,
                        help="Target max payload size for each insert RPC")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for Milvus local artifacts. Defaults to mirroring the "
             "original dstore path under dastore_miluvs/.",
    )
    args = parser.parse_args()

    build_milvus_index(
        dstore_path=args.dstore_path,
        collection_name=args.collection_name,
        nlist=args.nlist,
        m=args.m,
        nbits=args.nbits,
        insert_batch_size=args.insert_batch_size,
        milvus_uri=args.milvus_uri,
        output_dir=args.output_dir,
        num_insert_workers=args.num_insert_workers,
        build_index_after_insert=args.build_index_after_insert,
        log_every=args.log_every,
        rpc_message_limit_mb=args.rpc_message_limit_mb,
    )


if __name__ == "__main__":
    main()
