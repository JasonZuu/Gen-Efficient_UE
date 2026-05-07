"""LLM inference for CoQA using Gemma-4-E4B via vLLM.

Runs batched inference and saves per-token top-20 logits to parquet shards.
Supports resuming from a checkpoint if interrupted.

Usage
-----
python preprocessing/infer_llm.py \\
    --model_path google/gemma-4-e4b-it \\
    --output_dir data/llm-inference \\
    --data_split train \\
    --num_responses 10 \\
    --gpu_ids 0
"""

import gc
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer, AutoConfig
from vllm import LLM, SamplingParams

os.environ["POLARS_ALLOW_FORKING_THREAD"] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dataset.load_fn import load_coqa
from preprocessing.infer_fn import LLMConfig, infer_llm_on_dataset


# ---------------------------------------------------------------------------
# Inline utilities
# ---------------------------------------------------------------------------

def _set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_log_fname(task: str, max_input_len: str, llm_name: str, pe_method: str,
                   n_response: int, data_split: str) -> str:
    """Generate standardized output filename."""
    return f"{task}_{max_input_len}_{llm_name}_{pe_method}_nres-{n_response}_set-{data_split}"


def _resolve_llm_config(model_path: str) -> LLMConfig:
    """Return LLMConfig for the given model path (Gemma-4 only)."""
    model_path_lower = str(model_path).lower()
    if "gemma-4" in model_path_lower or "gemma4" in model_path_lower:
        return LLMConfig(temperature=1.0, top_k=20)
    # Default fallback
    return LLMConfig(temperature=1.0, top_k=20)


# ---------------------------------------------------------------------------
# Shard helpers
# ---------------------------------------------------------------------------

def _expected_rows_for_shard(shard_idx: int, n_total: int, n_per_shard: int) -> int:
    start = shard_idx * n_per_shard
    end = min(start + n_per_shard, n_total)
    return end - start


def _get_shard_paths(log_dir: Path, log_fname: str, logits_base_fname: str,
                     shard_idx: int, n_shards: int) -> dict:
    suffix = f"-{shard_idx:04d}-of-{n_shards:04d}.parquet"
    results_final = log_dir / f"{log_fname}{suffix}"
    logits_final = log_dir / f"{logits_base_fname}{suffix}"
    return {
        "results_final": results_final,
        "results_partial": results_final.with_suffix(".partial.parquet"),
        "logits_final": logits_final,
        "logits_partial": logits_final.with_suffix(".partial.parquet"),
        "status": results_final.with_suffix(".status.json"),
    }


def _get_parquet_row_count(path: Path):
    if not path.exists():
        return None
    return pq.read_metadata(path).num_rows


def _is_shard_complete(shard_paths, expected_rows: int) -> bool:
    if shard_paths["results_partial"].exists() or shard_paths["logits_partial"].exists():
        return False
    results_rows = _get_parquet_row_count(shard_paths["results_final"])
    logits_rows = _get_parquet_row_count(shard_paths["logits_final"])
    if results_rows != expected_rows or logits_rows != expected_rows:
        return False
    status_path = shard_paths["status"]
    if status_path.exists():
        payload = json.loads(status_path.read_text())
        if not payload.get("completed", False):
            return False
    return True


def _cleanup_incomplete_shard(shard_paths) -> None:
    for key in ("results_final", "results_partial", "logits_final", "logits_partial", "status"):
        path = shard_paths[key]
        if path.exists():
            path.unlink()


def _write_shard_status(status_path: Path, **kwargs):
    status_path.write_text(json.dumps(kwargs, indent=2))


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="CoQA inference with Gemma-4-E4B via vLLM")
    parser.add_argument("--model_path", type=str, default="google/gemma-4-e4b-it",
                        help="HuggingFace model name or local path")
    parser.add_argument("--output_dir", type=str, default="data/llm-inference",
                        help="Root output directory; results saved to output_dir/coqa/")
    parser.add_argument("--data_split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--pe_method", type=str, default="raw",
                        choices=["raw", "cot"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_responses", type=int, default=10,
                        help="Number of responses to generate per question")
    parser.add_argument("--gpu_ids", type=lambda s: [int(x) for x in s.split(",")],
                        default=[0], help="Comma-separated GPU IDs")
    parser.add_argument("--gpu_util", type=float, default=0.7,
                        help="Fraction of GPU memory for vLLM (0.0–1.0)")
    parser.add_argument("--max_input_len", type=str, default="32k",
                        help="Max input length, e.g. '32k'")
    parser.add_argument("--n_sample_per_parquet", type=int, default=10000,
                        help="Max samples per output shard")
    parser.add_argument("--vllm_batch_size", type=int, default=500,
                        help="Max prompts per llm.chat() call")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in args.gpu_ids)
    _set_random_seed(args.seed)

    llm_name = Path(args.model_path).name.split("--")[-1]
    log_dir = Path(args.output_dir) / "coqa"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_fname = _get_log_fname(
        task="coqa",
        max_input_len=args.max_input_len,
        llm_name=llm_name,
        pe_method=args.pe_method,
        n_response=args.num_responses,
        data_split=args.data_split,
    )
    m_set = re.search(r"(_set-\w+)$", log_fname)
    if m_set:
        logits_base_fname = log_fname[:m_set.start()] + "_logits" + m_set.group(1)
    else:
        logits_base_fname = log_fname + "_logits"

    output_fpath = log_dir / f"{log_fname}.parquet"
    ckpt_path = log_dir / f"{log_fname}_ckpt.json"
    running_info_path = log_dir / f"running_info_{log_fname}.json"

    print(f"Model:      {args.model_path}")
    print(f"Output dir: {log_dir}")
    print(f"Split:      {args.data_split}")

    if output_fpath.exists():
        print(f"Results already exist at: {output_fpath}")
        return
    if running_info_path.exists():
        print("All shards already completed (running_info exists). Nothing to do.")
        return

    existing_ckpt = None
    if ckpt_path.exists():
        existing_ckpt = json.loads(ckpt_path.read_text())
        if existing_ckpt["n_per_shard"] != args.n_sample_per_parquet:
            raise ValueError(
                f"--n_sample_per_parquet changed between runs "
                f"({existing_ckpt['n_per_shard']} → {args.n_sample_per_parquet}). "
                "Delete existing shards and the checkpoint, or restore the original value."
            )
        print("Checkpoint found. Will validate shard completeness after loading the dataset.")

    algo_config = _resolve_llm_config(args.model_path)
    algo_config.llm_name = args.model_path
    algo_config.pe_method = args.pe_method

    match_max = re.match(r"(\d+)k", args.max_input_len)
    if match_max:
        algo_config.max_input_len = int(match_max.group(1)) * 1024
    else:
        algo_config.max_input_len = int(args.max_input_len)
    algo_config.max_model_len = algo_config.output_token_len + algo_config.max_input_len

    print(f"Max model len:   {algo_config.max_model_len}")
    print(f"Max input len:   {algo_config.max_input_len}")
    print(f"Output tok len:  {algo_config.output_token_len}")
    print(f"Temperature:     {algo_config.temperature}")
    print(f"Top-K:           {algo_config.top_k}")

    sampling_params = SamplingParams(
        n=args.num_responses,
        temperature=algo_config.temperature,
        top_k=algo_config.top_k,
        max_tokens=algo_config.output_token_len,
        logprobs=20,
    )

    print(f"Loading CoQA dataset (split={args.data_split}) ...")
    dataset = load_coqa(data_split=args.data_split, pe_method=args.pe_method)

    n_total = len(dataset["prompt"])
    n_per_shard = args.n_sample_per_parquet
    n_shards = math.ceil(n_total / n_per_shard)
    print(f"Total samples: {n_total}, shards: {n_shards} (max {n_per_shard}/shard)")

    if existing_ckpt is not None:
        if existing_ckpt["n_total"] != n_total or existing_ckpt["n_shards"] != n_shards:
            raise ValueError(
                "Dataset size or shard count changed since the checkpoint was created."
            )

    ckpt_path.write_text(json.dumps(
        {"n_total": n_total, "n_shards": n_shards, "n_per_shard": n_per_shard}, indent=2
    ))

    pending_indices = []
    for shard_idx in range(n_shards):
        shard_paths = _get_shard_paths(log_dir, log_fname, logits_base_fname, shard_idx, n_shards)
        expected_rows = _expected_rows_for_shard(shard_idx, n_total, n_per_shard)
        if _is_shard_complete(shard_paths, expected_rows):
            continue
        pending_indices.append(shard_idx)

    if not pending_indices:
        print(f"All {n_shards} shards already exist. Nothing to do.")
        return
    n_done = n_shards - len(pending_indices)
    if n_done > 0:
        print(f"Resuming: {n_done}/{n_shards} shards done. Pending: {pending_indices}")
    else:
        print("Running inference ...")

    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        max_model_len=algo_config.max_model_len,
        enforce_eager=algo_config.eager_mode,
        gpu_memory_utilization=args.gpu_util,
        tensor_parallel_size=len(args.gpu_ids),
        trust_remote_code=True,
        logprobs_mode="raw_logits",
    )

    start_time = time.time()

    for shard_idx in pending_indices:
        start = shard_idx * n_per_shard
        end = min(start + n_per_shard, n_total)
        shard_paths = _get_shard_paths(log_dir, log_fname, logits_base_fname, shard_idx, n_shards)
        expected_rows = end - start
        chunk = dataset.select(range(start, end))
        chunk_size = end - start
        batch_size = args.vllm_batch_size
        n_batches = math.ceil(chunk_size / batch_size)

        print(f"Shard {shard_idx+1}/{n_shards}: samples {start}-{end-1} "
              f"({n_batches} batch(es) of ≤{batch_size})")

        if any(path.exists() for path in shard_paths.values()):
            print(f"  Removing incomplete shard artifacts ...")
            _cleanup_incomplete_shard(shard_paths)

        _write_shard_status(
            shard_paths["status"],
            shard_idx=shard_idx, n_shards=n_shards,
            start=start, end=end, n_batches=n_batches,
            expected_rows=expected_rows,
            written_rows_results=0, written_rows_logits=0, completed=False,
        )

        first_end = min(batch_size, chunk_size)
        first_batch = chunk.select(range(0, first_end))
        print(f"  Batch 1/{n_batches}: chunk rows 0-{first_end-1}")
        first_res_df, first_log_df = infer_llm_on_dataset(
            llm, first_batch, sampling_params, algo_config=algo_config, task_type="qa",
        )
        first_res_table = first_res_df.to_arrow()
        first_log_table = first_log_df.to_arrow()
        written_rows_results = first_res_table.num_rows
        written_rows_logits = first_log_table.num_rows

        with pq.ParquetWriter(str(shard_paths["results_partial"]), first_res_table.schema) as rw, \
             pq.ParquetWriter(str(shard_paths["logits_partial"]), first_log_table.schema) as lw:
            rw.write_table(first_res_table)
            lw.write_table(first_log_table)

            del first_batch, first_res_df, first_log_df, first_res_table, first_log_table
            gc.collect()

            for b_idx in range(1, n_batches):
                b_start = b_idx * batch_size
                b_end = min(b_start + batch_size, chunk_size)
                batch = chunk.select(range(b_start, b_end))
                print(f"  Batch {b_idx+1}/{n_batches}: chunk rows {b_start}-{b_end-1}")
                res_df, log_df = infer_llm_on_dataset(
                    llm, batch, sampling_params, algo_config=algo_config, task_type="qa",
                )
                res_table = res_df.to_arrow()
                log_table = log_df.to_arrow()
                rw.write_table(res_table)
                lw.write_table(log_table)
                written_rows_results += res_table.num_rows
                written_rows_logits += log_table.num_rows
                del batch, res_df, log_df, res_table, log_table
                gc.collect()

        actual_results_rows = _get_parquet_row_count(shard_paths["results_partial"])
        actual_logits_rows = _get_parquet_row_count(shard_paths["logits_partial"])
        if actual_results_rows != expected_rows or actual_logits_rows != expected_rows:
            raise RuntimeError(
                f"Shard {shard_idx} write incomplete: expected {expected_rows} rows, "
                f"got results={actual_results_rows}, logits={actual_logits_rows}"
            )

        os.replace(shard_paths["results_partial"], shard_paths["results_final"])
        os.replace(shard_paths["logits_partial"], shard_paths["logits_final"])
        _write_shard_status(
            shard_paths["status"],
            shard_idx=shard_idx, n_shards=n_shards,
            start=start, end=end, n_batches=n_batches,
            expected_rows=expected_rows,
            written_rows_results=actual_results_rows,
            written_rows_logits=actual_logits_rows,
            completed=True,
        )
        print(f"  Shard {shard_idx+1}/{n_shards} completed.")

    elapsed = time.time() - start_time
    running_info = {
        "inference_time": elapsed,
        "num_samples": n_total,
        "n_shards": n_shards,
    }
    with open(running_info_path, "w") as f:
        json.dump(running_info, f, indent=4)

    print(f"Saved {n_shards} shard(s) to {log_dir}")

    llm.llm_engine.engine_core.shutdown()
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("GPU memory released.")


if __name__ == "__main__":
    main()
