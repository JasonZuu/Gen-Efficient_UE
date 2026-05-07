"""Offline embedding encoding for MetaUE (vLLM-based).

Encodes prompts from inference result parquets using vLLM's embedding API
and saves vectors to parquet. Run this once before any sweep.

Embeddings depend only on the dataset and embedding model, NOT on which LLM
produced the inference data — so this needs to run only once per dataset.

Usage
-----
python encode_metaue.py \\
    --dataset coqa \\
    --embedding_model_name Qwen/Qwen3-VL-Embedding-2B \\
    --output_dir data/metaue_embeddings \\
    --inference_dir data/llm-inference \\
    --gpu_ids 0 \\
    --splits train val test

Output layout
-------------
{output_dir}/{dataset}_{safe_emb}/
    {split}.parquet    <- columns: prompt, label, embedding
"""

import argparse
import os
import sys
from pathlib import Path

# Set CUDA_VISIBLE_DEVICES before any CUDA-aware import
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gpu_ids", type=lambda s: [int(x) for x in s.split(",")], default=[0])
_pre_args, _ = _pre.parse_known_args(sys.argv[1:])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, _pre_args.gpu_ids))

from metaue_src.config import DEFAULT_CONFIG
from metaue_src.encode import encode_and_save, embedding_parquet_path
from logit_magnitude_src.logit_magnitude import _load_parquet_sharded


def _find_any_result_stem(
    inference_dir: str,
    dataset: str,
    split: str,
    max_input_len: str,
    pe_method: str,
    n_responses: int,
) -> "list[Path]":
    """Return stems for ALL available result parquets for the given split."""
    base = Path(inference_dir) / dataset
    pattern = f"{dataset}_{max_input_len}_*_{pe_method}_nres-{n_responses}_set-{split}"

    candidates = sorted(base.glob(f"{pattern}.parquet")) + sorted(base.glob(f"{pattern}-*-of-*.parquet"))
    matches = [p for p in candidates if "_logits" not in p.name]

    seen: set[Path] = set()
    stems: list[Path] = []
    for p in matches:
        name = p.name
        if "-of-" in name:
            stem_name = name.rsplit("-", 3)[0]
            stem = p.parent / stem_name
        else:
            stem = p.with_suffix("")
        if stem not in seen:
            seen.add(stem)
            stems.append(stem)
    return stems


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode MetaUE prompts to embedding parquets")
    parser.add_argument("--dataset", type=str, default=DEFAULT_CONFIG["dataset"])
    parser.add_argument("--embedding_model_name", type=str,
                        default=DEFAULT_CONFIG["embedding_model_name"])
    parser.add_argument("--output_dir", type=str,
                        default=DEFAULT_CONFIG["embedding_parquet_dir"])
    parser.add_argument("--inference_dir", type=str, default="data/llm-inference",
                        help="Root directory containing inference parquets")
    parser.add_argument("--splits", type=str, nargs="+",
                        default=["train", "val", "test"])
    parser.add_argument("--gpu_ids", type=lambda s: [int(x) for x in s.split(",")],
                        default=[0], help="GPU device IDs (already applied at startup)")
    parser.add_argument("--gpu_util", type=float, default=0.9)
    parser.add_argument("--max_input_len", type=str, default=DEFAULT_CONFIG["max_input_len"])
    parser.add_argument("--pe_method", type=str, default=DEFAULT_CONFIG["pe_method"])
    parser.add_argument("--n_responses", type=int, default=DEFAULT_CONFIG["n_responses"])
    return parser.parse_args()


def main():
    args = _parse_args()

    print(f"Dataset         : {args.dataset}")
    print(f"Embedding model : {args.embedding_model_name}")
    print(f"Output dir      : {args.output_dir}")
    print(f"Inference dir   : {args.inference_dir}")
    print(f"Splits          : {args.splits}")
    print()

    for split in args.splits:
        result_stems = _find_any_result_stem(
            args.inference_dir,
            args.dataset,
            split,
            args.max_input_len,
            args.pe_method,
            args.n_responses,
        )
        if not result_stems:
            print(f"[{split}] No inference result parquet found — skipping.")
            continue

        result_df = None
        for result_stem in result_stems:
            print(f"[{split}] Trying parquet stem: {result_stem}")
            try:
                result_df = _load_parquet_sharded(result_stem)
                print(f"[{split}] {result_df.height} rows loaded.")
                break
            except Exception as e:
                print(f"[{split}] Failed to load {result_stem}: {e} — trying next.")

        if result_df is None:
            print(f"[{split}] No valid parquet found — skipping.")
            continue

        encode_and_save(
            result_df=result_df,
            output_dir=args.output_dir,
            dataset=args.dataset,
            split=split,
            emb_model_name=args.embedding_model_name,
            gpu_ids=args.gpu_ids,
            gpu_util=args.gpu_util,
        )

    print()
    print("Done. Encoded parquets:")
    for split in args.splits:
        p = embedding_parquet_path(args.output_dir, args.dataset, split,
                                   args.embedding_model_name)
        status = "OK" if p.exists() else "MISSING"
        print(f"  [{status}] {p}")


if __name__ == "__main__":
    main()
