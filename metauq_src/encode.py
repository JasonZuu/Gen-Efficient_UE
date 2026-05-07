"""Offline embedding encoding pipeline for MetaUE (vLLM-based).

Encodes prompts from inference result parquets using vLLM's embedding API
and saves the vectors to a single parquet per split.

Parquet layout
--------------
Root: {output_dir}/{dataset}_{safe_emb}/
File: {split}.parquet
Schema: prompt (Utf8), label (Utf8), embedding (List[Float32])
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import polars as pl
import torch
from transformers import AutoTokenizer
from vllm import LLM


def _safe_emb(emb_model: str) -> str:
    """Return a filesystem-safe tag from an embedding model name."""
    name = emb_model.split("/")[-1]
    if "--" in name:
        name = name.split("--", 1)[-1]
    return name


def embedding_parquet_dir(output_dir: str, dataset: str, emb_model: str) -> Path:
    """Return the subdirectory for a (dataset, emb_model) pair."""
    return Path(output_dir) / f"{dataset}_{_safe_emb(emb_model)}"


def embedding_parquet_path(
    output_dir: str,
    dataset: str,
    split: str,
    emb_model: str,
) -> Path:
    """Return the parquet path for a specific (dataset, split, emb_model)."""
    return embedding_parquet_dir(output_dir, dataset, emb_model) / f"{split}.parquet"


def _vllm_extra_kwargs(model_name: str) -> dict:
    """Return model-specific extra kwargs for vLLM LLM initialization."""
    name_lower = model_name.lower()
    if "vl" in name_lower or "vision" in name_lower:
        return {
            "max_model_len": 32768,
            "limit_mm_per_prompt": {"image": 0, "video": 0},
        }
    return {}


def _apply_chat_template(tokenizer, prompt) -> str:
    """Convert a prompt value from the result parquet to a plain string."""
    if prompt is None:
        return ""
    if isinstance(prompt, list):
        try:
            return tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            return " ".join(m.get("content", "") for m in prompt if isinstance(m, dict))
    return str(prompt)


def encode_and_save(
    result_df: pl.DataFrame,
    output_dir: str,
    dataset: str,
    split: str,
    emb_model_name: str,
    gpu_ids: List[int],
    gpu_util: float = 0.9,
) -> None:
    """Encode all prompts from result_df with vLLM and write a single parquet.

    Skips if the output file already exists.

    Args:
        result_df:      Polars DataFrame with at least 'prompt' and 'label' columns.
        output_dir:     Root directory for parquet output.
        dataset:        Dataset name (used for path construction).
        split:          Split name, e.g. 'train', 'val', 'test'.
        emb_model_name: HuggingFace model name/path for the embedding model.
        gpu_ids:        List of GPU device IDs; len determines tensor_parallel_size.
        gpu_util:       Fraction of GPU memory vLLM may use (0.0–1.0).
    """
    out_path = embedding_parquet_path(output_dir, dataset, split, emb_model_name)

    if out_path.exists():
        print(f"  [{split}] Embedding parquet already exists, skipping. ({out_path})")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  [{split}] Formatting {result_df.height} prompts via chat template ...")
    tokenizer = AutoTokenizer.from_pretrained(emb_model_name, trust_remote_code=True)
    raw_prompts = result_df["prompt"].to_list()
    texts: List[str] = [_apply_chat_template(tokenizer, p) for p in raw_prompts]
    del tokenizer

    labels: List[str] = [str(l) if l is not None else "" for l in result_df["label"].to_list()]

    print(f"  [{split}] Encoding {len(texts)} prompts with vLLM "
          f"(tensor_parallel_size={len(gpu_ids)}, gpu_memory_utilization={gpu_util}) ...")
    llm = LLM(
        model=emb_model_name,
        runner="pooling",
        trust_remote_code=True,
        tensor_parallel_size=len(gpu_ids),
        gpu_memory_utilization=gpu_util,
        **_vllm_extra_kwargs(emb_model_name),
    )
    outputs = llm.embed(texts)
    embeddings: List[List[float]] = [o.outputs.embedding for o in outputs]
    del llm

    df = pl.DataFrame({
        "prompt": texts,
        "label": labels,
        "embedding": embeddings,
    })
    df.write_parquet(str(out_path))
    print(f"  [{split}] Saved: {out_path}  ({df.height} rows, embed_dim={len(embeddings[0])})")


def load_embedding_parquet(
    output_dir: str,
    dataset: str,
    split: str,
    emb_model: str,
) -> torch.Tensor:
    """Load a pre-encoded parquet and return an (N, D) float32 tensor on CPU."""
    path = embedding_parquet_path(output_dir, dataset, split, emb_model)
    if not path.exists():
        raise FileNotFoundError(
            f"Embedding parquet not found: {path}\n"
            f"Run first:  python encode_metaue.py --dataset {dataset} "
            f"--embedding_model_name {emb_model}"
        )
    df = pl.read_parquet(str(path))
    tensor = torch.tensor(df["embedding"].to_list(), dtype=torch.float32)
    return tensor
