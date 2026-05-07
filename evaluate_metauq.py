"""Evaluate a trained MetaUE model on LLM inference outputs.

Loads result parquet, logits parquet, and pre-encoded embedding parquet,
then runs the trained MetaUE MLP and reports AUROC/AUPRC/AURAC/BAS.

Usage
-----
python evaluate_metaue.py \\
    --result_parquet data/llm-inference/coqa/coqa_32k_gemma-4-e4b-it_raw_nres-1_set-test.parquet \\
    --logits_parquet data/llm-inference/coqa/coqa_32k_gemma-4-e4b-it_raw_nres-1_logits_set-test.parquet \\
    --embedding_parquet data/metaue_embeddings/coqa_Qwen3-VL-Embedding-2B/test.parquet \\
    --model_path data/metaue_sweep/last_best_model_path.txt \\
    --dataset coqa
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

from logit_magnitude_src.logit_magnitude import (
    _load_parquet_sharded,
    is_correct,
    is_correct_qa,
)
from metaue_src.dataset import MetaUEDataset
from metaue_src.evaluate import evaluate_metaue
from metaue_src.label_functions import LABEL_FUNCTIONS, LABEL_FN_NEEDS_LOGITS
from metaue_src.model import MetaUEMLP


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained MetaUE model")
    parser.add_argument("--result_parquet", type=str, required=True,
                        help="Path to result parquet (stem for sharded files)")
    parser.add_argument("--logits_parquet", type=str, required=True,
                        help="Path to logits parquet (stem for sharded files)")
    parser.add_argument("--embedding_parquet", type=str, required=True,
                        help="Path to pre-encoded embedding parquet")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to saved MetaUE model weights (.pt file)")
    parser.add_argument("--dataset", type=str, default="coqa",
                        help="Dataset name; determines correctness function")
    parser.add_argument("--response_idx", type=int, default=0,
                        help="Which LLM response to evaluate (default: 0)")
    parser.add_argument("--topk_worst", type=int, default=5,
                        help="Top-K worst for logit_magnitude label fn (default: 5)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--output_path", type=str, default=None,
                        help="Optional path to save metrics JSON")
    return parser.parse_args()


def main():
    args = get_args()

    print(f"Loading result parquet:    {args.result_parquet}")
    result_df = _load_parquet_sharded(Path(args.result_parquet))
    print(f"Loading logits parquet:    {args.logits_parquet}")
    logits_df = _load_parquet_sharded(Path(args.logits_parquet))
    print(f"Loading embedding parquet: {args.embedding_parquet}")
    emb_df = pl.read_parquet(args.embedding_parquet)
    embeddings = torch.tensor(emb_df["embedding"].to_list(), dtype=torch.float32)
    print(f"Loaded {result_df.height} samples, embed_dim={embeddings.shape[1]}.")

    # Compute logit_magnitude labels for dataset construction
    label_fn_kwargs = {"topk_worst": args.topk_worst}
    dataset = MetaUEDataset(
        result_df=result_df,
        logits_df=logits_df,
        label_fn_name="logit_magnitude",
        label_fn_kwargs=label_fn_kwargs,
        response_idx=args.response_idx,
        precomputed_embeddings=embeddings,
    )
    print(f"Dataset size: {len(dataset)} samples")

    # Compute GT correctness flags
    qa_datasets = {"coqa", "newsqa", "emrqa", "nq", "triviaqa"}
    correct_fn = is_correct_qa if args.dataset.lower() in qa_datasets else is_correct
    labels_list = result_df.get_column("label").to_list() if "label" in result_df.columns else []
    preds_list = result_df.get_column(f"answer_{args.response_idx}").to_list() \
        if f"answer_{args.response_idx}" in result_df.columns else []
    gt_correct_flags = [correct_fn(lbl, pred) for lbl, pred in zip(labels_list, preds_list)]

    # Load model
    embed_dim = embeddings.shape[1]
    model = MetaUEMLP(embed_dim)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.to(args.device)
    print(f"Loaded MetaUE model from {args.model_path}")

    # Evaluate
    print(f"\nEvaluating MetaUE on {args.dataset} ...")
    metrics = evaluate_metaue(
        model=model,
        test_dataset=dataset,
        device=args.device,
        batch_size=args.batch_size,
        gt_correct_flags=gt_correct_flags if gt_correct_flags else None,
    )

    print("\n=== MetaUE Results ===")
    print(f"  Dataset      : {args.dataset}")
    print(f"  Model        : {args.model_path}")
    print(f"  AUROC        : {metrics['auroc']:.4f}")
    print(f"  AURAC        : {metrics['aurac']:.4f}")
    print(f"  Balanced Acc : {metrics['balanced_acc']:.4f}")

    if args.output_path is not None:
        out = {k: v for k, v in metrics.items() if k not in ("per_sample",)}
        out.update({"dataset": args.dataset, "model_path": args.model_path})
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nMetrics saved to {args.output_path}")


if __name__ == "__main__":
    main()
