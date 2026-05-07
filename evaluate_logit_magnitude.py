"""Evaluate Logit Magnitude uncertainty on LLM inference outputs.

Loads result and logits parquets, computes per-token L2 norm with patience
aggregation, and prints AUROC/AURAC/Balanced Accuracy metrics.

Usage
-----
python evaluate_logit_magnitude.py \\
    --result_parquet data/llm-inference/coqa/coqa_32k_gemma-4-e4b-it_raw_nres-1_set-test.parquet \\
    --logits_parquet data/llm-inference/coqa/coqa_32k_gemma-4-e4b-it_raw_nres-1_logits_set-test.parquet \\
    --dataset coqa \\
    --M 5 --W 20
"""

import argparse
import json
from pathlib import Path

import polars as pl

from logit_magnitude_src.logit_magnitude import (
    _load_parquet_sharded,
    compute_logit_magnitude_patience_from_topk,
    is_correct,
    is_correct_qa,
)


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate Logit Magnitude UQ")
    parser.add_argument("--result_parquet", type=str, required=True,
                        help="Path to result parquet (stem for sharded files)")
    parser.add_argument("--logits_parquet", type=str, required=True,
                        help="Path to logits parquet (stem for sharded files)")
    parser.add_argument("--dataset", type=str, default="coqa",
                        help="Dataset name; determines correctness function")
    parser.add_argument("--response_idx", type=int, default=0,
                        help="Which LLM response to evaluate (default: 0)")
    parser.add_argument("--M", type=int, default=5,
                        help="Min-heap size for patience aggregation (default: 5)")
    parser.add_argument("--W", type=int, default=20,
                        help="Patience window — consecutive non-updates to halt (default: 20)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Optional path to save metrics JSON")
    return parser.parse_args()


def main():
    args = get_args()

    print(f"Loading result parquet: {args.result_parquet}")
    result_df = _load_parquet_sharded(Path(args.result_parquet))
    print(f"Loading logits parquet: {args.logits_parquet}")
    logits_df = _load_parquet_sharded(Path(args.logits_parquet))
    print(f"Loaded {result_df.height} samples.")

    qa_datasets = {"coqa", "newsqa", "emrqa", "nq", "triviaqa"}
    correct_fn = is_correct_qa if args.dataset.lower() in qa_datasets else is_correct

    print(f"\nComputing Logit Magnitude (M={args.M}, W={args.W}) ...")
    metrics = compute_logit_magnitude_patience_from_topk(
        result_df=result_df,
        logits_df=logits_df,
        response_idx=args.response_idx,
        correct_fn=correct_fn,
        M=args.M,
        W=args.W,
    )

    print("\n=== Logit Magnitude Results ===")
    print(f"  Dataset : {args.dataset}")
    print(f"  M={args.M}, W={args.W}, response_idx={args.response_idx}")
    print(f"  AUROC        : {metrics['auroc']:.4f}")
    print(f"  AURAC        : {metrics['aurac']:.4f}")
    print(f"  Balanced Acc : {metrics['balanced_acc']:.4f}")
    print(f"  Mean uncertainty : {metrics['mean']:.4f}")

    if args.output_path is not None:
        out = {k: v for k, v in metrics.items() if k != "per_sample"}
        out.update({"dataset": args.dataset, "M": args.M, "W": args.W})
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nMetrics saved to {args.output_path}")


if __name__ == "__main__":
    main()
