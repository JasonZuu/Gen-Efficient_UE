import argparse
import torch
from pathlib import Path
from typing import Any, Dict

DEFAULT_CONFIG: Dict[str, Any] = {
    # Model
    "embedding_model_name": "Qwen/Qwen3-VL-Embedding-2B",
    "hidden_dim": 256,
    "dropout": 0.1,
    "pooling": "mean",
    # Label
    "label_fn_name": "logit_magnitude",
    "topk_worst": 5,
    # Training
    "epochs": 100,
    "lr": 1e-3,
    "batch_size": 64,
    "weight_decay": 1e-4,
    "val_steps": 2000,
    "max_length": 4 * 1024,
    "use_bce_loss": False,
    "regression_loss": "mse",
    "rank_lambda": 0.5,
    # Data
    "dataset": "coqa",
    "data_split": "test",
    "train_split": "train",
    "val_split": "val",
    "test_split": "test",
    "max_input_len": "32k",
    "pe_method": "raw",
    "n_responses": 10,
    "response_idx": 0,
    # Paths
    "embedding_parquet_dir": "data/metaue_embeddings",
    "checkpoint_dir": "data/metaue_cache/checkpoints",
    "output_dir": "log/metaue",
}


def _model_tag_from_path(model_path: str) -> str:
    return Path(model_path).name.split("--")[-1]
