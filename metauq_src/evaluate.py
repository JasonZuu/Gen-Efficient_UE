import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Any, Callable, Dict, List, Optional

from metaue_src.dataset import MetaUEDataset, collate_fn_embeddings, collate_fn_tokenized
from metaue_src.model import MetaUEModel
from logit_magnitude_src.logit_magnitude import (
    summarize_with_detection_metrics,
    auroc_from_uncertainty,
)


def evaluate_metaue(
    model: MetaUEModel,
    test_dataset: MetaUEDataset,
    device: str = "cpu",
    batch_size: int = 128,
    gt_correct_flags: Optional[List[bool]] = None,
) -> Dict[str, Any]:
    """Evaluate MetaUE on test_dataset.

    Args:
        gt_correct_flags: Real GT correctness flags aligned with test_dataset order.
            When provided, AUROC/AURAC/Balanced Accuracy are computed against GT correctness.
            When None, falls back to pseudo-labels from dataset (for internal validation only).

    Returns AUROC, AURAC, Balanced Accuracy, and per-sample predictions.
    """
    from sklearn.metrics import roc_curve

    model.eval()
    model.to(device)

    collate = collate_fn_embeddings if test_dataset.use_embeddings else collate_fn_tokenized
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    all_probs: List[float] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for batch in loader:
            if test_dataset.use_embeddings:
                emb, lbl = batch
                emb = emb.to(device)
                probs = model.predict_proba_from_embedding(emb).cpu()
            else:
                input_ids, attention_mask, lbl = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                probs = model.predict_proba(input_ids, attention_mask).cpu()

            all_probs.extend(probs.tolist())
            all_labels.extend(lbl.int().tolist())

    # Pseudo-label correctness from dataset (label near 0 = certain = correct)
    pseudo_correct_flags = [l == 0 for l in all_labels]
    uncertainty = all_probs  # higher score → more uncertain

    correct_flags = gt_correct_flags if gt_correct_flags is not None else pseudo_correct_flags

    metrics = summarize_with_detection_metrics(
        uncertainty, correct_flags, return_per_sample=True
    )

    if gt_correct_flags is not None:
        pseudo_metrics = summarize_with_detection_metrics(
            uncertainty, pseudo_correct_flags, return_per_sample=False
        )
        metrics["pseudo_label_metrics"] = {k: v for k, v in pseudo_metrics.items()}

    metrics["n_samples"] = len(all_labels)

    if gt_correct_flags is not None:
        metrics["n_correct_gt"] = int(sum(gt_correct_flags))
        metrics["n_incorrect_gt"] = int(len(gt_correct_flags) - sum(gt_correct_flags))

    return metrics
