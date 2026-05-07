"""Logit Magnitude uncertainty estimation.

Core algorithm: per-token L2 norm of top-M logits, aggregated with a
patience stopping rule. Also contains correctness and evaluation utilities.

Public API
----------
compute_logit_magnitude_patience_from_topk(result_df, logits_df, M, W) -> dict
    Main entry point. Returns AUROC, AURAC, Balanced Accuracy and per-sample scores.
"""

from __future__ import annotations

import heapq
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from rouge_score import rouge_scorer as _rouge_scorer
from sklearn.metrics import balanced_accuracy_score, roc_curve

from logit_magnitude_src.metrics import (
    auroc as _auroc,
    aurac as _aurac,
)


# ---------------------------------------------------------------------------
# Parquet loading
# ---------------------------------------------------------------------------

def _load_parquet_sharded(path: Path) -> pl.DataFrame:
    """Load a single parquet or auto-detect shards matching {stem}-NNNN-of-MMMM.parquet."""
    if path.exists():
        return pl.read_parquet(path)
    shards = sorted(path.parent.glob(f"{path.name}-*-of-*.parquet"))
    if shards:
        return pl.concat([pl.read_parquet(s) for s in shards], how="diagonal_relaxed")
    raise FileNotFoundError(f"Parquet not found at {path} (no shards detected)")


# ---------------------------------------------------------------------------
# Correctness functions
# ---------------------------------------------------------------------------

def canon_answer(ans: Any) -> Optional[Any]:
    if ans is None:
        return None
    if isinstance(ans, (list, tuple)):
        return tuple(sorted([str(x) for x in ans if x is not None]))
    return str(ans)


def is_correct(label: Any, pred: Any) -> bool:
    label_norm = canon_answer(label)
    pred_norm = canon_answer(pred)
    if label_norm is None or pred_norm is None:
        return False
    return label_norm == pred_norm


_ROUGE_SCORER = _rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)


def _rouge_l_score(label_str: str, pred_str: str) -> float:
    """ROUGE-L F1 between label and prediction."""
    score = _ROUGE_SCORER.score(label_str.lower(), pred_str.lower())
    return float(score["rougeL"].fmeasure)


def is_correct_qa(label: Any, pred: Any, rouge_l_threshold: float = 0.3) -> bool:
    """Correctness for QA tasks using ROUGE-L >= threshold."""
    if label is None or pred is None:
        return False
    return _rouge_l_score(str(canon_answer(label)), str(canon_answer(pred))) >= rouge_l_threshold


# ---------------------------------------------------------------------------
# Per-token score
# ---------------------------------------------------------------------------

def _token_l2norm(topk: List[Dict]) -> Optional[float]:
    """L2 norm of the positive part of top-M logits at a single token position."""
    logits = [float(item["logit"]) for item in topk if item is not None]
    if not logits:
        return None
    pos = np.maximum(np.array(logits, dtype=np.float64), 0.0)
    return float(np.linalg.norm(pos))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate(values: List[float], topm_worst: int) -> Optional[float]:
    """Aggregate token-level scores.

    topm_worst=0 → mean over all tokens.
    topm_worst=M → mean of the M highest scores.
    """
    if not values:
        return None
    if topm_worst <= 0:
        return float(np.mean(values))
    k = min(topm_worst, len(values))
    return float(np.mean(sorted(values, reverse=True)[:k]))


def _patience_aggregate(
    scores: List[float], M: int, W: int
) -> Tuple[Optional[float], int, bool]:
    """Apply the patience rule to a per-token score sequence.

    Maintains a min-heap of size M tracking the M largest scores seen.
    Halts when W consecutive tokens fail to update the heap.

    Returns (aggregated_score, stopping_step_tau, triggered).
    """
    heap: List[float] = []
    n_patience = 0
    triggered = False
    tau = 0
    for t, score in enumerate(scores, start=1):
        tau = t
        if len(heap) < M:
            heapq.heappush(heap, score)
            n_patience = 0
        elif score > heap[0]:
            heapq.heapreplace(heap, score)
            n_patience = 0
        else:
            n_patience += 1
            if n_patience >= W:
                triggered = True
                break
    agg = float(np.mean(heap)) if heap else None
    return agg, tau, triggered


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def compute_logit_magnitude_patience_from_topk(
    result_df: pl.DataFrame,
    logits_df: Optional[pl.DataFrame] = None,
    labels: Optional[List[Any]] = None,
    response_idx: int = 0,
    return_per_sample: bool = False,
    correct_fn: Callable = is_correct,
    M: int = 5,
    W: int = 20,
) -> Dict[str, Any]:
    """Logit Magnitude with patience rule.

    For each sample, computes per-token L2 norms of the top-M logits and applies
    the patience stopping criterion: maintain the M largest scores (min-heap), halt
    when W consecutive tokens fail to update the heap. Returns mean(H_tau).

    Args:
        result_df:     Polars DataFrame with prompt, label, answer_0 columns.
        logits_df:     Polars DataFrame with gen_topk_logits_0 column.
        labels:        Optional list of ground-truth labels (defaults to result_df["label"]).
        response_idx:  Which response to evaluate (default 0).
        return_per_sample: If True, include per-sample scores in result.
        correct_fn:    Correctness function (is_correct or is_correct_qa).
        M:             Min-heap size for patience aggregation.
        W:             Patience window (consecutive non-updates to trigger halt).

    Returns:
        Dict with keys: mean, std, auroc, aurac, balanced_acc, and optionally per_sample.
    """
    if labels is None:
        labels = (
            result_df.get_column("label").to_list()
            if "label" in result_df.columns
            else [None] * result_df.height
        )

    ans_col = f"answer_{response_idx}"
    logits_col = f"gen_topk_logits_{response_idx}"
    logits_rows = logits_df.to_dicts() if logits_df is not None else None

    uncertainty: List[Optional[float]] = []
    correct_flags: List[bool] = []

    for i, row in enumerate(result_df.iter_rows(named=True)):
        pred = row.get(ans_col, None)
        label = labels[i] if i < len(labels) else None
        correct_flags.append(correct_fn(label, pred))

        if logits_rows is None:
            uncertainty.append(None)
            continue

        seq_topk = logits_rows[i].get(logits_col, None)
        if seq_topk is None or not isinstance(seq_topk, list) or len(seq_topk) == 0:
            uncertainty.append(None)
            continue

        token_scores: List[float] = []
        for token_topk in seq_topk:
            if isinstance(token_topk, list) and len(token_topk) > 0:
                s = _token_l2norm(token_topk)
                if s is not None:
                    token_scores.append(s)

        if not token_scores:
            uncertainty.append(None)
            continue

        agg, _tau, _triggered = _patience_aggregate(token_scores, M, W)
        uncertainty.append(agg)

    return summarize_with_detection_metrics(uncertainty, correct_flags, return_per_sample)


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

def auroc_from_uncertainty(uncertainty: List[Optional[float]], correct_flags: List[bool]) -> float:
    """AUROC for error detection (uncertainty convention: higher = more uncertain)."""
    y_true: List[int] = []
    y_score: List[float] = []
    for unc, correct in zip(uncertainty, correct_flags):
        if unc is None or not np.isfinite(unc):
            continue
        y_score.append(float(unc))
        y_true.append(0 if correct else 1)
    if len(set(y_true)) < 2 or len(y_score) == 0:
        return 0.5
    return _auroc(y_true, y_score)


def find_optimal_threshold_youden(
    uncertainty: List[Optional[float]],
    correct_flags: List[bool],
) -> float:
    """Optimal detection threshold via Youden's J (TPR - FPR)."""
    valid = [
        (float(u), bool(c))
        for u, c in zip(uncertainty, correct_flags)
        if u is not None and np.isfinite(u)
    ]
    if len(valid) < 2:
        return 0.5
    y_score = [v[0] for v in valid]
    y_true = [0 if v[1] else 1 for v in valid]
    if len(set(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    best_idx = int(np.argmax(tpr - fpr))
    return float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5


def _balanced_acc_at_youden(
    uncertainty: List[Optional[float]],
    correct_flags: List[bool],
) -> float:
    """Balanced accuracy at the optimal Youden threshold."""
    valid = [
        (float(u), bool(c))
        for u, c in zip(uncertainty, correct_flags)
        if u is not None and np.isfinite(u)
    ]
    if len(valid) < 2:
        return float("nan")
    y_score = [v[0] for v in valid]
    y_true = [0 if v[1] else 1 for v in valid]
    if len(set(y_true)) < 2:
        return float("nan")
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    best_idx = int(np.argmax(tpr - fpr))
    thresh = float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5
    y_pred = [1 if s >= thresh else 0 for s in y_score]
    return float(balanced_accuracy_score(y_true, y_pred))


def summarize_with_detection_metrics(
    uncertainty: List[Optional[float]],
    correct_flags: List[bool],
    return_per_sample: bool,
) -> Dict[str, Any]:
    values = np.array([val for val in uncertainty if val is not None], dtype=np.float64)

    valid = [
        (u, c)
        for u, c in zip(uncertainty, correct_flags)
        if u is not None and np.isfinite(u)
    ]
    if valid:
        aurac_unc = np.array([u for u, _ in valid], dtype=np.float64)
        aurac_acc = np.array([float(c) for _, c in valid], dtype=np.float64)
        aurac_val = _aurac(aurac_acc, aurac_unc)
    else:
        aurac_val = float("nan")

    result = {
        "mean": float(values.mean()) if values.size > 0 else float("nan"),
        "std": float(values.std(ddof=1)) if values.size > 1 else float("nan"),
        "auroc": auroc_from_uncertainty(uncertainty, correct_flags),
        "aurac": aurac_val,
        "balanced_acc": _balanced_acc_at_youden(uncertainty, correct_flags),
    }
    if return_per_sample:
        result["per_sample"] = uncertainty
    return result


def bootstrap_uq_metrics(
    uncertainty: List[Optional[float]],
    correct_flags: List[bool],
    n_bootstrap: int = 1000,
    seed: int = 0,
    opt_thresh: Optional[float] = None,
) -> Dict[str, Dict[str, float]]:
    """Bootstrap AUROC, AURAC, and balanced accuracy metrics.

    Returns {metric: {"mean": float, "std": float}} rounded to 3 decimal places.
    """
    valid = [
        (float(u), bool(c))
        for u, c in zip(uncertainty, correct_flags)
        if u is not None and np.isfinite(u)
    ]
    nan_dict: Dict[str, float] = {"mean": float("nan"), "std": float("nan")}

    if len(valid) < 2:
        return {
            "auroc": nan_dict, "aurac": nan_dict, "balanced_acc": nan_dict,
        }

    unc_arr = np.array([v[0] for v in valid], dtype=np.float64)
    cor_arr = np.array([v[1] for v in valid], dtype=bool)
    n = len(valid)

    rng = np.random.default_rng(seed)
    auroc_vals: List[float] = []
    aurac_vals: List[float] = []
    balanced_acc_vals: List[float] = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        u_b = unc_arr[idx]
        c_b = cor_arr[idx]
        if c_b.all() or not c_b.any():
            continue
        y_true = np.where(c_b, 0, 1)
        try:
            auroc_vals.append(float(_auroc(y_true, u_b)))
            aurac_vals.append(float(_aurac(c_b.astype(np.float64), u_b)))
        except Exception:
            pass

        try:
            if opt_thresh is not None:
                thresh = opt_thresh
            else:
                fpr_b, tpr_b, threshs_b = roc_curve(y_true, u_b)
                thresh = float(threshs_b[int(np.argmax(tpr_b - fpr_b))]) if len(threshs_b) > 0 else 0.5
            y_pred_b = (u_b >= thresh).astype(int)
            balanced_acc_vals.append(float(balanced_accuracy_score(y_true, y_pred_b)))
        except Exception:
            pass

    def _summarize(vals: List[float]) -> Dict[str, float]:
        if not vals:
            return nan_dict.copy()
        arr = np.array(vals)
        return {"mean": round(float(arr.mean()), 3), "std": round(float(arr.std()), 3)}

    return {
        "auroc": _summarize(auroc_vals),
        "aurac": _summarize(aurac_vals),
        "balanced_acc": _summarize(balanced_acc_vals),
    }
