"""UQ evaluation metrics: AUROC and AURAC."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def auroc(y_true, y_score) -> float:
    """Area Under the ROC Curve.

    Args:
        y_true:  Binary labels (0/1 array-like). 1 = positive class.
        y_score: Continuous scores; higher means more likely positive.

    Returns:
        AUROC as a float.
    """
    return float(roc_auc_score(y_true, y_score))


def accuracy_at_quantile(
    accuracies: np.ndarray,
    uncertainties: np.ndarray,
    quantile: float,
) -> float:
    """Mean accuracy of samples whose uncertainty is at or below the given quantile."""
    cutoff = np.quantile(uncertainties, quantile)
    select = uncertainties <= cutoff
    return float(np.mean(accuracies[select]))


def area_under_thresholded_accuracy(
    accuracies: np.ndarray,
    uncertainties: np.ndarray,
) -> float:
    """Area Under the Rejection-Accuracy Curve (AURAC).

    Sweeps coverage quantiles from 0.1 to 1.0 in 20 equal steps.
    At each quantile q, retains the most-confident samples and computes their mean accuracy.

    Args:
        accuracies:    Per-sample correctness as a 0/1 (or bool) numpy array.
        uncertainties: Per-sample uncertainty scores (lower = more confident).

    Returns:
        Scalar AURAC value (higher is better).
    """
    quantiles = np.linspace(0.1, 1, 20)
    select_accuracies = np.array(
        [accuracy_at_quantile(accuracies, uncertainties, q) for q in quantiles]
    )
    dx = quantiles[1] - quantiles[0]
    area = float((select_accuracies * dx).sum())
    return area


#: Public alias for area_under_thresholded_accuracy.
aurac = area_under_thresholded_accuracy
