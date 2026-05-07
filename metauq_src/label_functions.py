"""Label generation functions for MetaUE training.

Only logit_magnitude is included. Labels are always raw continuous scores
(regression mode, MSE loss). No GMM calibration or binarization.
"""

import numpy as np
from typing import Any, Callable, Dict, List, Optional

from logit_magnitude_src.logit_magnitude import _token_l2norm, _aggregate


LABEL_FUNCTIONS: Dict[str, Callable] = {}
LABEL_FN_NEEDS_LOGITS: Dict[str, bool] = {}
LABEL_FN_USES_TOPK: Dict[str, bool] = {}
LABEL_FN_IS_REGRESSION: Dict[str, bool] = {}


def register_label_fn(name: str, needs_logits: bool = False, uses_topk: bool = False, is_regression: bool = False):
    """Decorator to register a per-row label function."""
    def decorator(fn):
        LABEL_FUNCTIONS[name] = fn
        LABEL_FN_NEEDS_LOGITS[name] = needs_logits
        LABEL_FN_USES_TOPK[name] = uses_topk
        LABEL_FN_IS_REGRESSION[name] = is_regression
        return fn
    return decorator


@register_label_fn("logit_magnitude", needs_logits=True, uses_topk=True, is_regression=True)
def label_logit_magnitude(
    result_row: Dict[str, Any],
    logits_row: Optional[Dict[str, Any]] = None,
    response_idx: int = 0,
    topk_worst: int = 5,
    **kwargs,
) -> Optional[float]:
    """Raw logit magnitude score (higher = more uncertain).

    Returns the mean of the top-M worst token L2 norms as a continuous
    float label for MSE regression training.
    """
    if logits_row is None:
        return None
    seq_topk = logits_row.get(f"gen_topk_logits_{response_idx}")
    if seq_topk is None or not isinstance(seq_topk, list) or len(seq_topk) == 0:
        return None

    token_scores = []
    for token_topk in seq_topk:
        if not isinstance(token_topk, list) or len(token_topk) == 0:
            continue
        s = _token_l2norm(token_topk)
        if s is not None:
            token_scores.append(s)

    if not token_scores:
        return None

    return _aggregate(token_scores, topk_worst)
