"""Binary classification metrics for malignancy risk."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def compute_binary_metrics(
    y_true: Sequence[float],
    y_prob: Sequence[float],
    threshold: float = 0.5,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_prob, dtype=np.float64)
    if y.size == 0:
        return {"auc": 0.0, "accuracy": 0.0, "sensitivity": 0.0, "specificity": 0.0}

    pred = (p >= threshold).astype(np.int32)
    y_int = y.astype(np.int32)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_int, pred)),
    }

    # AUC requires both classes
    if len(np.unique(y_int)) > 1:
        metrics["auc"] = float(roc_auc_score(y_int, p))
        metrics["auprc"] = float(average_precision_score(y_int, p))
        fpr, tpr, thr = roc_curve(y_int, p)
        # operating point: max Youden
        j = tpr - fpr
        best = int(np.argmax(j))
        metrics["best_threshold"] = float(thr[best]) if best < len(thr) else threshold
    else:
        metrics["auc"] = 0.0
        metrics["auprc"] = 0.0
        metrics["best_threshold"] = threshold

    tn, fp, fn, tp = confusion_matrix(y_int, pred, labels=[0, 1]).ravel()
    metrics["sensitivity"] = float(tp / max(tp + fn, 1))
    metrics["specificity"] = float(tn / max(tn + fp, 1))
    metrics["precision"] = float(tp / max(tp + fp, 1))
    metrics["brier"] = float(brier_score_loss(y_int, p))
    metrics["tp"] = float(tp)
    metrics["tn"] = float(tn)
    metrics["fp"] = float(fp)
    metrics["fn"] = float(fn)
    return metrics
