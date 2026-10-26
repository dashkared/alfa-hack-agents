"""Metric evaluation layer (deterministic, no LLM).

Internal convention: a metric is a float plus a direction.  `is_better(new, old)`
compares two metric values with the correct direction.
"""
from typing import Optional, Tuple

import numpy as np

try:  # optional deps are imported lazily where they are cheap
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        log_loss,
        mean_absolute_error,
        mean_squared_error,
        median_absolute_error,
        r2_score,
        roc_auc_score,
    )
    _SKLEARN_OK = True
except Exception:  # pragma: no cover
    _SKLEARN_OK = False


MAXIMIZE = "maximize"
MINIMIZE = "minimize"

# metric name -> direction
_METRIC_DIRECTION = {
    "rmse": MINIMIZE, "mae": MINIMIZE, "medae": MINIMIZE, "r2": MAXIMIZE,
    "mape": MINIMIZE, "rmsle": MINIMIZE, "max_error": MINIMIZE, "mse": MINIMIZE,
    "macro f1": MAXIMIZE, "f1": MAXIMIZE, "weighted f1": MAXIMIZE,
    "accuracy": MAXIMIZE, "balanced accuracy": MAXIMIZE, "roc-auc": MAXIMIZE,
    "auc": MAXIMIZE, "log loss": MINIMIZE, "logloss": MINIMIZE,
}

REGRESSION_METRICS = {"rmse", "mae", "medae", "r2", "mape", "rmsle", "max_error", "mse"}
CLASSIFICATION_METRICS = {"macro f1", "f1", "weighted f1", "accuracy", "balanced accuracy",
                          "roc-auc", "auc", "log loss", "logloss"}


def metric_direction(metric: str) -> str:
    return _METRIC_DIRECTION.get(str(metric).strip().lower(), MINIMIZE)


def is_better(new: float, old: Optional[float], direction: str, eps: float = 1e-12) -> bool:
    if old is None or not np.isfinite(old):
        return True
    if direction == MAXIMIZE:
        return new > old + eps
    return new < old - eps


def relative_gain(new: float, old: float, direction: str) -> float:
    """Signed relative improvement (positive = better)."""
    if not np.isfinite(old) or abs(old) < 1e-12:
        return 0.0
    d = (new - old) if direction == MAXIMIZE else (old - new)
    return float(d / (abs(old) + 1e-12))


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_metric(y_true, y_pred, metric: str, proba=None) -> float:
    """Compute the requested metric.  Higher/lower direction handled by caller."""
    metric = str(metric).strip().lower()
    y_true = np.asarray(y_true)
    if metric in ("macro f1", "f1"):
        labels = np.unique(np.concatenate([np.unique(y_true), np.unique(np.asarray(y_pred))])) \
            if _SKLEARN_OK else None
        avg = "macro" if metric == "macro f1" else "weighted"
        return float(f1_score(y_true, np.asarray(y_pred), average=avg, labels=labels,
                              zero_division=0))
    if metric == "weighted f1":
        labels = np.unique(np.concatenate([np.unique(y_true), np.unique(np.asarray(y_pred))]))
        return float(f1_score(y_true, np.asarray(y_pred), average="weighted", labels=labels,
                              zero_division=0))
    if metric == "accuracy":
        return float(accuracy_score(y_true, np.asarray(y_pred)))
    if metric == "balanced accuracy":
        return float(balanced_accuracy_score(y_true, np.asarray(y_pred)))
    if metric in ("roc-auc", "auc"):
        return _safe_auc(y_true, proba)
    if metric in ("log loss", "logloss"):
        return _safe_logloss(y_true, proba)
    # regression
    y_pred = np.asarray(y_pred, dtype=float)
    if metric == "rmse":
        return rmse(y_true, y_pred)
    if metric == "mse":
        return float(np.mean((np.asarray(y_true, float) - y_pred) ** 2))
    if metric == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    if metric == "medae":
        return float(median_absolute_error(y_true, y_pred))
    if metric == "r2":
        return float(r2_score(y_true, y_pred))
    if metric == "mape":
        denom = np.maximum(np.abs(np.asarray(y_true, float)), 1e-6)
        return float(np.mean(np.abs((np.asarray(y_true, float) - y_pred) / denom)) * 100)
    if metric == "rmsle":
        yt = np.log1p(np.clip(np.asarray(y_true, float), 0, None))
        yp = np.log1p(np.clip(y_pred, 0, None))
        return float(np.sqrt(np.mean((yt - yp) ** 2)))
    if metric == "max_error":
        return float(np.max(np.abs(np.asarray(y_true, float) - y_pred)))
    return rmse(y_true, y_pred)


def _safe_auc(y_true, proba) -> float:
    if proba is None:
        return 0.0
    try:
        proba = np.asarray(proba, dtype=float)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return float(roc_auc_score(y_true, proba[:, 1]))
        if proba.ndim == 2:
            return float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
        return float(roc_auc_score(y_true, proba))
    except Exception:
        return 0.0


def _safe_logloss(y_true, proba) -> float:
    if proba is None:
        return float("inf")
    try:
        proba = np.asarray(proba, dtype=float)
        labels = np.arange(proba.shape[1]) if proba.ndim == 2 else None
        return float(log_loss(y_true, proba, labels=labels))
    except Exception:
        return float("inf")


def classification_metric_from_proba(y_true, proba, metric: str) -> float:
    pred = np.argmax(np.asarray(proba), axis=1)
    return compute_metric(y_true, pred, metric, proba=proba)
