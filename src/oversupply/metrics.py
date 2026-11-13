"""Forecast evaluation. All arrays are cells x target hours."""

from __future__ import annotations

import numpy as np

# Flags per hour when ranking cells: about as many cells as typically reach each level.
K_BY_LEVEL = {10: 60, 20: 10, 40: 3}


def average_precision(score: np.ndarray, positive: np.ndarray) -> float:
    """Area under the precision-recall curve, for rare events."""
    n_positive = int(positive.sum())
    if n_positive == 0:
        return float("nan")
    order = np.argsort(-score, kind="stable")
    hits = np.cumsum(positive[order])
    precision = hits / np.arange(1, len(order) + 1)
    return float(precision[positive[order]].sum() / n_positive)


def blend(first: np.ndarray, second: np.ndarray, weight_first: float) -> np.ndarray:
    """Weighted geometric mean of two forecasts of a count (zero stays near zero)."""
    eps = 1e-3
    return np.exp(weight_first * np.log(first + eps)
                  + (1 - weight_first) * np.log(second + eps)) - eps


def best_cutoff(
    pred: np.ndarray, positive: np.ndarray, candidates: np.ndarray
) -> tuple[float, float]:
    """The forecast value above which to flag a cell, maximizing F1. Returns (cutoff, F1)."""
    total = int(positive.sum())
    best, best_f1 = float(candidates[0]), -1.0
    for cutoff in candidates:
        flagged = pred >= cutoff
        hits = int((flagged & positive).sum())
        f1 = 2 * hits / max(int(flagged.sum()) + total, 1)
        if f1 > best_f1:
            best, best_f1 = float(cutoff), f1
    return best, best_f1


def top_flags(score: np.ndarray, allowed: np.ndarray, k: int) -> np.ndarray:
    """Boolean flags on each hour's ``k`` highest-scoring allowed cells."""
    flags = np.zeros_like(allowed, dtype=bool)
    for hour in range(score.shape[1]):
        s = np.where(allowed[:, hour], score[:, hour], -np.inf)
        top = np.argpartition(-s, k - 1)[:k]
        flags[top[np.isfinite(s[top])], hour] = True
    return flags


def top_k(score: np.ndarray, positive: np.ndarray, k: int,
          allowed: np.ndarray | None = None) -> tuple[float, float]:
    """Precision and recall when each hour's ``k`` highest-scoring cells are flagged.

    ``allowed`` restricts both the flags and the events, e.g. to cells not yet crowded.
    """
    hits = flagged = 0
    positive = positive if allowed is None else positive & allowed
    for hour in range(score.shape[1]):
        s = score[:, hour] if allowed is None else np.where(allowed[:, hour], score[:, hour],
                                                            -np.inf)
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.isfinite(s[top])]
        hits += int(positive[top, hour].sum())
        flagged += len(top)
    total = int(positive.sum())
    return hits / max(flagged, 1), hits / max(total, 1)


def evaluate(y: np.ndarray, pred: np.ndarray, origin: np.ndarray,
             levels: tuple[int, ...], k_by_level: dict[int, int]) -> dict[str, float]:
    """Accuracy overall, on busy cells, per level and for new pile-ups.

    A new pile-up is a cell below the level at forecast time that reaches it at the target hour.
    """
    y = y.astype(np.float32)
    errors = np.abs(y - pred)
    busy = y >= min(levels)
    out = {"mae": float(errors.mean()), "mae_busy": float(errors[busy].mean())}
    flat_pred = pred.ravel()
    for level in levels:
        positive = y >= level
        flagged = pred >= level
        hits = int((positive & flagged).sum())
        out[f"precision_{level}"] = hits / max(int(flagged.sum()), 1)
        out[f"recall_{level}"] = hits / max(int(positive.sum()), 1)
        out[f"ap_{level}"] = average_precision(flat_pred, positive.ravel())
        k = k_by_level[level]
        out[f"precision_top{k}_{level}"], out[f"recall_top{k}_{level}"] = top_k(pred, positive, k)
        below = origin < level
        out[f"new_share_{level}"] = float((positive & below).sum() / max(int(positive.sum()), 1))
        out[f"new_precision_top{k}_{level}"], out[f"new_recall_top{k}_{level}"] = top_k(
            pred, positive, k, allowed=below)
    return out
