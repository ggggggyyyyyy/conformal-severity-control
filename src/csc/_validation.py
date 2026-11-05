"""Input checks shared by calibration and evaluation."""

from __future__ import annotations

import numpy as np


def matrix(value, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain real numbers")
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError(f"{name} must have shape (n_samples, n_labels), n_labels > 0")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def scores(value) -> np.ndarray:
    array = matrix(value, "scores")
    if np.any((array < 0) | (array > 1)):
        raise ValueError("scores must lie in [0, 1]; convert logits before calibration")
    return array


def binary(value, name: str) -> np.ndarray:
    array = matrix(value, name)
    if np.any((array != 0) & (array != 1)):
        raise ValueError(f"{name} must contain only 0 and 1")
    return array.astype(bool)


def cost(value) -> np.ndarray:
    array = matrix(value, "cost_matrix")
    if array.shape[0] != array.shape[1]:
        raise ValueError("cost_matrix must be square")
    if np.any((array < 0) | (array > 1)):
        raise ValueError("cost_matrix entries must lie in [0, 1]")
    if np.any(np.diag(array) != 0):
        raise ValueError("cost_matrix must have a zero diagonal")
    return array


def beta(value: float) -> float:
    value = float(value)
    if not 0 <= value < 1:
        raise ValueError("beta must lie in [0, 1)")
    return value


def names(value, n_labels: int) -> tuple[str, ...]:
    if value is None:
        return tuple(str(i) for i in range(n_labels))
    if isinstance(value, str):
        raise ValueError("label_names must be a sequence of unique strings")
    result = tuple(value)
    if len(result) != n_labels or any(not isinstance(x, str) or not x.strip() for x in result):
        raise ValueError("label_names must contain one nonempty string per score column")
    if len(set(result)) != len(result):
        raise ValueError("label_names must be unique")
    return result
