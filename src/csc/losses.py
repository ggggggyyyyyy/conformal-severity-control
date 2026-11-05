"""Closest-match average and within-sample upper-tail severity."""

from __future__ import annotations

import numpy as np

from . import _validation as check


def _cvar(severities: np.ndarray, beta: float) -> np.ndarray:
    """Aggregate along axis zero, with exact fractional tail weighting."""
    if severities.shape[0] == 0:
        return np.zeros(severities.shape[1:], dtype=np.float64)
    if beta == 0:
        return np.mean(severities, axis=0)
    descending = np.sort(severities, axis=0)[::-1]
    mass = (1.0 - beta) * len(descending)
    if mass <= 1.0:
        return descending[0]
    whole = int(np.floor(mass))
    fraction = mass - whole
    total = np.sum(descending[:whole], axis=0)
    if fraction > 0 and whole < len(descending):
        total = total + fraction * descending[whole]
    return total / mass


def severity_loss(targets, selected, cost_matrix, *, beta: float = 0.0) -> np.ndarray:
    """Return one severity loss per sample.

    targets and selected are binary matrices with the same shape (n, K).
    cost_matrix[y, k] is the cost of matching true label y to predicted label k.
    An empty target has loss zero. For a nonempty target, an empty prediction
    has loss one. beta=0 gives the mean; beta>0 gives exact empirical CVaR.
    Costs must be fixed, finite, in [0, 1], and have a zero diagonal.
    """
    y = check.binary(targets, "targets")
    chosen = check.binary(selected, "selected")
    cost = check.cost(cost_matrix)
    beta = check.beta(beta)
    if y.shape != chosen.shape or y.shape[1] != len(cost):
        raise ValueError("targets, selected, and cost_matrix have incompatible shapes")
    losses = np.zeros(len(y), dtype=np.float64)
    for i, (truth, prediction) in enumerate(zip(y, chosen)):
        true_indices = np.flatnonzero(truth)
        if not len(true_indices):
            continue
        predicted_indices = np.flatnonzero(prediction)
        if not len(predicted_indices):
            losses[i] = 1.0
            continue
        severities = np.min(cost[np.ix_(true_indices, predicted_indices)], axis=1)
        losses[i] = _cvar(severities, beta)
    return losses


def _mean_loss_curve(scores, targets, cost, lambdas, beta):
    """Accumulate the grid mean without storing an n_samples x n_grid array."""
    totals = np.zeros(len(lambdas), dtype=np.float64)
    thresholds = 1.0 - lambdas
    for sample_scores, truth in zip(scores, targets):
        true_indices = np.flatnonzero(truth)
        if not len(true_indices):
            continue
        order = np.argsort(-sample_scores, kind="stable")
        counts = np.searchsorted(-sample_scores[order], -thresholds, side="right")
        cumulative = np.minimum.accumulate(cost[np.ix_(true_indices, order)], axis=1)
        severities = np.ones((len(true_indices), len(lambdas)), dtype=np.float64)
        nonempty = counts > 0
        severities[:, nonempty] = cumulative[:, counts[nonempty] - 1]
        totals += _cvar(severities, beta)
    return totals / len(scores)
