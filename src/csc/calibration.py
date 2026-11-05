"""Calibration on held-out examples and application of a saved threshold."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import _validation as check
from .losses import _mean_loss_curve, severity_loss

_GRID = np.linspace(0.0, 1.0, 401, dtype=np.float64)
_GRID.setflags(write=False)


@dataclass(frozen=True)
class CalibratedCSC:
    """A calibrated threshold, fixed costs, and ordered label names.

    Create with calibrate() or load(). No calibration scores, targets, or sample
    identifiers are retained. predict() accepts scores only; evaluate() reports
    metrics without selecting or changing any calibration settings.
    """

    alpha: float
    beta: float
    grid_index: int
    n_calibration: int
    calibration_risk: float
    corrected_risk: float
    full_set_fallback: bool
    label_names: tuple[str, ...]
    cost_matrix: np.ndarray = field(repr=False, compare=False)

    def __post_init__(self):
        for name in ("alpha", "calibration_risk", "corrected_risk"):
            object.__setattr__(self, name, float(getattr(self, name)))
        object.__setattr__(self, "beta", check.beta(self.beta))
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        for name, value, low, high in (
            ("grid_index", self.grid_index, 0, 400),
            ("n_calibration", self.n_calibration, 1, np.inf),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
                raise ValueError(f"invalid {name}")
        if not 0 <= self.calibration_risk <= 1 or not 0 <= self.corrected_risk <= 1:
            raise ValueError("calibration risks must lie in [0, 1]")
        expected = (self.n_calibration * self.calibration_risk + 1) / (self.n_calibration + 1)
        if not np.isclose(self.corrected_risk, expected, rtol=0, atol=1e-12):
            raise ValueError("corrected_risk is inconsistent with the finite-sample correction")
        if not isinstance(self.full_set_fallback, bool):
            raise ValueError("full_set_fallback must be boolean")
        fallback_expected = self.alpha < 1.0 / (self.n_calibration + 1)
        if self.full_set_fallback != fallback_expected:
            raise ValueError("full_set_fallback is inconsistent with alpha and n_calibration")
        if self.full_set_fallback:
            if self.grid_index != 400 or self.calibration_risk != 0:
                raise ValueError("the fallback must select the full label set")
        elif self.corrected_risk > self.alpha:
            raise ValueError("selected threshold does not satisfy the calibration criterion")
        cost = np.array(check.cost(self.cost_matrix), copy=True)
        cost.setflags(write=False)
        object.__setattr__(self, "cost_matrix", cost)
        object.__setattr__(self, "label_names", check.names(self.label_names, len(cost)))

    @property
    def lambda_hat(self) -> float:
        """Inclusiveness parameter: larger lambda produces larger sets."""
        return float(_GRID[self.grid_index])

    @property
    def score_threshold(self) -> float:
        """Include all labels with score >= this cutoff, including ties."""
        return 1.0 - self.lambda_hat

    def predict(self, scores, *, label_names=None) -> np.ndarray:
        """Return a boolean (n_samples, n_labels) matrix in the fitted order."""
        values = check.scores(scores)
        if values.shape[1] != len(self.label_names):
            raise ValueError("score columns do not match the calibrated label vocabulary")
        if label_names is not None and check.names(label_names, values.shape[1]) != self.label_names:
            raise ValueError("label_names must match the calibration vocabulary and column order")
        return values >= self.score_threshold

    def predict_labels(self, scores, *, label_names=None) -> list[list[str]]:
        """Return label names for each prediction set."""
        selected = self.predict(scores, label_names=label_names)
        return [[self.label_names[j] for j in np.flatnonzero(row)] for row in selected]

    def evaluate(self, scores, targets, *, label_names=None) -> dict[str, float]:
        """Report mean severity, mean size, and sample-macro precision/recall/F1.

        Undefined precision, recall, and F1 (zero denominator) are set to zero.
        Empty target sets have severity zero, as in the calibration loss.
        """
        selected = self.predict(scores, label_names=label_names)
        truth = check.binary(targets, "targets")
        if truth.shape != selected.shape or not len(truth):
            raise ValueError("evaluation needs matching, nonempty scores and targets")
        loss = severity_loss(truth, selected, self.cost_matrix, beta=self.beta)
        tp = np.sum(selected & truth, axis=1, dtype=np.float64)
        size = np.sum(selected, axis=1, dtype=np.float64)
        count = np.sum(truth, axis=1, dtype=np.float64)

        def ratio(numerator, denominator):
            return np.divide(numerator, denominator, out=np.zeros_like(tp), where=denominator > 0)

        return {
            "severity_mean": float(np.mean(loss)),
            "set_size_mean": float(np.mean(size)),
            "precision_macro": float(np.mean(ratio(tp, size))),
            "recall_macro": float(np.mean(ratio(tp, count))),
            "f1_macro": float(np.mean(ratio(2 * tp, size + count))),
        }

    def save(self, path) -> None:
        """Write a portable JSON file containing settings and aggregate statistics."""
        payload = {
            "format": "csc-calibration",
            "format_version": 1,
            "alpha": self.alpha,
            "beta": self.beta,
            "grid_index": self.grid_index,
            "n_calibration": self.n_calibration,
            "calibration_risk": self.calibration_risk,
            "corrected_risk": self.corrected_risk,
            "full_set_fallback": self.full_set_fallback,
            "label_names": list(self.label_names),
            "cost_matrix": self.cost_matrix.tolist(),
        }
        Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path) -> CalibratedCSC:
        """Load a model saved by save(); no pickle or executable objects are used."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.pop("format", None) != "csc-calibration":
            raise ValueError("not a CSC calibration file")
        if payload.pop("format_version", None) != 1:
            raise ValueError("unsupported CSC calibration file version")
        try:
            return cls(**payload)
        except TypeError as exc:
            raise ValueError("invalid fields in CSC calibration file") from exc


def calibrate(scores, targets, cost_matrix, *, alpha: float = 0.25,
              beta: float = 0.0, label_names=None) -> CalibratedCSC:
    """Fit CSC on a held-out calibration partition.

    scores: finite matrix (n, K), bounded in [0, 1].
    targets: binary matrix (n, K), using the same label order.
    cost_matrix: fixed (K, K) costs in [0, 1], with zero diagonal.
    alpha: prespecified tolerance in (0, 1).
    beta: prespecified within-sample CVaR level in [0, 1); zero gives the mean.

    Select the first of 401 equally spaced lambdas in [0, 1] satisfying
    (sum(losses) + 1) / (n + 1) <= alpha. If none is feasible, select lambda=1.
    At prediction time, include label k exactly when score[k] >= 1 - lambda.

    Fix the predictor, vocabulary, costs, alpha, and beta before calibration.
    Calibration and future examples must be exchangeable at the sample unit.
    This function does not accept or inspect test data.
    """
    values = check.scores(scores)
    truth = check.binary(targets, "targets")
    cost = check.cost(cost_matrix)
    alpha = float(alpha)
    beta = check.beta(beta)
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    if not len(values) or truth.shape != values.shape or values.shape[1] != len(cost):
        raise ValueError("calibration needs nonempty scores and targets matching cost_matrix")
    label_names = check.names(label_names, len(cost))
    mean_loss = _mean_loss_curve(values, truth, cost, _GRID, beta)
    n = len(values)
    corrected = (n * mean_loss + 1.0) / (n + 1.0)
    feasible = np.flatnonzero(corrected <= alpha)
    index = int(feasible[0]) if len(feasible) else 400
    return CalibratedCSC(
        alpha=alpha, beta=beta, grid_index=index, n_calibration=n,
        calibration_risk=float(mean_loss[index]), corrected_risk=float(corrected[index]),
        full_set_fallback=not bool(len(feasible)), label_names=label_names, cost_matrix=cost,
    )
