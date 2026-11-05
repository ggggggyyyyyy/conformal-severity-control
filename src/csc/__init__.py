"""Conformal severity control for multi-label prediction."""

from .calibration import CalibratedCSC, calibrate
from .costs import atc_cost_matrix, hierarchy_cost_matrix
from .losses import severity_loss

__version__ = "0.1.0"
__all__ = [
    "CalibratedCSC",
    "calibrate",
    "atc_cost_matrix",
    "hierarchy_cost_matrix",
    "severity_loss",
]
