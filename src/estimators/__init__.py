"""
SOC Estimation Module

This module contains various State of Charge (SOC) estimation algorithms:
- Traditional methods: Coulomb Counting, Extended Kalman Filter, Unscented Kalman Filter
- Machine Learning methods: LSTM, Transformer with uncertainty quantification
- Ensemble methods: Multi-model fusion with confidence-based switching
"""

from .coulomb_counting import CoulombCountingEstimator
from .kalman_filters import EKFEstimator, UKFEstimator
from .deep_learning import LSTMSOCEstimator, TransformerSOCEstimator
from .ensemble import EnsembleEstimator
from .base import BaseSOCEstimator

__all__ = [
    "BaseSOCEstimator",
    "CoulombCountingEstimator", 
    "EKFEstimator",
    "UKFEstimator",
    "LSTMSOCEstimator",
    "TransformerSOCEstimator", 
    "EnsembleEstimator"
] 