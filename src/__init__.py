"""
Advanced Battery State Estimation and Health Prediction Framework

This package provides comprehensive tools for:
- SOC (State of Charge) estimation using traditional and ML methods
- SOH (State of Health) prediction with multi-scale approaches
- Fault diagnosis and anomaly detection
- Real-time adaptive filtering
- Multi-physics battery modeling

Author: Luna Zhang
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Luna Zhang"

# Core imports
from . import battery_models
from . import estimators
from . import soh_prediction
from . import fault_diagnosis
from . import data_processing
from . import optimization
from . import utils

# Main classes for easy access
from .estimators import (
    CoulombCountingEstimator,
    EKFEstimator,
    UKFEstimator,
    LSTMSOCEstimator,
    TransformerSOCEstimator,
    EnsembleEstimator
)

from .battery_models import (
    EquivalentCircuitModel,
    TheveninModel,
    DualRCModel,
    PhysicsBasedModel
)

from .soh_prediction import (
    CapacityFadePredictor,
    ImpedanceAnalyzer,
    ICADVAAnalyzer,
    MLFusionPredictor,
    RULPredictor
)

from .fault_diagnosis import (
    AnomalyDetector,
    ThermalRunawayPredictor,
    FaultDiagnosisSystem
)

from .data_processing import (
    BatteryDataLoader,
    DataPreprocessor,
    FeatureExtractor
)

__all__ = [
    # Estimators
    "CoulombCountingEstimator",
    "EKFEstimator", 
    "UKFEstimator",
    "LSTMSOCEstimator",
    "TransformerSOCEstimator",
    "EnsembleEstimator",
    
    # Battery Models
    "EquivalentCircuitModel",
    "TheveninModel",
    "DualRCModel", 
    "PhysicsBasedModel",
    
    # SOH Prediction
    "CapacityFadePredictor",
    "ImpedanceAnalyzer",
    "ICADVAAnalyzer",
    "MLFusionPredictor",
    "RULPredictor",
    
    # Fault Diagnosis
    "AnomalyDetector",
    "ThermalRunawayPredictor",
    "FaultDiagnosisSystem",
    
    # Data Processing
    "BatteryDataLoader",
    "DataPreprocessor",
    "FeatureExtractor",
] 