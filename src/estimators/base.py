"""
Base class for SOC estimators

Provides common interface and functionality for all SOC estimation algorithms.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
import time
from dataclasses import dataclass


@dataclass
class SOCEstimationResult:
    """Container for SOC estimation results with uncertainty quantification"""
    soc: np.ndarray
    uncertainty: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    computation_time: Optional[float] = None
    convergence_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseSOCEstimator(ABC):
    """
    Abstract base class for all SOC estimation algorithms
    
    This class defines the common interface that all SOC estimators must implement,
    ensuring consistency across different estimation methods.
    """
    
    def __init__(self, name: str = "BaseSOCEstimator", **kwargs):
        """
        Initialize the SOC estimator
        
        Args:
            name: Name of the estimator
            **kwargs: Additional configuration parameters
        """
        self.name = name
        self.config = kwargs
        self.is_fitted = False
        self.training_history = {}
        self.performance_metrics = {}
        
    @abstractmethod
    def fit(self, train_data: Dict[str, np.ndarray], 
            val_data: Optional[Dict[str, np.ndarray]] = None) -> None:
        """
        Train the SOC estimator on the provided data
        
        Args:
            train_data: Training dataset containing voltage, current, temperature, true_soc
            val_data: Optional validation dataset for hyperparameter tuning
        """
        pass
    
    @abstractmethod
    def predict(self, data: Dict[str, np.ndarray], 
                return_uncertainty: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict SOC values for the given data
        
        Args:
            data: Input data containing voltage, current, temperature
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            SOC predictions and optionally uncertainty estimates
        """
        pass
    
    def estimate_soc(self, data: Dict[str, np.ndarray], 
                     return_full_result: bool = False) -> Union[np.ndarray, SOCEstimationResult]:
        """
        High-level SOC estimation method with timing and metadata
        
        Args:
            data: Input data containing voltage, current, temperature
            return_full_result: Whether to return full SOCEstimationResult object
            
        Returns:
            SOC estimates or full result object
        """
        start_time = time.time()
        
        if hasattr(self, 'predict'):
            uncertainty_enabled = getattr(self, 'uncertainty_quantification', False)
            if uncertainty_enabled:
                result = self.predict(data, return_uncertainty=True)
                if isinstance(result, tuple):
                    soc, uncertainty = result
                else:
                    soc, uncertainty = result, None
            else:
                result = self.predict(data, return_uncertainty=False)
                if isinstance(result, tuple):
                    soc = result[0]
                else:
                    soc = result
                uncertainty = None
        else:
            raise NotImplementedError("Estimator must implement predict method")
        
        computation_time = time.time() - start_time
        
        if return_full_result:
            return SOCEstimationResult(
                soc=soc,
                uncertainty=uncertainty,
                computation_time=computation_time,
                metadata={"estimator": self.name, "config": self.config}
            )
        else:
            return soc
    
    def evaluate(self, test_data: Dict[str, np.ndarray], 
                 metrics: Optional[list] = None) -> Dict[str, float]:
        """
        Evaluate the estimator performance on test data
        
        Args:
            test_data: Test dataset with true SOC values
            metrics: List of metrics to compute
            
        Returns:
            Dictionary of metric names and values
        """
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before evaluation")
        
        if metrics is None:
            metrics = ['rmse', 'mae', 'mape', 'max_error']
        
        soc_pred = self.predict(test_data, return_uncertainty=False)
        soc_true = test_data['soc_true']
        
        results = {}
        
        if 'rmse' in metrics:
            results['rmse'] = np.sqrt(np.mean((soc_pred - soc_true) ** 2))
        
        if 'mae' in metrics:
            results['mae'] = np.mean(np.abs(soc_pred - soc_true))
        
        if 'mape' in metrics:
            results['mape'] = np.mean(np.abs((soc_pred - soc_true) / soc_true)) * 100
        
        if 'max_error' in metrics:
            results['max_error'] = np.max(np.abs(soc_pred - soc_true))
        
        if 'r2_score' in metrics:
            ss_res = np.sum((soc_true - soc_pred) ** 2)
            ss_tot = np.sum((soc_true - np.mean(soc_true)) ** 2)
            results['r2_score'] = 1 - (ss_res / ss_tot)
        
        self.performance_metrics.update(results)
        return results
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk
        
        Args:
            filepath: Path to save the model
        """
        import pickle
        
        model_data = {
            'name': self.name,
            'config': self.config,
            'is_fitted': self.is_fitted,
            'performance_metrics': self.performance_metrics,
            'training_history': self.training_history
        }
        
        # Add model-specific data
        if hasattr(self, 'get_model_state'):
            model_data['model_state'] = getattr(self, 'get_model_state')()
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk
        
        Args:
            filepath: Path to the saved model
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.name = model_data['name']
        self.config = model_data['config']
        self.is_fitted = model_data['is_fitted']
        self.performance_metrics = model_data['performance_metrics']
        self.training_history = model_data['training_history']
        
        # Load model-specific data
        if 'model_state' in model_data and hasattr(self, 'set_model_state'):
            getattr(self, 'set_model_state')(model_data['model_state'])
    
    def get_config(self) -> Dict[str, Any]:
        """Get the estimator configuration"""
        return self.config.copy()
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """Update the estimator configuration"""
        self.config.update(config)
    
    def reset(self) -> None:
        """Reset the estimator to initial state"""
        self.is_fitted = False
        self.training_history = {}
        self.performance_metrics = {}
    
    def __str__(self) -> str:
        """String representation of the estimator"""
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.name} ({status})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"{self.__class__.__name__}(name='{self.name}', config={self.config})" 