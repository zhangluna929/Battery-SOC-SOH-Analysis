"""
Ensemble SOC Estimator

Combines multiple SOC estimation methods for improved accuracy and robustness.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

from .base import BaseSOCEstimator


class EnsembleEstimator(BaseSOCEstimator):
    """
    Ensemble SOC Estimator that combines multiple estimation methods
    
    Features:
    - Multiple base estimators combination
    - Performance-based weighting
    - Confidence-based switching
    - Uncertainty quantification from ensemble variance
    """
    
    def __init__(self, estimators: List[BaseSOCEstimator], 
                 combination_method: str = "weighted_average",
                 weights: Optional[List[float]] = None,
                 confidence_threshold: float = 0.8,
                 **kwargs):
        
        super().__init__(name="Ensemble_SOC_Estimator", **kwargs)
        
        self.estimators = estimators
        self.combination_method = combination_method
        self.weights = weights
        self.confidence_threshold = confidence_threshold
        self.estimator_weights = None
        self.estimator_performance = {}
        
    def fit(self, train_data: Dict[str, np.ndarray], 
            val_data: Optional[Dict[str, np.ndarray]] = None) -> None:
        """
        Fit all base estimators and compute their weights
        """
        
        # Fit all base estimators
        for estimator in self.estimators:
            estimator.fit(train_data, val_data)
        
        # Compute performance-based weights if validation data is available
        if val_data is not None and 'soc_true' in val_data:
            self._compute_performance_weights(val_data)
        else:
            # Use equal weights
            self.estimator_weights = np.ones(len(self.estimators)) / len(self.estimators)
        
        self.is_fitted = True
    
    def _compute_performance_weights(self, val_data: Dict[str, np.ndarray]) -> None:
        """
        Compute performance-based weights for each estimator
        """
        performances = []
        
        for estimator in self.estimators:
            # Get predictions
            predictions = estimator.predict(val_data, return_uncertainty=False)
            true_soc = val_data['soc_true']
            
            # Compute RMSE
            rmse = np.sqrt(np.mean((predictions - true_soc) ** 2))
            mae = np.mean(np.abs(predictions - true_soc))
            
            # Store performance metrics
            self.estimator_performance[estimator.name] = {
                'rmse': rmse,
                'mae': mae
            }
            
            # Use inverse RMSE as performance score (higher is better)
            performance_score = 1.0 / (rmse + 1e-6)
            performances.append(performance_score)
        
        # Normalize to get weights
        performances = np.array(performances)
        self.estimator_weights = performances / np.sum(performances)
    
    def predict(self, data: Dict[str, np.ndarray], 
                return_uncertainty: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict SOC using ensemble of estimators
        """
        
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        # Get predictions from all estimators
        predictions = []
        uncertainties = []
        
        for estimator in self.estimators:
            if hasattr(estimator, 'uncertainty_quantification') and estimator.uncertainty_quantification:
                try:
                    pred, unc = estimator.predict(data, return_uncertainty=True)
                    predictions.append(pred)
                    uncertainties.append(unc)
                except:
                    pred = estimator.predict(data, return_uncertainty=False)
                    predictions.append(pred)
                    uncertainties.append(np.zeros_like(pred))
            else:
                pred = estimator.predict(data, return_uncertainty=False)
                predictions.append(pred)
                uncertainties.append(np.zeros_like(pred))
        
        predictions = np.array(predictions)
        uncertainties = np.array(uncertainties)
        
        # Combine predictions based on method
        if self.combination_method == "simple_average":
            ensemble_prediction = np.mean(predictions, axis=0)
            
        elif self.combination_method == "weighted_average":
            if self.weights is not None:
                weights = np.array(self.weights)
            else:
                weights = self.estimator_weights
            
            ensemble_prediction = np.average(predictions, axis=0, weights=weights)
            
        elif self.combination_method == "confidence_based":
            ensemble_prediction = self._confidence_based_combination(predictions, uncertainties)
            
        elif self.combination_method == "median":
            ensemble_prediction = np.median(predictions, axis=0)
            
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")
        
        if return_uncertainty:
            # Ensemble uncertainty from prediction variance
            ensemble_uncertainty = self._compute_ensemble_uncertainty(predictions, uncertainties)
            return ensemble_prediction, ensemble_uncertainty
        else:
            return ensemble_prediction
    
    def _confidence_based_combination(self, predictions: np.ndarray, 
                                    uncertainties: np.ndarray) -> np.ndarray:
        """
        Combine predictions based on confidence (inverse uncertainty)
        """
        n_estimators, n_samples = predictions.shape
        ensemble_prediction = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Compute confidence weights (inverse uncertainty)
            sample_uncertainties = uncertainties[:, i]
            sample_uncertainties = np.maximum(sample_uncertainties, 1e-6)  # Avoid division by zero
            
            confidences = 1.0 / sample_uncertainties
            weights = confidences / np.sum(confidences)
            
            # Weighted combination
            ensemble_prediction[i] = np.sum(weights * predictions[:, i])
        
        return ensemble_prediction
    
    def _compute_ensemble_uncertainty(self, predictions: np.ndarray, 
                                    uncertainties: np.ndarray) -> np.ndarray:
        """
        Compute ensemble uncertainty combining prediction variance and individual uncertainties
        """
        # Variance among predictions (epistemic uncertainty)
        prediction_variance = np.var(predictions, axis=0)
        
        # Average of individual uncertainties (aleatoric uncertainty)
        if self.weights is not None:
            weights = np.array(self.weights)
        else:
            weights = self.estimator_weights
        
        avg_uncertainty = np.average(uncertainties**2, axis=0, weights=weights)
        
        # Total uncertainty
        total_uncertainty = np.sqrt(prediction_variance + avg_uncertainty)
        
        return total_uncertainty
    
    def get_estimator_weights(self) -> Dict[str, float]:
        """Get the current weights for each estimator"""
        if self.estimator_weights is not None:
            return {est.name: weight for est, weight in zip(self.estimators, self.estimator_weights)}
        else:
            return {est.name: 1.0/len(self.estimators) for est in self.estimators}
    
    def get_performance_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for each estimator"""
        return self.estimator_performance.copy()
    
    def predict_individual(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Get predictions from each individual estimator
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        individual_predictions = {}
        
        for estimator in self.estimators:
            try:
                pred = estimator.predict(data, return_uncertainty=False)
                individual_predictions[estimator.name] = pred
            except Exception as e:
                warnings.warn(f"Failed to get prediction from {estimator.name}: {e}")
        
        return individual_predictions
    
    def add_estimator(self, estimator: BaseSOCEstimator, weight: Optional[float] = None) -> None:
        """
        Add a new estimator to the ensemble
        """
        self.estimators.append(estimator)
        
        if self.estimator_weights is not None:
            if weight is not None:
                # Add the new weight and renormalize
                new_weights = np.append(self.estimator_weights, weight)
                self.estimator_weights = new_weights / np.sum(new_weights)
            else:
                # Assign equal weight and renormalize
                n_estimators = len(self.estimators)
                self.estimator_weights = np.ones(n_estimators) / n_estimators
        
        # If already fitted, fit the new estimator with stored data
        if self.is_fitted and hasattr(self, '_last_train_data'):
            estimator.fit(self._last_train_data)
    
    def remove_estimator(self, estimator_name: str) -> None:
        """
        Remove an estimator from the ensemble
        """
        for i, estimator in enumerate(self.estimators):
            if estimator.name == estimator_name:
                self.estimators.pop(i)
                if self.estimator_weights is not None:
                    self.estimator_weights = np.delete(self.estimator_weights, i)
                    if len(self.estimator_weights) > 0:
                        self.estimator_weights = self.estimator_weights / np.sum(self.estimator_weights)
                break
        else:
            raise ValueError(f"Estimator {estimator_name} not found in ensemble")
    
    def evaluate_ensemble_diversity(self, data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Evaluate the diversity of predictions among ensemble members
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before evaluation")
        
        # Get individual predictions
        individual_preds = self.predict_individual(data)
        predictions = np.array(list(individual_preds.values()))
        
        # Compute diversity metrics
        mean_pairwise_distance = 0.0
        n_pairs = 0
        
        for i in range(len(predictions)):
            for j in range(i+1, len(predictions)):
                distance = np.mean(np.abs(predictions[i] - predictions[j]))
                mean_pairwise_distance += distance
                n_pairs += 1
        
        if n_pairs > 0:
            mean_pairwise_distance /= n_pairs
        
        # Prediction variance
        prediction_variance = np.mean(np.var(predictions, axis=0))
        
        # Disagreement rate (percentage of time predictions differ by more than threshold)
        threshold = 0.05  # 5% SOC difference
        disagreements = 0
        n_samples = predictions.shape[1]
        
        for i in range(n_samples):
            sample_preds = predictions[:, i]
            if np.max(sample_preds) - np.min(sample_preds) > threshold:
                disagreements += 1
        
        disagreement_rate = disagreements / n_samples
        
        return {
            'mean_pairwise_distance': mean_pairwise_distance,
            'prediction_variance': prediction_variance,
            'disagreement_rate': disagreement_rate
        } 