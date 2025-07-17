"""
Coulomb Counting SOC Estimator

Traditional SOC estimation method based on current integration with OCV correction.
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union
import warnings

from .base import BaseSOCEstimator


class CoulombCountingEstimator(BaseSOCEstimator):
    """
    Coulomb Counting SOC Estimator with OCV correction
    
    Features:
    - Current integration for SOC tracking
    - Periodic OCV-based correction
    - Efficiency factor consideration
    - Temperature compensation
    """
    
    def __init__(self, initial_soc: float = 0.8, efficiency: float = 0.98,
                 capacity: float = 2.5, correction_interval: int = 300,
                 ocv_correction: bool = True, **kwargs):
        
        super().__init__(name="Coulomb_Counting_Estimator", **kwargs)
        
        self.initial_soc = initial_soc
        self.efficiency = efficiency
        self.capacity = capacity  # Battery capacity in Ah
        self.correction_interval = correction_interval  # Steps between OCV corrections
        self.ocv_correction = ocv_correction
        self.dt = 1.0  # Default time step in seconds
        
        # OCV-SOC lookup table (can be customized)
        self.ocv_soc_table = np.array([
            [0.0, 3.0],
            [0.1, 3.2], 
            [0.2, 3.3],
            [0.4, 3.5],
            [0.6, 3.65],
            [0.8, 3.8],
            [1.0, 4.1]
        ])
    
    def _ocv_to_soc(self, ocv: float) -> float:
        """Convert OCV to SOC using lookup table"""
        return float(np.interp(ocv, self.ocv_soc_table[:, 1], self.ocv_soc_table[:, 0]))
    
    def _soc_to_ocv(self, soc: float) -> float:
        """Convert SOC to OCV using lookup table"""
        return float(np.interp(soc, self.ocv_soc_table[:, 0], self.ocv_soc_table[:, 1]))
    
    def fit(self, train_data: Dict[str, np.ndarray], 
            val_data: Optional[Dict[str, np.ndarray]] = None) -> None:
        """
        Fit the coulomb counting parameters
        
        This method can be used to optimize efficiency and capacity parameters
        based on training data if true SOC is available.
        """
        
        # Extract time step from data
        if 'time' in train_data and len(train_data['time']) > 1:
            self.dt = train_data['time'][1] - train_data['time'][0]
        
        # If true SOC is available, we could optimize parameters
        if 'soc_true' in train_data:
            # Simple parameter optimization could be implemented here
            # For now, we use the provided parameters
            pass
        
        self.is_fitted = True
    
    def predict(self, data: Dict[str, np.ndarray], 
                return_uncertainty: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict SOC using coulomb counting with optional OCV correction
        """
        
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before prediction")
        
        current = data['current']
        voltage = data.get('voltage', None)
        
        if self.ocv_correction and voltage is None:
            warnings.warn("OCV correction enabled but no voltage data provided")
            self.ocv_correction = False
        
        n_samples = len(current)
        soc_estimates = np.zeros(n_samples)
        uncertainties = np.zeros(n_samples) if return_uncertainty else None
        
        # Initialize SOC
        soc = self.initial_soc
        cumulative_error = 0.0  # Track cumulative error for uncertainty
        
        for i in range(n_samples):
            # Coulomb counting update
            # Negative current = discharge, positive = charge
            efficiency = self.efficiency if current[i] > 0 else 1.0  # Only apply efficiency to charging
            
            delta_soc = -current[i] * self.dt / 3600 / self.capacity * efficiency
            soc += delta_soc
            
            # Apply SOC bounds
            soc = np.clip(soc, 0, 1)
            
            # OCV correction
            if self.ocv_correction and i % self.correction_interval == 0 and voltage is not None:
                # Estimate SOC from OCV
                ocv_measured = voltage[i]
                soc_from_ocv = self._ocv_to_soc(ocv_measured)
                
                # Apply correction
                correction = soc_from_ocv - soc
                soc += correction
                soc = np.clip(soc, 0, 1)
                
                # Reset cumulative error after correction
                cumulative_error = 0.0
            else:
                # Accumulate uncertainty between corrections
                cumulative_error += abs(delta_soc) * 0.01  # 1% error per step
            
            soc_estimates[i] = soc
            
            if return_uncertainty:
                # Simple uncertainty model based on time since last correction
                base_uncertainty = 0.02  # 2% base uncertainty
                time_uncertainty = cumulative_error
                uncertainties[i] = np.sqrt(base_uncertainty**2 + time_uncertainty**2)
        
        if return_uncertainty:
            if uncertainties is not None:
                return soc_estimates, uncertainties
            else:
                return soc_estimates, np.zeros_like(soc_estimates)
        else:
            return soc_estimates


class CoulombCountingWithOCVCorrection(CoulombCountingEstimator):
    """
    Enhanced Coulomb Counting with advanced OCV correction strategies
    
    Additional features:
    - Adaptive correction intervals
    - Weighted correction based on current magnitude
    - Temperature-compensated OCV
    """
    
    def __init__(self, adaptive_correction: bool = True, 
                 current_threshold: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.name = "CC_with_OCV_Correction"
        self.adaptive_correction = adaptive_correction
        self.current_threshold = current_threshold  # Threshold for OCV validity
    
    def _is_ocv_valid(self, current: float, time_since_current: float = 0) -> bool:
        """
        Determine if OCV measurement is valid for correction
        
        Args:
            current: Current value
            time_since_current: Time since significant current flow
        """
        # OCV is valid when current is low and has been low for some time
        current_ok = abs(current) < self.current_threshold
        time_ok = time_since_current > 30  # 30 seconds of low current
        
        return current_ok and time_ok
    
    def predict(self, data: Dict[str, np.ndarray], 
                return_uncertainty: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Enhanced prediction with adaptive OCV correction"""
        
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before prediction")
        
        current = data['current']
        voltage = data.get('voltage', None)
        temperature = data.get('temperature', None)
        
        if self.ocv_correction and voltage is None:
            warnings.warn("OCV correction enabled but no voltage data provided")
            self.ocv_correction = False
        
        n_samples = len(current)
        soc_estimates = np.zeros(n_samples)
        uncertainties = np.zeros(n_samples) if return_uncertainty else None
        
        # Initialize
        soc = self.initial_soc
        cumulative_error = 0.0
        time_since_low_current = 0
        last_correction_step = 0
        
        for i in range(n_samples):
            # Track time since low current
            if abs(current[i]) < self.current_threshold:
                time_since_low_current += self.dt
            else:
                time_since_low_current = 0
            
            # Coulomb counting update
            efficiency = self.efficiency if current[i] > 0 else 1.0
            delta_soc = -current[i] * self.dt / 3600 / self.capacity * efficiency
            soc += delta_soc
            soc = np.clip(soc, 0, 1)
            
            # Adaptive OCV correction
            if self.ocv_correction and voltage is not None:
                correction_needed = False
                
                if self.adaptive_correction:
                    # Adaptive correction based on current and time
                    steps_since_correction = i - last_correction_step
                    
                    if (self._is_ocv_valid(current[i], time_since_low_current) and 
                        steps_since_correction > 60):  # At least 60 steps since last correction
                        correction_needed = True
                else:
                    # Fixed interval correction
                    if i % self.correction_interval == 0:
                        correction_needed = True
                
                if correction_needed:
                    # Temperature compensation (if available)
                    ocv_measured = voltage[i]
                    if temperature is not None:
                        # Simple temperature compensation (can be enhanced)
                        temp_coeff = -0.0005  # V/°C
                        temp_ref = 25  # Reference temperature
                        ocv_measured += temp_coeff * (temperature[i] - temp_ref)
                    
                    # SOC from OCV
                    soc_from_ocv = self._ocv_to_soc(ocv_measured)
                    
                    # Weighted correction based on confidence
                    confidence = min(time_since_low_current / 60, 1.0)  # Max confidence after 60s
                    correction = (soc_from_ocv - soc) * confidence
                    
                    soc += correction
                    soc = np.clip(soc, 0, 1)
                    
                    # Update tracking variables
                    cumulative_error *= (1 - confidence)  # Reduce uncertainty
                    last_correction_step = i
            
            # Update cumulative error
            if i > last_correction_step:
                cumulative_error += abs(delta_soc) * 0.01
            
            soc_estimates[i] = soc
            
            if return_uncertainty:
                # Enhanced uncertainty model
                base_uncertainty = 0.02
                time_uncertainty = cumulative_error
                current_uncertainty = abs(current[i]) * 0.001  # Higher uncertainty during high current
                
                total_uncertainty = np.sqrt(
                    base_uncertainty**2 + time_uncertainty**2 + current_uncertainty**2
                )
                uncertainties[i] = total_uncertainty
        
        if return_uncertainty:
            if uncertainties is not None:
                return soc_estimates, uncertainties
            else:
                return soc_estimates, np.zeros_like(soc_estimates)
        else:
            return soc_estimates 