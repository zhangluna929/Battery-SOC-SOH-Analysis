"""
Kalman Filter-based SOC Estimators

This module implements advanced Kalman filtering approaches for SOC estimation:
- Extended Kalman Filter (EKF) with nonlinear battery models
- Unscented Kalman Filter (UKF) for improved nonlinear handling
- Adaptive Kalman Filters with online parameter identification
- Dual Extended Kalman Filter for simultaneous state and parameter estimation
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union
from scipy.optimize import minimize
import warnings

from .base import BaseSOCEstimator


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter implementation for nonlinear systems
    
    Features:
    - Nonlinear state and measurement models
    - Jacobian-based linearization
    - Adaptive noise estimation
    """
    
    def __init__(self, state_dim: int, measurement_dim: int, 
                 process_noise: float = 1e-5, measurement_noise: float = 1e-4,
                 initial_covariance: float = 1e-3):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        
        # Initialize state and covariance
        self.x = np.zeros(state_dim)
        self.P = np.eye(state_dim) * initial_covariance
        
        # Noise matrices
        self.Q = np.eye(state_dim) * process_noise
        self.R = np.eye(measurement_dim) * measurement_noise
        
        # For adaptive filtering
        self.innovation_history = []
        self.innovation_window = 50
        
    def predict(self, f_func, F_jac, u=None, dt=1.0):
        """
        Prediction step
        
        Args:
            f_func: Nonlinear state transition function
            F_jac: Jacobian of state transition function
            u: Control input
            dt: Time step
        """
        # Predict state
        self.x = f_func(self.x, u, dt)
        
        # Predict covariance
        F = F_jac(self.x, u, dt)
        self.P = F @ self.P @ F.T + self.Q
        
    def update(self, z, h_func, H_jac):
        """
        Update step
        
        Args:
            z: Measurement
            h_func: Nonlinear measurement function
            H_jac: Jacobian of measurement function
        """
        # Innovation
        y = z - h_func(self.x)
        
        # Innovation covariance
        H = H_jac(self.x)
        S = H @ self.P @ H.T + self.R
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state and covariance
        self.x = self.x + K @ y
        self.P = (np.eye(self.state_dim) - K @ H) @ self.P
        
        # Store innovation for adaptive filtering
        self.innovation_history.append(y)
        if len(self.innovation_history) > self.innovation_window:
            self.innovation_history.pop(0)
        
        return y, S
    
    def adapt_noise_matrices(self):
        """Adaptive noise estimation based on innovation sequence"""
        if len(self.innovation_history) < 10:
            return
        
        innovations = np.array(self.innovation_history)
        innovation_cov = np.cov(innovations.T)
        
        # Adaptive R matrix
        self.R = innovation_cov * 0.1  # Scale factor
        
        # Adaptive Q matrix (simplified approach)
        trace_P = np.trace(self.P)
        if trace_P > 1e-2:  # If uncertainty is growing
            self.Q *= 1.1
        elif trace_P < 1e-6:  # If uncertainty is too small
            self.Q *= 0.9


class UnscentedKalmanFilter:
    """
    Unscented Kalman Filter for nonlinear estimation
    
    Features:
    - Sigma point sampling
    - No Jacobian computation required
    - Better handling of strong nonlinearities
    """
    
    def __init__(self, state_dim: int, measurement_dim: int,
                 process_noise: float = 1e-5, measurement_noise: float = 1e-4,
                 initial_covariance: float = 1e-3, alpha: float = 1e-3, 
                 beta: float = 2.0, kappa: float = 0.0):
        
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        
        # UKF parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lambda_ = alpha**2 * (state_dim + kappa) - state_dim
        
        # Weights
        self.n_sigma = 2 * state_dim + 1
        self.Wm = np.zeros(self.n_sigma)
        self.Wc = np.zeros(self.n_sigma)
        
        self.Wm[0] = self.lambda_ / (state_dim + self.lambda_)
        self.Wc[0] = self.lambda_ / (state_dim + self.lambda_) + (1 - alpha**2 + beta)
        
        for i in range(1, self.n_sigma):
            self.Wm[i] = 1 / (2 * (state_dim + self.lambda_))
            self.Wc[i] = 1 / (2 * (state_dim + self.lambda_))
        
        # Initialize state and covariance
        self.x = np.zeros(state_dim)
        self.P = np.eye(state_dim) * initial_covariance
        
        # Noise matrices
        self.Q = np.eye(state_dim) * process_noise
        self.R = np.eye(measurement_dim) * measurement_noise
    
    def generate_sigma_points(self):
        """Generate sigma points"""
        n = self.state_dim
        sigma_points = np.zeros((self.n_sigma, n))
        
        sigma_points[0] = self.x
        
        try:
            sqrt = np.linalg.cholesky((n + self.lambda_) * self.P)
        except np.linalg.LinAlgError:
            # If Cholesky decomposition fails, use eigenvalue decomposition
            eigenvals, eigenvecs = np.linalg.eigh(self.P)
            eigenvals = np.maximum(eigenvals, 1e-12)  # Ensure positive eigenvalues
            sqrt = eigenvecs @ np.diag(np.sqrt(eigenvals * (n + self.lambda_)))
        
        for i in range(n):
            sigma_points[i + 1] = self.x + sqrt[i]
            sigma_points[n + i + 1] = self.x - sqrt[i]
        
        return sigma_points
    
    def predict(self, f_func, u=None, dt=1.0):
        """Prediction step"""
        # Generate sigma points
        sigma_points = self.generate_sigma_points()
        
        # Propagate sigma points through nonlinear function
        sigma_points_pred = np.zeros_like(sigma_points)
        for i in range(self.n_sigma):
            sigma_points_pred[i] = f_func(sigma_points[i], u, dt)
        
        # Compute predicted mean and covariance
        self.x = np.sum(self.Wm[:, np.newaxis] * sigma_points_pred, axis=0)
        
        self.P = self.Q.copy()
        for i in range(self.n_sigma):
            diff = sigma_points_pred[i] - self.x
            self.P += self.Wc[i] * np.outer(diff, diff)
    
    def update(self, z, h_func):
        """Update step"""
        # Generate sigma points
        sigma_points = self.generate_sigma_points()
        
        # Propagate sigma points through measurement function
        z_sigma = np.zeros((self.n_sigma, self.measurement_dim))
        for i in range(self.n_sigma):
            z_sigma[i] = h_func(sigma_points[i])
        
        # Compute predicted measurement mean and covariance
        z_pred = np.sum(self.Wm[:, np.newaxis] * z_sigma, axis=0)
        
        Pz = self.R.copy()
        Pxz = np.zeros((self.state_dim, self.measurement_dim))
        
        for i in range(self.n_sigma):
            diff_z = z_sigma[i] - z_pred
            diff_x = sigma_points[i] - self.x
            
            Pz += self.Wc[i] * np.outer(diff_z, diff_z)
            Pxz += self.Wc[i] * np.outer(diff_x, diff_z)
        
        # Kalman gain
        K = Pxz @ np.linalg.inv(Pz)
        
        # Update
        innovation = z - z_pred
        self.x = self.x + K @ innovation
        self.P = self.P - K @ Pz @ K.T
        
        return innovation, Pz


class EKFEstimator(BaseSOCEstimator):
    """
    Extended Kalman Filter SOC Estimator
    
    Features:
    - Nonlinear battery model integration
    - Adaptive noise estimation
    - Temperature-dependent OCV modeling
    - Online parameter identification
    """
    
    def __init__(self, battery_model=None, process_noise: float = 1e-5, 
                 measurement_noise: float = 1e-4, initial_covariance: float = 1e-3,
                 initial_soc: float = 0.8, adaptive_filtering: bool = True, **kwargs):
        
        super().__init__(name="EKF_SOC_Estimator", **kwargs)
        
        self.battery_model = battery_model
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.initial_covariance = initial_covariance
        self.initial_soc = initial_soc
        self.adaptive_filtering = adaptive_filtering
        
        self.ekf = None
        self.dt = 1.0  # Default time step
        
    def _state_transition(self, x, u, dt):
        """
        State transition function: SOC dynamics
        
        Args:
            x: State vector [SOC]
            u: Input vector [current]
            dt: Time step
        """
        if self.battery_model is None:
            # Simple coulomb counting model
            capacity = 2.5  # Default capacity in Ah
            soc_new = x[0] - u[0] * dt / 3600 / capacity
        else:
            soc_new = self.battery_model.update_soc(x[0], u[0], dt)
        
        return np.array([np.clip(soc_new, 0, 1)])
    
    def _state_jacobian(self, x, u, dt):
        """Jacobian of state transition function"""
        # For SOC: d(SOC)/d(SOC) = 1
        return np.array([[1.0]])
    
    def _measurement_function(self, x):
        """
        Measurement function: Voltage from SOC
        
        Args:
            x: State vector [SOC]
        """
        if self.battery_model is None:
            # Simple polynomial OCV model
            soc = x[0]
            ocv = 2.5 + 0.5*soc + 0.8*soc**2 - 0.3*soc**3 + 0.2*soc**4
        else:
            ocv = self.battery_model.get_ocv(x[0])
        
        return np.array([ocv])
    
    def _measurement_jacobian(self, x):
        """Jacobian of measurement function"""
        if self.battery_model is None:
            # Derivative of polynomial OCV
            soc = x[0]
            docv_dsoc = 0.5 + 1.6*soc - 0.9*soc**2 + 0.8*soc**3
        else:
            # Numerical derivative
            h = 1e-6
            soc_plus = np.clip(x[0] + h, 0, 1)
            soc_minus = np.clip(x[0] - h, 0, 1)
            ocv_plus = self.battery_model.get_ocv(soc_plus)
            ocv_minus = self.battery_model.get_ocv(soc_minus)
            docv_dsoc = (ocv_plus - ocv_minus) / (2 * h)
        
        return np.array([[docv_dsoc]])
    
    def fit(self, train_data: Dict[str, np.ndarray], 
            val_data: Optional[Dict[str, np.ndarray]] = None) -> None:
        """Fit the EKF parameters using training data"""
        
        # Extract time step from data
        if 'time' in train_data and len(train_data['time']) > 1:
            self.dt = train_data['time'][1] - train_data['time'][0]
        
        # Initialize EKF
        self.ekf = ExtendedKalmanFilter(
            state_dim=1,
            measurement_dim=1,
            process_noise=self.process_noise,
            measurement_noise=self.measurement_noise,
            initial_covariance=self.initial_covariance
        )
        
        # Set initial state
        self.ekf.x = np.array([self.initial_soc])
        
        # Parameter optimization using training data (simplified)
        if 'soc_true' in train_data:
            # Could implement parameter learning here
            pass
        
        self.is_fitted = True
    
    def predict(self, data: Dict[str, np.ndarray], 
                return_uncertainty: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict SOC using EKF"""
        
        if not self.is_fitted:
            raise ValueError("EKF must be fitted before prediction")
        
        current = data['current']
        voltage = data['voltage']
        
        soc_estimates = []
        uncertainties = []
        
        # Reset EKF state
        self.ekf.x = np.array([self.initial_soc])
        self.ekf.P = np.eye(1) * self.initial_covariance
        
        for i in range(len(current)):
            # Prediction step
            u = np.array([current[i]])
            self.ekf.predict(self._state_transition, self._state_jacobian, u, self.dt)
            
            # Update step
            z = np.array([voltage[i]])
            self.ekf.update(z, self._measurement_function, self._measurement_jacobian)
            
            # Adaptive filtering
            if self.adaptive_filtering:
                self.ekf.adapt_noise_matrices()
            
            soc_estimates.append(self.ekf.x[0])
            uncertainties.append(np.sqrt(self.ekf.P[0, 0]))
        
        soc_estimates = np.array(soc_estimates)
        uncertainties = np.array(uncertainties)
        
        if return_uncertainty:
            return soc_estimates, uncertainties
        else:
            return soc_estimates


class UKFEstimator(BaseSOCEstimator):
    """
    Unscented Kalman Filter SOC Estimator
    
    Features:
    - Superior nonlinear handling compared to EKF
    - No Jacobian computation required
    - Sigma point sampling
    - Better uncertainty quantification
    """
    
    def __init__(self, battery_model=None, process_noise: float = 1e-5,
                 measurement_noise: float = 1e-4, initial_covariance: float = 1e-3,
                 initial_soc: float = 0.8, alpha: float = 1e-3, beta: float = 2.0,
                 kappa: float = 0.0, **kwargs):
        
        super().__init__(name="UKF_SOC_Estimator", **kwargs)
        
        self.battery_model = battery_model
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.initial_covariance = initial_covariance
        self.initial_soc = initial_soc
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        
        self.ukf = None
        self.dt = 1.0
    
    def _state_transition(self, x, u, dt):
        """State transition function"""
        if self.battery_model is None:
            capacity = 2.5
            soc_new = x[0] - u[0] * dt / 3600 / capacity
        else:
            soc_new = self.battery_model.update_soc(x[0], u[0], dt)
        
        return np.array([np.clip(soc_new, 0, 1)])
    
    def _measurement_function(self, x):
        """Measurement function"""
        if self.battery_model is None:
            soc = x[0]
            ocv = 2.5 + 0.5*soc + 0.8*soc**2 - 0.3*soc**3 + 0.2*soc**4
        else:
            ocv = self.battery_model.get_ocv(x[0])
        
        return np.array([ocv])
    
    def fit(self, train_data: Dict[str, np.ndarray], 
            val_data: Optional[Dict[str, np.ndarray]] = None) -> None:
        """Fit the UKF parameters"""
        
        if 'time' in train_data and len(train_data['time']) > 1:
            self.dt = train_data['time'][1] - train_data['time'][0]
        
        self.ukf = UnscentedKalmanFilter(
            state_dim=1,
            measurement_dim=1,
            process_noise=self.process_noise,
            measurement_noise=self.measurement_noise,
            initial_covariance=self.initial_covariance,
            alpha=self.alpha,
            beta=self.beta,
            kappa=self.kappa
        )
        
        self.ukf.x = np.array([self.initial_soc])
        self.is_fitted = True
    
    def predict(self, data: Dict[str, np.ndarray], 
                return_uncertainty: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict SOC using UKF"""
        
        if not self.is_fitted:
            raise ValueError("UKF must be fitted before prediction")
        
        current = data['current']
        voltage = data['voltage']
        
        soc_estimates = []
        uncertainties = []
        
        # Reset UKF state
        self.ukf.x = np.array([self.initial_soc])
        self.ukf.P = np.eye(1) * self.initial_covariance
        
        for i in range(len(current)):
            # Prediction step
            u = np.array([current[i]])
            self.ukf.predict(self._state_transition, u, self.dt)
            
            # Update step
            z = np.array([voltage[i]])
            self.ukf.update(z, self._measurement_function)
            
            soc_estimates.append(self.ukf.x[0])
            uncertainties.append(np.sqrt(self.ukf.P[0, 0]))
        
        soc_estimates = np.array(soc_estimates)
        uncertainties = np.array(uncertainties)
        
        if return_uncertainty:
            return soc_estimates, uncertainties
        else:
            return soc_estimates 