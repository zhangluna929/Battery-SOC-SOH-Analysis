"""
Battery Data Loader

Comprehensive data loading and management for battery datasets.
"""

import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class BatteryDataLoader:
    """
    Comprehensive battery data loader supporting multiple formats and datasets
    
    Features:
    - Multiple file format support (CSV, HDF5, MAT, NPZ)
    - Data validation and cleaning
    - Automatic feature detection
    - Train/validation/test splitting
    - Data normalization and preprocessing
    """
    
    def __init__(self, data_path: Union[str, Path], 
                 data_format: str = "auto",
                 feature_mapping: Optional[Dict[str, str]] = None):
        """
        Initialize data loader
        
        Args:
            data_path: Path to data file or directory
            data_format: Data format (auto, csv, hdf5, mat, npz)
            feature_mapping: Mapping from file columns to standard names
        """
        self.data_path = Path(data_path)
        self.data_format = data_format
        self.feature_mapping = feature_mapping or {}
        
        # Standard feature names
        self.standard_features = {
            'time', 'voltage', 'current', 'temperature', 
            'soc_true', 'capacity', 'cycle_number'
        }
        
        self.data = {}
        self.metadata = {}
        self.scaler = None
        
    def load_data(self) -> Dict[str, np.ndarray]:
        """
        Load data from file
        
        Returns:
            Dictionary containing loaded data arrays
        """
        if self.data_format == "auto":
            self.data_format = self._detect_format()
        
        if self.data_format == "csv":
            self.data = self._load_csv()
        elif self.data_format == "hdf5":
            self.data = self._load_hdf5()
        elif self.data_format == "npz":
            self.data = self._load_npz()
        elif self.data_format == "synthetic":
            self.data = self._generate_synthetic_data()
        else:
            raise ValueError(f"Unsupported data format: {self.data_format}")
        
        self._validate_data()
        self._standardize_feature_names()
        return self.data
    
    def _detect_format(self) -> str:
        """Auto-detect data format from file extension"""
        if not self.data_path.exists():
            return "synthetic"
        
        suffix = self.data_path.suffix.lower()
        if suffix == ".csv":
            return "csv"
        elif suffix in [".h5", ".hdf5"]:
            return "hdf5"
        elif suffix == ".npz":
            return "npz"
        else:
            return "csv"  # Default fallback
    
    def _load_csv(self) -> Dict[str, np.ndarray]:
        """Load data from CSV file"""
        try:
            df = pd.read_csv(self.data_path)
            data = {}
            for col in df.columns:
                data[col.lower()] = df[col].values
            return data
        except Exception as e:
            raise ValueError(f"Failed to load CSV file: {e}")
    
    def _load_hdf5(self) -> Dict[str, np.ndarray]:
        """Load data from HDF5 file"""
        try:
            data = {}
            with h5py.File(self.data_path, 'r') as f:
                for key in f.keys():
                    data[key.lower()] = f[key][:]
            return data
        except Exception as e:
            raise ValueError(f"Failed to load HDF5 file: {e}")
    
    def _load_npz(self) -> Dict[str, np.ndarray]:
        """Load data from NPZ file"""
        try:
            npz_data = np.load(self.data_path)
            data = {}
            for key in npz_data.files:
                data[key.lower()] = npz_data[key]
            return data
        except Exception as e:
            raise ValueError(f"Failed to load NPZ file: {e}")
    
    def _generate_synthetic_data(self, n_samples: int = 3600) -> Dict[str, np.ndarray]:
        """
        Generate synthetic battery data for testing
        
        Args:
            n_samples: Number of data points to generate
            
        Returns:
            Dictionary with synthetic battery data
        """
        np.random.seed(42)
        
        # Time vector
        dt = 1.0  # 1 second intervals
        time = np.arange(n_samples) * dt
        
        # Generate realistic current profile
        current = self._generate_current_profile(n_samples)
        
        # Generate temperature profile
        temperature = 25 + 5 * np.sin(2 * np.pi * time / 3600) + np.random.normal(0, 1, n_samples)
        temperature = np.clip(temperature, 15, 40)  # Reasonable temperature range
        
        # Generate SOC using coulomb counting
        capacity = 2.5  # Ah
        soc_true = np.zeros(n_samples)
        soc_true[0] = 0.8  # Start at 80% SOC
        
        for i in range(1, n_samples):
            delta_soc = -current[i] * dt / 3600 / capacity
            soc_true[i] = np.clip(soc_true[i-1] + delta_soc, 0, 1)
        
        # Generate voltage using OCV model + noise
        voltage = self._generate_voltage(soc_true, current, temperature)
        
        # Generate cycle number (slowly increasing)
        cycle_number = np.floor(time / 3600).astype(int)
        
        # Generate capacity (slightly decreasing with cycles)
        capacity_array = np.full(n_samples, capacity) - cycle_number * 0.001
        
        return {
            'time': time,
            'voltage': voltage,
            'current': current,
            'temperature': temperature,
            'soc_true': soc_true,
            'capacity': capacity_array,
            'cycle_number': cycle_number
        }
    
    def _generate_current_profile(self, n_samples: int) -> np.ndarray:
        """Generate realistic current profile"""
        current = np.zeros(n_samples)
        
        # Create segments with different current patterns
        segment_length = n_samples // 4
        
        # Discharge segment
        current[:segment_length] = -2.0 + np.random.normal(0, 0.1, segment_length)
        
        # Rest segment
        current[segment_length:2*segment_length] = np.random.normal(0, 0.05, segment_length)
        
        # Charge segment
        current[2*segment_length:3*segment_length] = 1.5 + np.random.normal(0, 0.1, segment_length)
        
        # Variable load segment
        t = np.linspace(0, 4*np.pi, segment_length)
        current[3*segment_length:] = -1.0 * np.sin(t) + np.random.normal(0, 0.1, segment_length)
        
        return current
    
    def _generate_voltage(self, soc: np.ndarray, current: np.ndarray, 
                         temperature: np.ndarray) -> np.ndarray:
        """Generate realistic voltage profile"""
        # OCV model (polynomial fit)
        ocv = 2.5 + 0.5*soc + 0.8*soc**2 - 0.3*soc**3 + 0.2*soc**4
        
        # Internal resistance (temperature and SOC dependent)
        R0 = 0.01 * (1 + 0.01 * (25 - temperature))  # Temperature dependence
        R0 *= (1 + 0.5 * (np.exp(-10*soc) + np.exp(-10*(1-soc))))  # SOC dependence
        
        # Terminal voltage
        voltage = ocv - current * R0
        
        # Add measurement noise
        voltage += np.random.normal(0, 0.005, len(voltage))
        
        return voltage
    
    def _validate_data(self) -> None:
        """Validate loaded data for consistency and quality"""
        if not self.data:
            raise ValueError("No data loaded")
        
        # Check for required features
        required = {'voltage', 'current'}
        available = set(self.data.keys())
        missing = required - available
        if missing:
            warnings.warn(f"Missing required features: {missing}")
        
        # Check array lengths consistency
        lengths = [len(arr) for arr in self.data.values()]
        if len(set(lengths)) > 1:
            raise ValueError("Inconsistent array lengths in data")
        
        # Check for NaN values
        for key, arr in self.data.items():
            if np.isnan(arr).any():
                warnings.warn(f"NaN values found in {key}")
        
        # Check for reasonable value ranges
        if 'voltage' in self.data:
            voltage = self.data['voltage']
            if np.any(voltage < 1.0) or np.any(voltage > 5.0):
                warnings.warn("Voltage values outside reasonable range (1-5V)")
        
        if 'current' in self.data:
            current = self.data['current']
            if np.any(np.abs(current) > 20):
                warnings.warn("Current values seem unusually high (>20A)")
        
        if 'soc_true' in self.data:
            soc = self.data['soc_true']
            if np.any(soc < 0) or np.any(soc > 1):
                warnings.warn("SOC values outside [0,1] range")
    
    def _standardize_feature_names(self) -> None:
        """Apply feature name mapping to standardize column names"""
        if not self.feature_mapping:
            return
        
        new_data = {}
        for old_name, new_name in self.feature_mapping.items():
            if old_name in self.data:
                new_data[new_name] = self.data[old_name]
        
        # Add unmapped features
        for key, value in self.data.items():
            if key not in self.feature_mapping:
                new_data[key] = value
        
        self.data = new_data
    
    def preprocess_data(self, normalization: str = "standard",
                       smooth_data: bool = True,
                       remove_outliers: bool = True) -> Dict[str, np.ndarray]:
        """
        Preprocess the loaded data
        
        Args:
            normalization: Type of normalization (standard, minmax, robust, none)
            smooth_data: Whether to apply smoothing filters
            remove_outliers: Whether to remove outlier data points
            
        Returns:
            Preprocessed data dictionary
        """
        processed_data = self.data.copy()
        
        # Smooth data if requested
        if smooth_data:
            processed_data = self._smooth_data(processed_data)
        
        # Remove outliers if requested
        if remove_outliers:
            processed_data = self._remove_outliers(processed_data)
        
        # Normalize data if requested
        if normalization != "none":
            processed_data = self._normalize_data(processed_data, normalization)
        
        return processed_data
    
    def _smooth_data(self, data: Dict[str, np.ndarray], 
                    window_length: int = 5) -> Dict[str, np.ndarray]:
        """Apply smoothing filters to noisy signals"""
        smoothed_data = data.copy()
        
        # Features that benefit from smoothing
        smooth_features = {'voltage', 'current', 'temperature'}
        
        for feature in smooth_features:
            if feature in data and len(data[feature]) > window_length:
                if window_length % 2 == 0:
                    window_length += 1  # Ensure odd window length
                
                try:
                    smoothed_data[feature] = savgol_filter(
                        data[feature], window_length, polyorder=2
                    )
                except:
                    # Fall back to simple moving average
                    smoothed_data[feature] = np.convolve(
                        data[feature], np.ones(window_length)/window_length, mode='same'
                    )
        
        return smoothed_data
    
    def _remove_outliers(self, data: Dict[str, np.ndarray], 
                        threshold: float = 3.0) -> Dict[str, np.ndarray]:
        """Remove outlier data points using IQR method"""
        cleaned_data = data.copy()
        
        # Features to check for outliers
        check_features = {'voltage', 'current', 'temperature'}
        
        outlier_mask = np.ones(len(next(iter(data.values()))), dtype=bool)
        
        for feature in check_features:
            if feature in data:
                arr = data[feature]
                Q1 = np.percentile(arr, 25)
                Q3 = np.percentile(arr, 75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                feature_mask = (arr >= lower_bound) & (arr <= upper_bound)
                outlier_mask &= feature_mask
        
        # Apply mask to all features
        for key in cleaned_data:
            cleaned_data[key] = cleaned_data[key][outlier_mask]
        
        removed_count = np.sum(~outlier_mask)
        if removed_count > 0:
            print(f"Removed {removed_count} outlier data points")
        
        return cleaned_data
    
    def _normalize_data(self, data: Dict[str, np.ndarray], 
                       method: str) -> Dict[str, np.ndarray]:
        """Normalize specified features"""
        normalized_data = data.copy()
        
        # Features to normalize (exclude time and discrete features)
        normalize_features = {'voltage', 'current', 'temperature'}
        
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        elif method == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        for feature in normalize_features:
            if feature in data:
                arr = data[feature].reshape(-1, 1)
                normalized_data[feature] = scaler.fit_transform(arr).flatten()
        
        self.scaler = scaler
        return normalized_data
    
    def train_test_split(self, test_size: float = 0.2, 
                        val_size: float = 0.1,
                        strategy: str = "temporal") -> Tuple[Dict, Dict, Dict]:
        """
        Split data into train, validation, and test sets
        
        Args:
            test_size: Fraction of data for testing
            val_size: Fraction of data for validation  
            strategy: Split strategy (temporal, random, stratified)
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        if not self.data:
            raise ValueError("No data loaded. Call load_data() first.")
        
        n_samples = len(next(iter(self.data.values())))
        
        if strategy == "temporal":
            # Temporal split - use early data for training, later for testing
            val_start = int(n_samples * (1 - test_size - val_size))
            test_start = int(n_samples * (1 - test_size))
            
            train_indices = np.arange(0, val_start)
            val_indices = np.arange(val_start, test_start)
            test_indices = np.arange(test_start, n_samples)
            
        elif strategy == "random":
            # Random split
            indices = np.random.permutation(n_samples)
            val_end = int(n_samples * val_size)
            test_end = int(n_samples * (val_size + test_size))
            
            val_indices = indices[:val_end]
            test_indices = indices[val_end:test_end]
            train_indices = indices[test_end:]
            
        else:
            raise ValueError(f"Unknown split strategy: {strategy}")
        
        # Create data splits
        train_data = {key: arr[train_indices] for key, arr in self.data.items()}
        val_data = {key: arr[val_indices] for key, arr in self.data.items()}
        test_data = {key: arr[test_indices] for key, arr in self.data.items()}
        
        return train_data, val_data, test_data
    
    def get_metadata(self) -> Dict[str, any]:
        """Get metadata about the loaded dataset"""
        if not self.data:
            return {}
        
        n_samples = len(next(iter(self.data.values())))
        features = list(self.data.keys())
        
        metadata = {
            'n_samples': n_samples,
            'features': features,
            'data_path': str(self.data_path),
            'data_format': self.data_format
        }
        
        # Add feature statistics
        for key, arr in self.data.items():
            metadata[f'{key}_stats'] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr))
            }
        
        return metadata
    
    def save_data(self, output_path: Union[str, Path], 
                 format: str = "npz") -> None:
        """
        Save processed data to file
        
        Args:
            output_path: Path for output file
            format: Output format (npz, csv, hdf5)
        """
        output_path = Path(output_path)
        
        if format == "npz":
            np.savez_compressed(output_path, **self.data)
        elif format == "csv":
            df = pd.DataFrame(self.data)
            df.to_csv(output_path, index=False)
        elif format == "hdf5":
            with h5py.File(output_path, 'w') as f:
                for key, arr in self.data.items():
                    f.create_dataset(key, data=arr)
        else:
            raise ValueError(f"Unsupported output format: {format}") 