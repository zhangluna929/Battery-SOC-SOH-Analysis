"""
Deep Learning SOC Estimators

This module implements state-of-the-art deep learning approaches for SOC estimation:
- LSTM with uncertainty quantification using Monte Carlo Dropout
- Transformer architecture with attention mechanisms  
- Bayesian neural networks for uncertainty estimation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Optional, Tuple, Union
import math
import warnings

from .base import BaseSOCEstimator


class BayesianLinear(nn.Module):
    """Bayesian Linear Layer for uncertainty quantification"""
    
    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_rho = nn.Parameter(torch.randn(out_features, in_features) * -3)
        
        # Bias parameters  
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.randn(out_features) * -3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Sample weights and biases
            weight_std = torch.log(1 + torch.exp(self.weight_rho))
            weight = self.weight_mu + weight_std * torch.randn_like(weight_std)
            
            bias_std = torch.log(1 + torch.exp(self.bias_rho))
            bias = self.bias_mu + bias_std * torch.randn_like(bias_std)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
            
        return nn.functional.linear(x, weight, bias)


class LSTMNet(nn.Module):
    """LSTM Network with optional Bayesian layers for uncertainty quantification"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 dropout: float = 0.2, uncertainty_quantification: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.uncertainty_quantification = uncertainty_quantification
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           dropout=dropout, batch_first=True)
        
        if uncertainty_quantification:
            self.fc = BayesianLinear(hidden_size, 1)
        else:
            self.fc = nn.Linear(hidden_size, 1)
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        
        # Use the last output
        output = self.fc(lstm_out[:, -1, :])
        return torch.sigmoid(output.squeeze())


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer architecture"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class TransformerNet(nn.Module):
    """Transformer Network for SOC estimation with attention mechanisms"""
    
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8, 
                 num_layers: int = 6, dim_feedforward: int = 512, 
                 dropout: float = 0.1, uncertainty_quantification: bool = True):
        super().__init__()
        self.d_model = d_model
        self.uncertainty_quantification = uncertainty_quantification
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        if uncertainty_quantification:
            self.output_layer = BayesianLinear(d_model, 1)
        else:
            self.output_layer = nn.Linear(d_model, 1)
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input projection
        x = self.input_projection(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Output
        output = self.output_layer(x)
        return torch.sigmoid(output.squeeze())


class LSTMSOCEstimator(BaseSOCEstimator):
    """
    LSTM-based SOC estimator with uncertainty quantification
    
    Features:
    - Monte Carlo Dropout for uncertainty estimation
    - Bayesian neural networks option
    - Sequence-based learning for temporal dependencies
    - Early stopping and learning rate scheduling
    """
    
    def __init__(self, input_features: list = None, sequence_length: int = 50,
                 hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2,
                 learning_rate: float = 1e-3, batch_size: int = 32, epochs: int = 100,
                 uncertainty_quantification: bool = True, device: str = 'auto', **kwargs):
        
        super().__init__(name="LSTM_SOC_Estimator", **kwargs)
        
        self.input_features = input_features or ['voltage', 'current', 'temperature']
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.uncertainty_quantification = uncertainty_quantification
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        
    def _prepare_sequences(self, data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequential data for LSTM training"""
        # Stack input features
        X_list = []
        for feature in self.input_features:
            if feature in data:
                X_list.append(data[feature])
            else:
                warnings.warn(f"Feature {feature} not found in data")
        
        X = np.stack(X_list, axis=1)  # Shape: (n_samples, n_features)
        
        if 'soc_true' in data:
            y = data['soc_true']
        else:
            y = None
        
        # Create sequences
        X_seq = []
        y_seq = []
        
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i-self.sequence_length:i])
            if y is not None:
                y_seq.append(y[i])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq) if y is not None else None
        
        return X_seq, y_seq
    
    def _normalize_data(self, X: np.ndarray, y: np.ndarray = None, fit_scaler: bool = True):
        """Normalize input and output data"""
        from sklearn.preprocessing import StandardScaler
        
        if fit_scaler:
            self.scaler_X = StandardScaler()
            X_norm = self.scaler_X.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            
            if y is not None:
                self.scaler_y = StandardScaler()
                y_norm = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
            else:
                y_norm = None
        else:
            if self.scaler_X is not None:
                X_norm = self.scaler_X.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            else:
                X_norm = X
            
            if y is not None and self.scaler_y is not None:
                y_norm = self.scaler_y.transform(y.reshape(-1, 1)).flatten()
            else:
                y_norm = y
        
        return X_norm, y_norm
    
    def fit(self, train_data: Dict[str, np.ndarray], 
            val_data: Optional[Dict[str, np.ndarray]] = None) -> None:
        """Train the LSTM model"""
        
        # Prepare training data
        X_train, y_train = self._prepare_sequences(train_data)
        X_train, y_train = self._normalize_data(X_train, y_train, fit_scaler=True)
        
        # Prepare validation data
        if val_data is not None:
            X_val, y_val = self._prepare_sequences(val_data)
            X_val, y_val = self._normalize_data(X_val, y_val, fit_scaler=False)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        
        # Create model
        input_size = len(self.input_features)
        self.model = LSTMNet(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            uncertainty_quantification=self.uncertainty_quantification
        ).to(self.device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Create data loader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            epoch_train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation
            if val_data is not None:
                self.model.eval()
                with torch.no_grad():
                    X_val_tensor = torch.FloatTensor(X_val).to(self.device)
                    y_val_tensor = torch.FloatTensor(y_val).to(self.device)
                    val_outputs = self.model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                    val_losses.append(val_loss)
                    
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 20:
                        print(f"Early stopping at epoch {epoch}")
                        break
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.6f}, "
                      f"Val Loss: {val_losses[-1] if val_data else 'N/A':.6f}")
        
        self.training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        self.is_fitted = True
    
    def predict(self, data: Dict[str, np.ndarray], 
                return_uncertainty: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict SOC with optional uncertainty estimation"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_test, _ = self._prepare_sequences(data)
        X_test, _ = self._normalize_data(X_test, fit_scaler=False)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        if return_uncertainty and self.uncertainty_quantification:
            # Monte Carlo Dropout for uncertainty estimation
            self.model.train()  # Enable dropout
            n_samples = 100
            predictions = []
            
            with torch.no_grad():
                for _ in range(n_samples):
                    pred = self.model(X_test_tensor)
                    predictions.append(pred.cpu().numpy())
            
            predictions = np.array(predictions)
            mean_pred = np.mean(predictions, axis=0)
            std_pred = np.std(predictions, axis=0)
            
            # Denormalize if needed
            if self.scaler_y is not None:
                mean_pred = self.scaler_y.inverse_transform(mean_pred.reshape(-1, 1)).flatten()
                std_pred = std_pred * self.scaler_y.scale_[0]  # Scale standard deviation
            
            return mean_pred, std_pred
        else:
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(X_test_tensor).cpu().numpy()
            
            # Denormalize if needed
            if self.scaler_y is not None:
                predictions = self.scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
            
            return predictions


class TransformerSOCEstimator(BaseSOCEstimator):
    """
    Transformer-based SOC estimator with attention mechanisms
    
    Features:
    - Self-attention for capturing long-range dependencies
    - Positional encoding for sequence awareness
    - Bayesian layers for uncertainty quantification
    - Multi-head attention for feature interactions
    """
    
    def __init__(self, input_features: list = None, sequence_length: int = 50,
                 d_model: int = 128, nhead: int = 8, num_layers: int = 6,
                 dim_feedforward: int = 512, dropout: float = 0.1,
                 learning_rate: float = 1e-4, batch_size: int = 16, epochs: int = 100,
                 uncertainty_quantification: bool = True, device: str = 'auto', **kwargs):
        
        super().__init__(name="Transformer_SOC_Estimator", **kwargs)
        
        self.input_features = input_features or ['voltage', 'current', 'temperature']
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.uncertainty_quantification = uncertainty_quantification
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
    
    def _prepare_sequences(self, data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequential data for Transformer training"""
        # Same as LSTM preparation
        X_list = []
        for feature in self.input_features:
            if feature in data:
                X_list.append(data[feature])
            else:
                warnings.warn(f"Feature {feature} not found in data")
        
        X = np.stack(X_list, axis=1)
        
        if 'soc_true' in data:
            y = data['soc_true']
        else:
            y = None
        
        # Create sequences
        X_seq = []
        y_seq = []
        
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i-self.sequence_length:i])
            if y is not None:
                y_seq.append(y[i])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq) if y is not None else None
        
        return X_seq, y_seq
    
    def _normalize_data(self, X: np.ndarray, y: np.ndarray = None, fit_scaler: bool = True):
        """Normalize input and output data"""
        from sklearn.preprocessing import StandardScaler
        
        if fit_scaler:
            self.scaler_X = StandardScaler()
            X_norm = self.scaler_X.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            
            if y is not None:
                self.scaler_y = StandardScaler()
                y_norm = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
            else:
                y_norm = None
        else:
            if self.scaler_X is not None:
                X_norm = self.scaler_X.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            else:
                X_norm = X
            
            if y is not None and self.scaler_y is not None:
                y_norm = self.scaler_y.transform(y.reshape(-1, 1)).flatten()
            else:
                y_norm = y
        
        return X_norm, y_norm
    
    def fit(self, train_data: Dict[str, np.ndarray], 
            val_data: Optional[Dict[str, np.ndarray]] = None) -> None:
        """Train the Transformer model"""
        
        # Prepare training data
        X_train, y_train = self._prepare_sequences(train_data)
        X_train, y_train = self._normalize_data(X_train, y_train, fit_scaler=True)
        
        # Prepare validation data
        if val_data is not None:
            X_val, y_val = self._prepare_sequences(val_data)
            X_val, y_val = self._normalize_data(X_val, y_val, fit_scaler=False)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        
        # Create model
        input_size = len(self.input_features)
        self.model = TransformerNet(
            input_size=input_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            uncertainty_quantification=self.uncertainty_quantification
        ).to(self.device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        
        # Create data loader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            epoch_train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            scheduler.step()
            
            # Validation
            if val_data is not None:
                self.model.eval()
                with torch.no_grad():
                    X_val_tensor = torch.FloatTensor(X_val).to(self.device)
                    y_val_tensor = torch.FloatTensor(y_val).to(self.device)
                    val_outputs = self.model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                    val_losses.append(val_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.6f}, "
                      f"Val Loss: {val_losses[-1] if val_data else 'N/A':.6f}")
        
        self.training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        self.is_fitted = True
    
    def predict(self, data: Dict[str, np.ndarray], 
                return_uncertainty: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict SOC with optional uncertainty estimation"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_test, _ = self._prepare_sequences(data)
        X_test, _ = self._normalize_data(X_test, fit_scaler=False)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        if return_uncertainty and self.uncertainty_quantification:
            # Monte Carlo Dropout for uncertainty estimation
            self.model.train()  # Enable dropout
            n_samples = 100
            predictions = []
            
            with torch.no_grad():
                for _ in range(n_samples):
                    pred = self.model(X_test_tensor)
                    predictions.append(pred.cpu().numpy())
            
            predictions = np.array(predictions)
            mean_pred = np.mean(predictions, axis=0)
            std_pred = np.std(predictions, axis=0)
            
            # Denormalize if needed
            if self.scaler_y is not None:
                mean_pred = self.scaler_y.inverse_transform(mean_pred.reshape(-1, 1)).flatten()
                std_pred = std_pred * self.scaler_y.scale_[0]
            
            return mean_pred, std_pred
        else:
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(X_test_tensor).cpu().numpy()
            
            # Denormalize if needed
            if self.scaler_y is not None:
                predictions = self.scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
            
            return predictions 