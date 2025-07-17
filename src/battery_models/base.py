"""
Base Battery Model

Abstract base class for all battery models.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional, Tuple


class BaseBatteryModel(ABC):
    """
    Abstract base class for battery models
    
    Defines the common interface for all battery modeling approaches.
    """
    
    def __init__(self, name: str = "BaseBatteryModel", **kwargs):
        """
        Initialize the battery model
        
        Args:
            name: Name of the model
            **kwargs: Model-specific parameters
        """
        self.name = name
        self.parameters = kwargs
        self.is_initialized = False
        
        # Default battery parameters
        self.nominal_capacity = kwargs.get('nominal_capacity', 2.5)  # Ah
        self.nominal_voltage = kwargs.get('nominal_voltage', 3.3)    # V
        self.max_voltage = kwargs.get('max_voltage', 3.6)            # V
        self.min_voltage = kwargs.get('min_voltage', 2.5)            # V
        
    @abstractmethod
    def get_ocv(self, soc: float, temperature: float = 25.0) -> float:
        """
        Get Open Circuit Voltage for given SOC and temperature
        
        Args:
            soc: State of Charge (0-1)
            temperature: Temperature in Celsius
            
        Returns:
            Open circuit voltage in Volts
        """
        pass
    
    @abstractmethod
    def get_voltage(self, soc: float, current: float, temperature: float = 25.0, 
                   dt: float = 1.0) -> float:
        """
        Get terminal voltage for given conditions
        
        Args:
            soc: State of Charge (0-1)
            current: Current in Amperes (positive = charge, negative = discharge)
            temperature: Temperature in Celsius
            dt: Time step in seconds
            
        Returns:
            Terminal voltage in Volts
        """
        pass
    
    @abstractmethod 
    def update_soc(self, soc: float, current: float, dt: float = 1.0) -> float:
        """
        Update SOC based on current flow
        
        Args:
            soc: Current state of charge (0-1)
            current: Current in Amperes
            dt: Time step in seconds
            
        Returns:
            Updated state of charge
        """
        pass
    
    def update_state(self, state: Dict[str, Any], current: float, 
                    temperature: float = 25.0, dt: float = 1.0) -> Dict[str, Any]:
        """
        Update all model states
        
        Args:
            state: Current state dictionary
            current: Current in Amperes
            temperature: Temperature in Celsius
            dt: Time step in seconds
            
        Returns:
            Updated state dictionary
        """
        # Default implementation - subclasses should override for more complex states
        new_state = state.copy()
        if 'soc' in state:
            new_state['soc'] = self.update_soc(state['soc'], current, dt)
        return new_state
    
    def get_capacity(self, temperature: float = 25.0, age_factor: float = 1.0) -> float:
        """
        Get effective capacity considering temperature and aging
        
        Args:
            temperature: Temperature in Celsius
            age_factor: Aging factor (1.0 = new, 0.8 = 80% capacity)
            
        Returns:
            Effective capacity in Ah
        """
        # Simple temperature dependence (can be enhanced)
        temp_factor = 1.0 - 0.001 * (25.0 - temperature)  # 0.1% per degree
        temp_factor = max(0.5, min(1.2, temp_factor))  # Bound between 50% and 120%
        
        return self.nominal_capacity * temp_factor * age_factor
    
    def get_internal_resistance(self, soc: float, temperature: float = 25.0, 
                              age_factor: float = 1.0) -> float:
        """
        Get internal resistance
        
        Args:
            soc: State of charge (0-1)
            temperature: Temperature in Celsius
            age_factor: Aging factor (1.0 = new, 2.0 = doubled resistance)
            
        Returns:
            Internal resistance in Ohms
        """
        # Default simple model - subclasses should override
        base_resistance = 0.01  # 10 mOhm
        
        # SOC dependence (higher resistance at extremes)
        soc_factor = 1.0 + 0.5 * (np.exp(-10*soc) + np.exp(-10*(1-soc)))
        
        # Temperature dependence
        temp_factor = np.exp(0.01 * (25.0 - temperature))
        
        return base_resistance * soc_factor * temp_factor * age_factor
    
    def set_parameters(self, **parameters) -> None:
        """Update model parameters"""
        self.parameters.update(parameters)
        
        # Update key parameters if provided
        if 'nominal_capacity' in parameters:
            self.nominal_capacity = parameters['nominal_capacity']
        if 'nominal_voltage' in parameters:
            self.nominal_voltage = parameters['nominal_voltage']
        if 'max_voltage' in parameters:
            self.max_voltage = parameters['max_voltage']
        if 'min_voltage' in parameters:
            self.min_voltage = parameters['min_voltage']
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get all model parameters"""
        return self.parameters.copy()
    
    def reset(self) -> None:
        """Reset model to initial state"""
        self.is_initialized = False
    
    def validate_inputs(self, soc: float, current: float, temperature: float) -> None:
        """Validate input parameters"""
        if not 0 <= soc <= 1:
            raise ValueError(f"SOC must be between 0 and 1, got {soc}")
        
        if abs(current) > 20:  # Reasonable current limit
            raise ValueError(f"Current seems too high: {current} A")
        
        if not -40 <= temperature <= 80:  # Reasonable temperature range
            raise ValueError(f"Temperature out of reasonable range: {temperature} °C")
    
    def __str__(self) -> str:
        """String representation"""
        return f"{self.name} (initialized: {self.is_initialized})"
    
    def __repr__(self) -> str:
        """Detailed representation"""
        return f"{self.__class__.__name__}(name='{self.name}', parameters={self.parameters})" 