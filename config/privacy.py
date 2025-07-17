"""
Differential privacy utilities for federated learning.
"""

import numpy as np
import logging
from typing import Tuple, Optional
from config.settings import PRIVACY_CONFIG

logger = logging.getLogger(__name__)

class DifferentialPrivacy:
    """
    Implements differential privacy mechanisms for federated learning.
    """
    
    def __init__(self, 
                 epsilon: float = None, 
                 delta: float = None,
                 noise_multiplier: float = None):
        """
        Initialize differential privacy mechanism.
        
        Args:
            epsilon: Privacy budget parameter (smaller = more private)
            delta: Probability of privacy breach (should be very small)
            noise_multiplier: Multiplier for noise scale
        """
        self.epsilon = epsilon or PRIVACY_CONFIG["epsilon"]
        self.delta = delta or PRIVACY_CONFIG["delta"]
        self.noise_multiplier = noise_multiplier or PRIVACY_CONFIG["noise_multiplier"]
        
        logger.info(f"Initialized DP with epsilon={self.epsilon}, delta={self.delta}")
    
    def add_gaussian_noise(self, 
                          parameters: np.ndarray, 
                          sensitivity: float = 1.0,
                          clip_norm: Optional[float] = None) -> np.ndarray:
        """
        Add Gaussian noise to model parameters for differential privacy.
        
        Args:
            parameters: Model parameters (coefficients, intercept)
            sensitivity: L2 sensitivity of the function
            clip_norm: Maximum L2 norm for gradient clipping
            
        Returns:
            Noisy parameters
        """
        if clip_norm is not None:
            # Clip parameters to bound sensitivity
            param_norm = np.linalg.norm(parameters)
            if param_norm > clip_norm:
                parameters = parameters * (clip_norm / param_norm)
                logger.debug(f"Clipped parameters: norm {param_norm:.4f} -> {clip_norm}")
        
        # Calculate noise scale based on privacy parameters
        noise_scale = self._calculate_noise_scale(sensitivity)
        
        # Generate and add Gaussian noise
        noise = np.random.normal(0, noise_scale, parameters.shape)
        noisy_parameters = parameters + noise
        
        noise_magnitude = np.linalg.norm(noise)
        logger.debug(f"Added noise with magnitude: {noise_magnitude:.4f}")
        
        return noisy_parameters
    
    def _calculate_noise_scale(self, sensitivity: float) -> float:
        """
        Calculate the scale of Gaussian noise for given privacy parameters.
        
        Args:
            sensitivity: L2 sensitivity of the function
            
        Returns:
            Noise scale (standard deviation)
        """
        # For Gaussian mechanism: σ = sensitivity * sqrt(2 * ln(1.25/δ)) / ε
        if self.delta <= 0:
            raise ValueError("Delta must be positive for Gaussian mechanism")
        
        noise_scale = (sensitivity * np.sqrt(2 * np.log(1.25 / self.delta))) / self.epsilon
        noise_scale *= self.noise_multiplier  # Additional scaling factor
        
        return noise_scale
    
    def calculate_privacy_spent(self, num_rounds: int) -> Tuple[float, float]:
        """
        Calculate total privacy budget spent over multiple rounds.
        
        Args:
            num_rounds: Number of federated learning rounds
            
        Returns:
            Tuple of (total_epsilon, total_delta)
        """
        # Simple composition (conservative bound)
        total_epsilon = self.epsilon * num_rounds
        total_delta = self.delta * num_rounds
        
        logger.info(f"Privacy spent over {num_rounds} rounds: "
                   f"ε={total_epsilon:.4f}, δ={total_delta:.2e}")
        
        return total_epsilon, total_delta
    
    def get_privacy_report(self, num_rounds: int) -> dict:
        """
        Generate a privacy report for the federated learning process.
        
        Args:
            num_rounds: Number of federated learning rounds
            
        Returns:
            Dictionary containing privacy analysis
        """
        total_epsilon, total_delta = self.calculate_privacy_spent(num_rounds)
        
        return {
            "per_round_epsilon": self.epsilon,
            "per_round_delta": self.delta,
            "total_epsilon": total_epsilon,
            "total_delta": total_delta,
            "num_rounds": num_rounds,
            "noise_multiplier": self.noise_multiplier,
            "privacy_level": self._assess_privacy_level(total_epsilon),
            "recommendations": self._get_privacy_recommendations(total_epsilon, total_delta)
        }
    
    def _assess_privacy_level(self, epsilon: float) -> str:
        """Assess the privacy level based on epsilon value."""
        if epsilon < 0.1:
            return "Very High Privacy"
        elif epsilon < 1.0:
            return "High Privacy"
        elif epsilon < 5.0:
            return "Moderate Privacy"
        elif epsilon < 10.0:
            return "Low Privacy"
        else:
            return "Very Low Privacy"
    
    def _get_privacy_recommendations(self, epsilon: float, delta: float) -> list:
        """Get recommendations for privacy parameters."""
        recommendations = []
        
        if epsilon > 10.0:
            recommendations.append("Consider reducing epsilon for better privacy")
        
        if delta > 1e-3:
            recommendations.append("Consider reducing delta (should be << 1/dataset_size)")
        
        if len(recommendations) == 0:
            recommendations.append("Privacy parameters are within reasonable bounds")
        
        return recommendations

# Global differential privacy instance
dp_mechanism = DifferentialPrivacy()
