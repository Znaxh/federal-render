"""
Linear regression model with differential privacy for federated learning.
"""

import numpy as np
import logging
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from typing import Tuple, Dict, Any
import pickle

from config.privacy import dp_mechanism
from config.settings import MODEL_CONFIG, PRIVACY_CONFIG

logger = logging.getLogger(__name__)

class PrivateLinearRegression:
    """
    Linear regression model with differential privacy capabilities.
    """
    
    def __init__(self, add_privacy: bool = True):
        """
        Initialize the private linear regression model.
        
        Args:
            add_privacy: Whether to add differential privacy
        """
        self.model = LinearRegression()
        self.add_privacy = add_privacy and PRIVACY_CONFIG["enable_differential_privacy"]
        self.is_fitted = False
        
        logger.info(f"Initialized PrivateLinearRegression with privacy: {self.add_privacy}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PrivateLinearRegression':
        """
        Fit the linear regression model.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Self for method chaining
        """
        # Fit the model
        self.model.fit(X, y)
        self.is_fitted = True
        
        # Log training information
        logger.info(f"Model fitted on {X.shape[0]} samples with {X.shape[1]} features")
        
        return self
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """
        Get model parameters with optional differential privacy.
        
        Returns:
            Dictionary containing coefficients and intercept
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting parameters")
        
        # Get raw parameters
        coefficients = self.model.coef_.copy()
        intercept = np.array([self.model.intercept_])
        
        # Combine parameters for privacy mechanism
        all_params = np.concatenate([coefficients, intercept])
        
        if self.add_privacy:
            # Add differential privacy noise
            noisy_params = dp_mechanism.add_gaussian_noise(
                all_params,
                sensitivity=1.0,  # L2 sensitivity
                clip_norm=PRIVACY_CONFIG.get("max_grad_norm", 1.0)
            )
            
            # Split back into coefficients and intercept
            noisy_coefficients = noisy_params[:-1]
            noisy_intercept = noisy_params[-1]
            
            logger.debug("Added differential privacy noise to parameters")
            
            return {
                "coefficients": noisy_coefficients,
                "intercept": noisy_intercept,
                "num_samples": len(coefficients)
            }
        else:
            return {
                "coefficients": coefficients,
                "intercept": intercept,
                "num_samples": len(coefficients)
            }
    
    def set_parameters(self, parameters: Dict[str, np.ndarray]) -> None:
        """
        Set model parameters.
        
        Args:
            parameters: Dictionary containing coefficients and intercept
        """
        self.model.coef_ = parameters["coefficients"]
        self.model.intercept_ = parameters["intercept"]
        self.is_fitted = True
        
        logger.debug("Updated model parameters")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Feature matrix
            y: True target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(X)
        
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y - predictions))
        
        metrics = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "num_samples": len(y)
        }
        
        logger.info(f"Evaluation metrics: MSE={mse:.4f}, R²={r2:.4f}")
        
        return metrics
    
    def save_model(self, filepath: str) -> None:
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            "model": self.model,
            "add_privacy": self.add_privacy,
            "is_fitted": self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a model from a file.
        
        Args:
            filepath: Path to load the model from
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data["model"]
        self.add_privacy = model_data["add_privacy"]
        self.is_fitted = model_data["is_fitted"]
        
        logger.info(f"Model loaded from {filepath}")

class FederatedAveraging:
    """
    Implements Federated Averaging (FedAvg) for linear regression.
    """
    
    @staticmethod
    def aggregate_parameters(client_parameters: list) -> Dict[str, np.ndarray]:
        """
        Aggregate parameters from multiple clients using weighted averaging.
        
        Args:
            client_parameters: List of parameter dictionaries from clients
            
        Returns:
            Aggregated parameters
        """
        if not client_parameters:
            raise ValueError("No client parameters provided")
        
        # Calculate total number of samples
        total_samples = sum(params["num_samples"] for params in client_parameters)
        
        # Initialize aggregated parameters
        aggregated_coefficients = np.zeros_like(client_parameters[0]["coefficients"])
        aggregated_intercept = 0.0
        
        # Weighted averaging
        for params in client_parameters:
            weight = params["num_samples"] / total_samples
            aggregated_coefficients += weight * params["coefficients"]
            aggregated_intercept += weight * params["intercept"]
        
        logger.info(f"Aggregated parameters from {len(client_parameters)} clients "
                   f"with total {total_samples} samples")
        
        return {
            "coefficients": aggregated_coefficients,
            "intercept": aggregated_intercept,
            "num_samples": total_samples
        }
    
    @staticmethod
    def aggregate_metrics(client_metrics: list) -> Dict[str, float]:
        """
        Aggregate evaluation metrics from multiple clients.
        
        Args:
            client_metrics: List of metric dictionaries from clients
            
        Returns:
            Aggregated metrics
        """
        if not client_metrics:
            return {}
        
        total_samples = sum(metrics["num_samples"] for metrics in client_metrics)
        
        # Weighted averaging for metrics
        aggregated_metrics = {}
        for metric_name in ["mse", "rmse", "mae", "r2"]:
            if metric_name in client_metrics[0]:
                weighted_sum = sum(
                    metrics[metric_name] * metrics["num_samples"] 
                    for metrics in client_metrics
                )
                aggregated_metrics[metric_name] = weighted_sum / total_samples
        
        aggregated_metrics["num_samples"] = total_samples
        aggregated_metrics["num_clients"] = len(client_metrics)
        
        logger.info(f"Aggregated metrics from {len(client_metrics)} clients")
        
        return aggregated_metrics
