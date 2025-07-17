"""
Federated Learning Client for Hospital Data.
Implements Flower NumPyClient for linear regression with differential privacy.
"""

import argparse
import logging
import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import flwr as fl
from sklearn.model_selection import train_test_split

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from models.linear_regression import PrivateLinearRegression
from config.settings import get_config
from config.privacy import dp_mechanism

# Setup logging
def setup_logging(hospital_id: int):
    """Setup logging for the client."""
    log_config = get_config("logging")
    log_file = log_config["client_log_file"].format(hospital_id=hospital_id)
    
    # Create logs directory
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_config["level"]),
        format=log_config["format"],
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class HospitalClient(fl.client.NumPyClient):
    """
    Flower client representing a hospital in the federated learning system.
    """
    
    def __init__(self, hospital_id: int, data_path: str):
        """
        Initialize the hospital client.
        
        Args:
            hospital_id: Unique identifier for the hospital
            data_path: Path to the hospital's data file
        """
        self.hospital_id = hospital_id
        self.data_path = data_path
        self.logger = logging.getLogger(f"Hospital_{hospital_id}")
        
        # Load configuration
        self.model_config = get_config("model")
        self.privacy_config = get_config("privacy")
        
        # Initialize model
        self.model = PrivateLinearRegression(add_privacy=True)
        
        # Load and prepare data
        self.X_train, self.X_test, self.y_train, self.y_test = self._load_data()
        
        self.logger.info(f"Hospital {hospital_id} initialized with "
                        f"{len(self.X_train)} training and {len(self.X_test)} test samples")
    
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and split hospital data.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        try:
            # Load hospital data
            df = pd.read_csv(self.data_path)
            self.logger.info(f"Loaded data from {self.data_path}: {df.shape}")
            
            # Prepare features and target
            features = self.model_config["features"]
            target = self.model_config["target"]
            
            X = df[features].values
            y = df[target].values
            
            # Split into train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.model_config["test_size"],
                random_state=self.model_config["random_state"],
                stratify=y
            )
            
            self.logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """
        Get model parameters for sharing with the server.
        
        Args:
            config: Configuration from server
            
        Returns:
            List of model parameters
        """
        if not self.model.is_fitted:
            # Return random initialization if model not fitted
            num_features = len(self.model_config["features"])
            return [
                np.random.normal(0, 0.01, num_features),  # coefficients
                np.array([0.0])  # intercept
            ]
        
        # Get parameters with differential privacy
        params = self.model.get_parameters()
        
        self.logger.debug(f"Sharing parameters: coef_shape={params['coefficients'].shape}")
        
        return [params["coefficients"], np.array([params["intercept"]])]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set model parameters received from the server.
        
        Args:
            parameters: List of model parameters from server
        """
        if len(parameters) != 2:
            raise ValueError(f"Expected 2 parameters, got {len(parameters)}")
        
        coefficients, intercept = parameters
        
        # Set parameters in the model
        params_dict = {
            "coefficients": coefficients,
            "intercept": intercept[0] if len(intercept) > 0 else 0.0,
            "num_samples": len(self.X_train)
        }
        
        self.model.set_parameters(params_dict)
        
        self.logger.debug("Updated model parameters from server")
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train the model on local data.
        
        Args:
            parameters: Model parameters from server
            config: Training configuration
            
        Returns:
            Tuple of (updated_parameters, num_samples, metrics)
        """
        self.logger.info(f"Starting training round {config.get('server_round', 'unknown')}")
        
        # Set parameters from server
        self.set_parameters(parameters)
        
        # Train the model
        self.model.fit(self.X_train, self.y_train)
        
        # Evaluate on local test set
        train_metrics = self.model.evaluate(self.X_train, self.y_train)
        test_metrics = self.model.evaluate(self.X_test, self.y_test)
        
        # Log training results
        self.logger.info(f"Training completed - Train MSE: {train_metrics['mse']:.4f}, "
                        f"Test MSE: {test_metrics['mse']:.4f}, "
                        f"Test R²: {test_metrics['r2']:.4f}")
        
        # Prepare metrics for server
        metrics = {
            "train_mse": train_metrics["mse"],
            "train_r2": train_metrics["r2"],
            "test_mse": test_metrics["mse"],
            "test_r2": test_metrics["r2"],
            "hospital_id": self.hospital_id,
            "privacy_enabled": self.model.add_privacy
        }
        
        # Get updated parameters with privacy
        updated_parameters = self.get_parameters(config)
        
        return updated_parameters, len(self.X_train), metrics
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """
        Evaluate the model on local test data.
        
        Args:
            parameters: Model parameters from server
            config: Evaluation configuration
            
        Returns:
            Tuple of (loss, num_samples, metrics)
        """
        # Set parameters from server
        self.set_parameters(parameters)
        
        # Evaluate on test set
        test_metrics = self.model.evaluate(self.X_test, self.y_test)
        
        self.logger.info(f"Evaluation - MSE: {test_metrics['mse']:.4f}, "
                        f"R²: {test_metrics['r2']:.4f}")
        
        # Prepare metrics
        metrics = {
            "mse": test_metrics["mse"],
            "rmse": test_metrics["rmse"],
            "mae": test_metrics["mae"],
            "r2": test_metrics["r2"],
            "hospital_id": self.hospital_id
        }
        
        return test_metrics["mse"], len(self.X_test), metrics

def create_client(hospital_id: int) -> HospitalClient:
    """
    Create a hospital client instance.
    
    Args:
        hospital_id: Hospital identifier
        
    Returns:
        HospitalClient instance
    """
    data_config = get_config("data")
    data_path = os.path.join(
        data_config["hospital_data_dir"], 
        f"hospital_{hospital_id}.csv"
    )
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Hospital data not found: {data_path}")
    
    return HospitalClient(hospital_id, data_path)

def main():
    """Main function to run the client."""
    parser = argparse.ArgumentParser(description="Federated Learning Hospital Client")
    parser.add_argument("--hospital-id", type=int, required=True,
                       help="Hospital ID (0, 1, 2, ...)")
    parser.add_argument("--server-address", type=str, 
                       default=None,
                       help="Server address (default from config)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger = setup_logging(args.hospital_id)
    
    try:
        # Get server address
        client_config = get_config("client")
        server_address = args.server_address or client_config["server_address"]
        
        logger.info(f"Starting Hospital {args.hospital_id} client")
        logger.info(f"Connecting to server at {server_address}")
        
        # Create client
        client = create_client(args.hospital_id)
        
        # Generate privacy report
        privacy_report = dp_mechanism.get_privacy_report(
            get_config("server")["rounds"]
        )
        logger.info(f"Privacy configuration: {privacy_report['privacy_level']} "
                   f"(ε={privacy_report['per_round_epsilon']:.2f})")
        
        # Start client
        fl.client.start_numpy_client(
            server_address=server_address,
            client=client
        )
        
        logger.info(f"Hospital {args.hospital_id} client completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Client interrupted by user")
    except Exception as e:
        logger.error(f"Client failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
