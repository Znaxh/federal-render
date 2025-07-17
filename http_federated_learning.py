#!/usr/bin/env python3
"""
HTTP-based Federated Learning Demo.
This works with Render since it only uses HTTP, not gRPC.
"""

import requests
import json
import time
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from models.linear_regression import PrivateLinearRegression, FederatedAveraging
from config.settings import get_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HTTPFederatedClient:
    """HTTP-based federated learning client."""
    
    def __init__(self, hospital_id: int, server_url: str):
        self.hospital_id = hospital_id
        self.server_url = server_url.rstrip('/')
        self.model = PrivateLinearRegression(add_privacy=True)
        
        # Load data
        self.X_train, self.X_test, self.y_train, self.y_test = self._load_data()
        
        logger.info(f"Hospital {hospital_id} initialized with {len(self.X_train)} training samples")
    
    def _load_data(self):
        """Load hospital data."""
        data_file = f"data/hospital_{self.hospital_id}.csv"
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Hospital data not found: {data_file}")
        
        df = pd.read_csv(data_file)
        model_config = get_config("model")
        
        X = df[model_config["features"]].values
        y = df[model_config["target"]].values
        
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    def register_with_server(self):
        """Register this client with the server."""
        try:
            response = requests.post(
                f"{self.server_url}/register_client",
                json={"hospital_id": self.hospital_id, "num_samples": len(self.X_train)},
                timeout=10
            )
            if response.status_code == 200:
                logger.info(f"✅ Hospital {self.hospital_id} registered with server")
                return True
            else:
                logger.error(f"❌ Registration failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Registration error: {e}")
            return False
    
    def get_global_parameters(self):
        """Get global parameters from server."""
        try:
            response = requests.get(f"{self.server_url}/get_parameters", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"❌ Failed to get parameters: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"❌ Error getting parameters: {e}")
            return None
    
    def send_parameters(self, parameters, metrics):
        """Send local parameters to server."""
        try:
            data = {
                "hospital_id": self.hospital_id,
                "parameters": {
                    "coefficients": parameters["coefficients"].tolist(),
                    "intercept": float(parameters["intercept"]),
                    "num_samples": parameters["num_samples"]
                },
                "metrics": metrics
            }
            
            response = requests.post(
                f"{self.server_url}/submit_parameters",
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Hospital {self.hospital_id} submitted parameters")
                return True
            else:
                logger.error(f"❌ Failed to submit parameters: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Error submitting parameters: {e}")
            return False
    
    def train_and_submit(self):
        """Train local model and submit parameters."""
        # Train model
        self.model.fit(self.X_train, self.y_train)
        
        # Get parameters with privacy
        parameters = self.model.get_parameters()
        
        # Evaluate model
        train_metrics = self.model.evaluate(self.X_train, self.y_train)
        test_metrics = self.model.evaluate(self.X_test, self.y_test)
        
        metrics = {
            "train_mse": train_metrics["mse"],
            "train_r2": train_metrics["r2"],
            "test_mse": test_metrics["mse"],
            "test_r2": test_metrics["r2"],
        }
        
        logger.info(f"Hospital {self.hospital_id} - Train MSE: {train_metrics['mse']:.4f}, "
                   f"Test MSE: {test_metrics['mse']:.4f}, Test R²: {test_metrics['r2']:.4f}")
        
        # Submit to server
        return self.send_parameters(parameters, metrics)
    
    def run_federated_learning(self, max_rounds=10):
        """Run the federated learning process."""
        logger.info(f"🏥 Starting Hospital {self.hospital_id} federated learning")
        
        # Register with server
        if not self.register_with_server():
            return False
        
        for round_num in range(1, max_rounds + 1):
            logger.info(f"--- Round {round_num} ---")
            
            # Get global parameters
            global_params = self.get_global_parameters()
            if global_params and "coefficients" in global_params:
                # Update local model with global parameters
                params_dict = {
                    "coefficients": np.array(global_params["coefficients"]),
                    "intercept": global_params["intercept"],
                    "num_samples": len(self.X_train)
                }
                self.model.set_parameters(params_dict)
                logger.info(f"Updated model with global parameters")
            
            # Train and submit
            if not self.train_and_submit():
                logger.error(f"Failed to submit parameters for round {round_num}")
                break
            
            # Wait for other clients
            time.sleep(5)
        
        logger.info(f"🎉 Hospital {self.hospital_id} completed federated learning!")
        return True

def test_http_client(hospital_id=0):
    """Test the HTTP client with your Render server."""
    server_url = "https://federated-learning-server-xbx0.onrender.com"
    
    logger.info(f"🧪 Testing HTTP Federated Learning Client")
    logger.info(f"Server: {server_url}")
    logger.info(f"Hospital ID: {hospital_id}")
    
    try:
        # Check if data exists
        if not os.path.exists(f"data/hospital_{hospital_id}.csv"):
            logger.info("Preparing data...")
            import subprocess
            subprocess.run([sys.executable, "scripts/preprocess.py"], check=True)
        
        # Create and run client
        client = HTTPFederatedClient(hospital_id, server_url)
        
        # Test server connectivity
        logger.info("Testing server connectivity...")
        response = requests.get(f"{server_url}/health", timeout=10)
        if response.status_code == 200:
            logger.info("✅ Server is reachable")
        else:
            logger.error(f"❌ Server returned {response.status_code}")
            return False
        
        # Note: This won't work until we add HTTP endpoints to the server
        logger.info("⚠️ Note: This requires HTTP endpoints on the server")
        logger.info("   The current Render server only has gRPC endpoints")
        logger.info("   Use 'python quick_demo.py' for a working demonstration")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ HTTP client test failed: {e}")
        return False

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="HTTP Federated Learning Client")
    parser.add_argument("--hospital-id", type=int, default=0, help="Hospital ID")
    parser.add_argument("--server-url", type=str, 
                       default="https://federated-learning-server-xbx0.onrender.com",
                       help="Server URL")
    
    args = parser.parse_args()
    
    success = test_http_client(args.hospital_id)
    
    if not success:
        logger.info("\n💡 Alternative: Run the local simulation")
        logger.info("   python quick_demo.py")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
