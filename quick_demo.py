#!/usr/bin/env python3
"""
Quick demo of the federated learning system.
This runs a simplified version to demonstrate the system works.
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from models.linear_regression import PrivateLinearRegression, FederatedAveraging
from config.settings import get_config
from config.privacy import dp_mechanism

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def simulate_federated_learning():
    """Simulate the federated learning process locally."""
    
    logger.info("🚀 Starting Federated Learning Demo")
    logger.info("=" * 60)
    
    # Load configuration
    model_config = get_config("model")
    privacy_config = get_config("privacy")
    
    logger.info(f"Privacy settings: ε={privacy_config['epsilon']}, δ={privacy_config['delta']}")
    
    # Load hospital data
    hospital_models = []
    hospital_test_data = []
    
    logger.info("📊 Loading hospital data...")
    for i in range(3):
        # Load hospital data
        hospital_file = f"data/hospital_{i}.csv"
        if not os.path.exists(hospital_file):
            logger.error(f"Hospital data not found: {hospital_file}")
            logger.info("Please run: python scripts/preprocess.py")
            return False
        
        df = pd.read_csv(hospital_file)
        
        # Prepare data
        X = df[model_config['features']].values
        y = df[model_config['target']].values
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create and train model
        model = PrivateLinearRegression(add_privacy=True)
        model.fit(X_train, y_train)
        
        hospital_models.append(model)
        hospital_test_data.append((X_test, y_test))
        
        logger.info(f"✅ Hospital {i}: {len(X_train)} training samples")
    
    # Simulate federated learning rounds
    logger.info("\n🔄 Starting federated learning rounds...")
    
    for round_num in range(1, 4):  # 3 rounds for demo
        logger.info(f"\n--- Round {round_num} ---")
        
        # Get parameters from each hospital (with privacy)
        client_parameters = []
        for i, model in enumerate(hospital_models):
            params = model.get_parameters()
            client_parameters.append(params)
            logger.info(f"Hospital {i} shared parameters (with privacy noise)")
        
        # Aggregate parameters using FedAvg
        global_params = FederatedAveraging.aggregate_parameters(client_parameters)
        logger.info("🔄 Server aggregated parameters using FedAvg")
        
        # Update each hospital's model with global parameters
        for i, model in enumerate(hospital_models):
            model.set_parameters(global_params)
        
        # Evaluate global model on each hospital's test data
        total_mse = 0
        total_r2 = 0
        total_samples = 0
        
        for i, (model, (X_test, y_test)) in enumerate(zip(hospital_models, hospital_test_data)):
            metrics = model.evaluate(X_test, y_test)
            total_mse += metrics['mse'] * len(X_test)
            total_r2 += metrics['r2'] * len(X_test)
            total_samples += len(X_test)
            
            logger.info(f"Hospital {i} test: MSE={metrics['mse']:.4f}, R²={metrics['r2']:.4f}")
        
        # Calculate global metrics
        global_mse = total_mse / total_samples
        global_r2 = total_r2 / total_samples
        
        logger.info(f"🌍 Global metrics: MSE={global_mse:.4f}, R²={global_r2:.4f}")
        
        # Privacy report
        privacy_report = dp_mechanism.get_privacy_report(round_num)
        logger.info(f"🔒 Privacy: {privacy_report['privacy_level']} (ε={privacy_report['total_epsilon']:.2f})")
    
    logger.info("\n" + "=" * 60)
    logger.info("🎉 FEDERATED LEARNING DEMO COMPLETED!")
    logger.info("=" * 60)
    logger.info("✅ Successfully trained model across 3 hospitals")
    logger.info("✅ Protected patient privacy with differential privacy")
    logger.info("✅ No raw data was shared between hospitals")
    logger.info(f"✅ Final global MSE: {global_mse:.4f}")
    logger.info(f"✅ Final global R²: {global_r2:.4f}")
    logger.info(f"✅ Privacy level: {privacy_report['privacy_level']}")
    logger.info("=" * 60)
    
    return True

def main():
    """Main function."""
    try:
        # Check if data exists
        if not os.path.exists("data/hospital_0.csv"):
            logger.info("📊 Preparing data first...")
            import subprocess
            result = subprocess.run([sys.executable, "scripts/preprocess.py"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("Failed to prepare data")
                return False
        
        # Run the demo
        success = simulate_federated_learning()
        
        if success:
            logger.info("\n🚀 Want to run the full system?")
            logger.info("   python run_federated_learning.py")
            logger.info("\n📚 Want to learn more?")
            logger.info("   jupyter notebook notebooks/demo.ipynb")
            logger.info("\n🧪 Want to run tests?")
            logger.info("   python test_system.py")
        
        return success
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Demo interrupted by user")
        return False
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
