#!/usr/bin/env python3
"""
Test script for the federated learning system.
Tests all components without running the full FL process.
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from models.linear_regression import PrivateLinearRegression, FederatedAveraging
from config.settings import get_config
from config.privacy import dp_mechanism

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_configuration():
    """Test configuration loading."""
    logger.info("Testing configuration...")
    
    configs = ["server", "client", "model", "privacy", "data"]
    for config_name in configs:
        config = get_config(config_name)
        assert config, f"Failed to load {config_name} config"
        logger.info(f"✅ {config_name} config loaded")
    
    logger.info("✅ All configurations loaded successfully")

def test_privacy_mechanism():
    """Test differential privacy mechanism."""
    logger.info("Testing privacy mechanism...")
    
    # Test privacy report
    privacy_report = dp_mechanism.get_privacy_report(10)
    assert "privacy_level" in privacy_report
    assert "total_epsilon" in privacy_report
    logger.info(f"✅ Privacy level: {privacy_report['privacy_level']}")
    
    # Test noise addition
    params = np.array([1.0, 2.0, 3.0])
    noisy_params = dp_mechanism.add_gaussian_noise(params)
    assert len(noisy_params) == len(params)
    logger.info("✅ Differential privacy mechanism working")

def test_model():
    """Test the private linear regression model."""
    logger.info("Testing model...")
    
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(100, 8)
    y = np.random.randn(100)
    
    # Test model without privacy
    model_no_privacy = PrivateLinearRegression(add_privacy=False)
    model_no_privacy.fit(X, y)
    params_no_privacy = model_no_privacy.get_parameters()
    
    # Test model with privacy
    model_with_privacy = PrivateLinearRegression(add_privacy=True)
    model_with_privacy.fit(X, y)
    params_with_privacy = model_with_privacy.get_parameters()
    
    # Verify parameters have expected structure
    assert "coefficients" in params_no_privacy
    assert "intercept" in params_no_privacy
    assert "coefficients" in params_with_privacy
    assert "intercept" in params_with_privacy
    
    logger.info("✅ Model training and parameter extraction working")

def test_federated_averaging():
    """Test federated averaging mechanism."""
    logger.info("Testing federated averaging...")
    
    # Create sample client parameters
    client_params = []
    for i in range(3):
        params = {
            "coefficients": np.random.randn(8),
            "intercept": np.random.randn(),
            "num_samples": 100 + i * 10
        }
        client_params.append(params)
    
    # Test aggregation
    global_params = FederatedAveraging.aggregate_parameters(client_params)
    assert "coefficients" in global_params
    assert "intercept" in global_params
    assert "num_samples" in global_params
    
    logger.info("✅ Federated averaging working")

def test_data_loading():
    """Test data loading and hospital partitions."""
    logger.info("Testing data loading...")
    
    data_config = get_config("data")
    model_config = get_config("model")
    
    # Check if hospital data exists
    for i in range(data_config["num_hospitals"]):
        hospital_file = f"data/hospital_{i}.csv"
        assert os.path.exists(hospital_file), f"Hospital {i} data not found"
        
        # Load and verify data
        df = pd.read_csv(hospital_file)
        assert len(df) > 0, f"Hospital {i} data is empty"
        assert model_config["target"] in df.columns, f"Target column missing in hospital {i}"
        
        for feature in model_config["features"]:
            assert feature in df.columns, f"Feature {feature} missing in hospital {i}"
    
    logger.info("✅ All hospital data files loaded successfully")

def test_client_simulation():
    """Test client functionality without Flower."""
    logger.info("Testing client simulation...")
    
    # Load hospital data
    df = pd.read_csv("data/hospital_0.csv")
    model_config = get_config("model")
    
    # Prepare data
    X = df[model_config["features"]].values
    y = df[model_config["target"]].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = PrivateLinearRegression(add_privacy=True)
    model.fit(X_train, y_train)
    
    # Get parameters
    params = model.get_parameters()
    assert len(params["coefficients"]) == len(model_config["features"])
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    assert "mse" in metrics
    assert "r2" in metrics
    
    logger.info(f"✅ Client simulation working - MSE: {metrics['mse']:.4f}, R²: {metrics['r2']:.4f}")

def test_visualization_setup():
    """Test visualization components."""
    logger.info("Testing visualization setup...")
    
    # Create sample metrics for testing
    sample_metrics = [
        {
            "round": 1,
            "timestamp": "2024-01-01T10:00:00",
            "mse": 0.25,
            "r2": 0.75,
            "privacy_epsilon": 1.0,
            "privacy_delta": 1e-5,
            "privacy_level": "High Privacy"
        },
        {
            "round": 2,
            "timestamp": "2024-01-01T10:01:00",
            "mse": 0.22,
            "r2": 0.78,
            "privacy_epsilon": 2.0,
            "privacy_delta": 2e-5,
            "privacy_level": "High Privacy"
        }
    ]
    
    # Save sample metrics
    os.makedirs("logs", exist_ok=True)
    import json
    with open("logs/test_metrics.json", 'w') as f:
        json.dump(sample_metrics, f)
    
    # Test visualization import
    from scripts.visualize import FLVisualizer
    visualizer = FLVisualizer("logs/test_metrics.json", "results/test/")
    
    # Test loading
    assert visualizer.load_metrics()
    
    logger.info("✅ Visualization components working")

def run_all_tests():
    """Run all tests."""
    logger.info("🧪 Starting system tests...")
    logger.info("=" * 50)
    
    tests = [
        test_configuration,
        test_privacy_mechanism,
        test_model,
        test_federated_averaging,
        test_data_loading,
        test_client_simulation,
        test_visualization_setup
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            logger.error(f"❌ {test.__name__} failed: {e}")
            failed += 1
    
    logger.info("=" * 50)
    logger.info(f"🧪 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 All tests passed! System is ready for federated learning.")
        return True
    else:
        logger.error("💥 Some tests failed. Please fix the issues before proceeding.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
