"""
Configuration settings for the federated learning system.
"""

import os
from typing import Dict, Any

# Server Configuration
SERVER_CONFIG = {
    "host": os.getenv("FL_SERVER_HOST", "0.0.0.0"),
    "port": int(os.getenv("FL_SERVER_PORT", "8080")),
    "rounds": int(os.getenv("FL_ROUNDS", "10")),
    "min_clients": int(os.getenv("FL_MIN_CLIENTS", "2")),
    "min_available_clients": int(os.getenv("FL_MIN_AVAILABLE_CLIENTS", "3")),
    "min_fit_clients": int(os.getenv("FL_MIN_FIT_CLIENTS", "2")),
    "min_evaluate_clients": int(os.getenv("FL_MIN_EVALUATE_CLIENTS", "2")),
}

# Client Configuration
CLIENT_CONFIG = {
    "server_address": os.getenv("FL_SERVER_ADDRESS", "localhost:8080"),
    "max_retries": int(os.getenv("FL_CLIENT_MAX_RETRIES", "3")),
    "retry_delay": float(os.getenv("FL_CLIENT_RETRY_DELAY", "1.0")),
}

# Model Configuration
MODEL_CONFIG = {
    "features": [
        "Pregnancies", 
        "Glucose", 
        "BloodPressure", 
        "SkinThickness", 
        "Insulin", 
        "BMI", 
        "DiabetesPedigreeFunction", 
        "Age"
    ],
    "target": "Outcome",
    "test_size": 0.2,
    "random_state": 42,
    "normalize": True,
}

# Privacy Configuration
PRIVACY_CONFIG = {
    "enable_differential_privacy": True,
    "epsilon": float(os.getenv("DP_EPSILON", "1.0")),
    "delta": float(os.getenv("DP_DELTA", "1e-5")),
    "noise_multiplier": float(os.getenv("DP_NOISE_MULTIPLIER", "0.1")),
    "max_grad_norm": float(os.getenv("DP_MAX_GRAD_NORM", "1.0")),
}

# Data Configuration
DATA_CONFIG = {
    "dataset_path": "data/diabetes.csv",
    "hospital_data_dir": "data/",
    "num_hospitals": 3,
    "split_method": "random",  # "random" or "stratified"
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_dir": "logs/",
    "server_log_file": "logs/server.log",
    "client_log_file": "logs/client_{hospital_id}.log",
    "metrics_file": "logs/metrics.json",
}

# Visualization Configuration
VIZ_CONFIG = {
    "results_dir": "results/",
    "plot_format": "png",
    "plot_dpi": 300,
    "figure_size": (10, 6),
    "style": "seaborn-v0_8",
}

# Deployment Configuration
DEPLOYMENT_CONFIG = {
    "render": {
        "build_command": "pip install -r requirements.txt",
        "start_command": "python server.py --host 0.0.0.0 --port $PORT",
        "environment": "python3",
        "region": "oregon",
    },
    "docker": {
        "base_image": "python:3.9-slim",
        "working_dir": "/app",
        "expose_port": 8080,
    }
}

def get_config(section: str = None) -> Dict[str, Any]:
    """
    Get configuration for a specific section or all configurations.
    
    Args:
        section: Configuration section name (e.g., 'server', 'client', 'model')
    
    Returns:
        Configuration dictionary
    """
    configs = {
        "server": SERVER_CONFIG,
        "client": CLIENT_CONFIG,
        "model": MODEL_CONFIG,
        "privacy": PRIVACY_CONFIG,
        "data": DATA_CONFIG,
        "logging": LOGGING_CONFIG,
        "visualization": VIZ_CONFIG,
        "deployment": DEPLOYMENT_CONFIG,
    }
    
    if section:
        return configs.get(section, {})
    return configs

def update_config(section: str, updates: Dict[str, Any]) -> None:
    """
    Update configuration values for a specific section.
    
    Args:
        section: Configuration section name
        updates: Dictionary of updates to apply
    """
    configs = get_config()
    if section in configs:
        configs[section].update(updates)
        globals()[f"{section.upper()}_CONFIG"] = configs[section]
