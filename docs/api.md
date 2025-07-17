# API Reference - Federated Learning System

This document provides detailed API documentation for the federated learning system components.

## Core Components

### PrivateLinearRegression

The main model class that implements linear regression with differential privacy.

```python
from models.linear_regression import PrivateLinearRegression

# Initialize model
model = PrivateLinearRegression(add_privacy=True)
```

#### Methods

##### `__init__(add_privacy: bool = True)`
Initialize the private linear regression model.

**Parameters:**
- `add_privacy` (bool): Whether to add differential privacy to parameters

##### `fit(X: np.ndarray, y: np.ndarray) -> PrivateLinearRegression`
Fit the linear regression model.

**Parameters:**
- `X` (np.ndarray): Feature matrix of shape (n_samples, n_features)
- `y` (np.ndarray): Target vector of shape (n_samples,)

**Returns:**
- Self for method chaining

##### `get_parameters() -> Dict[str, np.ndarray]`
Get model parameters with optional differential privacy.

**Returns:**
- Dictionary containing:
  - `coefficients` (np.ndarray): Model coefficients
  - `intercept` (float): Model intercept
  - `num_samples` (int): Number of training samples

##### `set_parameters(parameters: Dict[str, np.ndarray]) -> None`
Set model parameters from aggregated values.

**Parameters:**
- `parameters` (dict): Dictionary with coefficients and intercept

##### `predict(X: np.ndarray) -> np.ndarray`
Make predictions using the fitted model.

**Parameters:**
- `X` (np.ndarray): Feature matrix

**Returns:**
- Predictions array

##### `evaluate(X: np.ndarray, y: np.ndarray) -> Dict[str, float]`
Evaluate model performance.

**Parameters:**
- `X` (np.ndarray): Feature matrix
- `y` (np.ndarray): True target values

**Returns:**
- Dictionary with metrics: mse, rmse, mae, r2, num_samples

### FederatedAveraging

Implements the FedAvg algorithm for parameter aggregation.

```python
from models.linear_regression import FederatedAveraging

# Aggregate parameters from multiple clients
global_params = FederatedAveraging.aggregate_parameters(client_parameters)
```

#### Static Methods

##### `aggregate_parameters(client_parameters: List[Dict]) -> Dict[str, np.ndarray]`
Aggregate parameters from multiple clients using weighted averaging.

**Parameters:**
- `client_parameters` (list): List of parameter dictionaries from clients

**Returns:**
- Aggregated parameters dictionary

##### `aggregate_metrics(client_metrics: List[Dict]) -> Dict[str, float]`
Aggregate evaluation metrics from multiple clients.

**Parameters:**
- `client_metrics` (list): List of metric dictionaries from clients

**Returns:**
- Aggregated metrics dictionary

### DifferentialPrivacy

Handles differential privacy mechanisms.

```python
from config.privacy import dp_mechanism

# Add noise to parameters
noisy_params = dp_mechanism.add_gaussian_noise(parameters)
```

#### Methods

##### `add_gaussian_noise(parameters: np.ndarray, sensitivity: float = 1.0, clip_norm: Optional[float] = None) -> np.ndarray`
Add Gaussian noise to model parameters.

**Parameters:**
- `parameters` (np.ndarray): Model parameters
- `sensitivity` (float): L2 sensitivity of the function
- `clip_norm` (float, optional): Maximum L2 norm for gradient clipping

**Returns:**
- Noisy parameters

##### `get_privacy_report(num_rounds: int) -> Dict`
Generate a privacy report for the federated learning process.

**Parameters:**
- `num_rounds` (int): Number of federated learning rounds

**Returns:**
- Dictionary containing privacy analysis

### HospitalClient

Flower client implementation for hospitals.

```python
from client import HospitalClient

# Create client
client = HospitalClient(hospital_id=0, data_path="data/hospital_0.csv")
```

#### Methods

##### `__init__(hospital_id: int, data_path: str)`
Initialize the hospital client.

**Parameters:**
- `hospital_id` (int): Unique identifier for the hospital
- `data_path` (str): Path to the hospital's data file

##### `get_parameters(config: Dict) -> List[np.ndarray]`
Get model parameters for sharing with the server.

**Parameters:**
- `config` (dict): Configuration from server

**Returns:**
- List of model parameters

##### `set_parameters(parameters: List[np.ndarray]) -> None`
Set model parameters received from the server.

**Parameters:**
- `parameters` (list): List of model parameters from server

##### `fit(parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]`
Train the model on local data.

**Parameters:**
- `parameters` (list): Model parameters from server
- `config` (dict): Training configuration

**Returns:**
- Tuple of (updated_parameters, num_samples, metrics)

##### `evaluate(parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]`
Evaluate the model on local test data.

**Parameters:**
- `parameters` (list): Model parameters from server
- `config` (dict): Evaluation configuration

**Returns:**
- Tuple of (loss, num_samples, metrics)

## Configuration API

### get_config(section: str = None) -> Dict[str, Any]

Get configuration for a specific section or all configurations.

```python
from config.settings import get_config

# Get server configuration
server_config = get_config("server")

# Get all configurations
all_configs = get_config()
```

**Parameters:**
- `section` (str, optional): Configuration section name

**Returns:**
- Configuration dictionary

### Available Configuration Sections

- `server`: Server settings (host, port, rounds, etc.)
- `client`: Client settings (server address, retries, etc.)
- `model`: Model configuration (features, target, etc.)
- `privacy`: Privacy settings (epsilon, delta, noise, etc.)
- `data`: Data configuration (paths, splits, etc.)
- `logging`: Logging settings (level, format, files, etc.)
- `visualization`: Visualization settings (style, format, etc.)
- `deployment`: Deployment configuration (Docker, Render, etc.)

## Visualization API

### FLVisualizer

Creates visualizations of federated learning results.

```python
from scripts.visualize import FLVisualizer

# Create visualizer
visualizer = FLVisualizer(metrics_file="logs/metrics.json")

# Generate all visualizations
results = visualizer.visualize_all()
```

#### Methods

##### `__init__(metrics_file: str = None, output_dir: str = None)`
Initialize the visualizer.

**Parameters:**
- `metrics_file` (str, optional): Path to metrics JSON file
- `output_dir` (str, optional): Directory to save plots

##### `load_metrics() -> bool`
Load metrics from JSON file.

**Returns:**
- True if successful, False otherwise

##### `create_training_progress_plot() -> str`
Create a plot showing training progress over rounds.

**Returns:**
- Path to saved plot

##### `create_privacy_analysis_plot() -> str`
Create a plot showing privacy budget consumption.

**Returns:**
- Path to saved plot

##### `create_summary_dashboard() -> str`
Create a comprehensive dashboard with all metrics.

**Returns:**
- Path to saved plot

##### `visualize_all() -> Dict[str, str]`
Generate all visualizations and reports.

**Returns:**
- Dictionary mapping visualization names to file paths

## Data Processing API

### DataPreprocessor

Handles data preprocessing for federated learning.

```python
from scripts.preprocess import DataPreprocessor

# Create preprocessor
preprocessor = DataPreprocessor()

# Run complete preprocessing pipeline
preprocessor.process_all()
```

#### Methods

##### `load_and_clean_data() -> pd.DataFrame`
Load and clean the diabetes dataset.

**Returns:**
- Cleaned DataFrame

##### `normalize_features(df: pd.DataFrame) -> pd.DataFrame`
Normalize features using StandardScaler.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame

**Returns:**
- DataFrame with normalized features

##### `split_for_hospitals(df: pd.DataFrame) -> List[pd.DataFrame]`
Split data into partitions for different hospitals.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame

**Returns:**
- List of DataFrames, one for each hospital

## Error Handling

### Common Exceptions

- `ValueError`: Invalid parameters or configuration
- `FileNotFoundError`: Missing data files or configuration
- `ConnectionError`: Network issues during federated learning
- `PrivacyBudgetExceeded`: Privacy budget exhausted

### Example Error Handling

```python
try:
    model = PrivateLinearRegression()
    model.fit(X, y)
    params = model.get_parameters()
except ValueError as e:
    logger.error(f"Invalid model parameters: {e}")
except Exception as e:
    logger.error(f"Unexpected error: {e}")
```

## Usage Examples

### Basic Model Training

```python
import numpy as np
from models.linear_regression import PrivateLinearRegression

# Create sample data
X = np.random.randn(100, 8)
y = np.random.randn(100)

# Train model with privacy
model = PrivateLinearRegression(add_privacy=True)
model.fit(X, y)

# Get parameters
params = model.get_parameters()
print(f"Coefficients: {params['coefficients']}")
print(f"Intercept: {params['intercept']}")

# Make predictions
predictions = model.predict(X)

# Evaluate performance
metrics = model.evaluate(X, y)
print(f"MSE: {metrics['mse']:.4f}")
print(f"R²: {metrics['r2']:.4f}")
```

### Federated Averaging

```python
from models.linear_regression import FederatedAveraging

# Simulate client parameters
client_params = [
    {"coefficients": np.random.randn(8), "intercept": 0.1, "num_samples": 100},
    {"coefficients": np.random.randn(8), "intercept": 0.2, "num_samples": 150},
    {"coefficients": np.random.randn(8), "intercept": 0.3, "num_samples": 120}
]

# Aggregate parameters
global_params = FederatedAveraging.aggregate_parameters(client_params)
print(f"Global coefficients: {global_params['coefficients']}")
print(f"Global intercept: {global_params['intercept']}")
```

### Privacy Analysis

```python
from config.privacy import dp_mechanism

# Generate privacy report
privacy_report = dp_mechanism.get_privacy_report(num_rounds=10)
print(f"Privacy Level: {privacy_report['privacy_level']}")
print(f"Total Epsilon: {privacy_report['total_epsilon']:.4f}")
print(f"Recommendations: {privacy_report['recommendations']}")
```
