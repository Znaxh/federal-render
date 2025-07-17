# Federated Learning System for Diabetes Prediction

A production-ready federated learning system where multiple hospitals collaboratively train a linear regression model to predict diabetes outcomes without sharing patient data.

## 🎯 Project Overview

This system uses the Pima Indians Diabetes Dataset to demonstrate federated learning in healthcare. Multiple hospital clients train local models and share only model coefficients with a central server, preserving patient privacy.

### Key Features
- **Federated Learning**: Uses Flower framework for distributed training
- **Privacy-Preserving**: Implements differential privacy with Gaussian noise
- **Production-Ready**: Deployable on Render's free tier with HTTPS
- **Healthcare Focus**: Simulates real hospital collaboration scenarios

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Hospital A    │    │   Hospital B    │    │   Hospital C    │
│  (Client 1)     │    │  (Client 2)     │    │  (Client 3)     │
│                 │    │                 │    │                 │
│ Local Data      │    │ Local Data      │    │ Local Data      │
│ Linear Reg      │    │ Linear Reg      │    │ Linear Reg      │
│ + Diff Privacy  │    │ + Diff Privacy  │    │ + Diff Privacy  │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          │    Model Coefficients Only (No Raw Data)    │
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │     Flower Server         │
                    │   (Deployed on Render)    │
                    │                           │
                    │ • FedAvg Aggregation      │
                    │ • Global Model Updates    │
                    │ • Performance Monitoring  │
                    └───────────────────────────┘
```

## 📊 Dataset

**Pima Indians Diabetes Dataset**
- **Features**: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
- **Target**: Outcome (0: Non-diabetic, 1: Diabetic)
- **Size**: 768 samples
- **Split**: Divided into 3 hospital partitions for federated training

## 🚀 Quick Start

### Option 1: Quick Demo (Fastest - Try This First!)

```bash
# Clone and setup
git clone <repository-url>
cd fedral-main

# Install core dependencies
pip install flwr scikit-learn pandas numpy matplotlib seaborn flask requests

# Run quick demo
python quick_demo.py
```

### Option 2: Automated Run

```bash
# Clone and setup
git clone <repository-url>
cd fedral-main

# Install all dependencies (may have compilation issues)
pip install -r requirements.txt

# Run complete federated learning system
python run_federated_learning.py
```

### Option 3: Manual Step-by-Step

1. **Clone and Setup**
```bash
git clone <repository-url>
cd fedral-main
pip install flwr scikit-learn pandas numpy matplotlib seaborn flask requests
```

2. **Test System**
```bash
python test_system.py
```

3. **Prepare Data**
```bash
python scripts/preprocess.py
```

4. **Start Server**
```bash
python server.py
```

5. **Run Clients** (in separate terminals)
```bash
python client.py --hospital-id 0
python client.py --hospital-id 1
python client.py --hospital-id 2
```

6. **Visualize Results**
```bash
python scripts/visualize.py
```

### Option 4: Interactive Demo

```bash
# Install Jupyter (if needed)
pip install jupyter notebook

# Launch Jupyter notebook
jupyter notebook notebooks/demo.ipynb
```

## 🚨 Having Issues?

If you encounter errors:

1. **Try the quick demo first**: `python quick_demo.py`
2. **Install dependencies individually**:
   ```bash
   pip install flwr scikit-learn pandas numpy matplotlib seaborn flask requests
   ```
3. **Check the troubleshooting guide**: `TROUBLESHOOTING.md`
4. **Run system tests**: `python test_system.py`

### Cloud Deployment (Render)

1. **Deploy Server**
   - Connect GitHub repo to Render
   - Set build command: `pip install -r requirements.txt`
   - Set start command: `python server.py --host 0.0.0.0 --port $PORT`

2. **Connect Clients**
   - Update server address in client configuration
   - Run clients from local machines or other servers

## 📁 Project Structure

```
fedral-main/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── server.py                # Flower server implementation
├── client.py                # Flower client implementation
├── config/
│   ├── __init__.py
│   ├── settings.py          # Configuration settings
│   └── privacy.py           # Differential privacy utilities
├── data/
│   ├── __init__.py
│   ├── diabetes.csv         # Pima Indians dataset
│   └── hospital_*.csv       # Split datasets for each hospital
├── scripts/
│   ├── __init__.py
│   ├── preprocess.py        # Data preprocessing
│   ├── visualize.py         # Results visualization
│   └── deploy.py            # Deployment utilities
├── models/
│   ├── __init__.py
│   └── linear_regression.py # Linear regression with privacy
├── logs/                    # Training logs and metrics
├── results/                 # Plots and evaluation results
├── notebooks/
│   └── demo.ipynb          # Jupyter demo notebook
└── deployment/
    ├── Dockerfile          # Docker configuration
    ├── render.yaml         # Render deployment config
    └── nginx.conf          # Nginx configuration (if needed)
```

## 🔒 Privacy Features

- **Differential Privacy**: Gaussian noise added to model coefficients
- **No Data Sharing**: Only model parameters are transmitted
- **Secure Communication**: HTTPS encryption for all communications
- **Configurable Privacy Budget**: Adjustable epsilon and delta parameters

## 📈 Monitoring

- **Real-time Metrics**: MSE, R², training loss per round
- **Visualization**: Automatic plot generation for model performance
- **Logging**: Comprehensive logging for debugging and monitoring

## 🤖 Automated Runner

The `run_federated_learning.py` script provides a complete automated solution:

```bash
# Run with default settings (3 clients, 10 rounds)
python run_federated_learning.py

# Customize number of clients and rounds
python run_federated_learning.py --clients 5 --rounds 15

# Run specific components only
python run_federated_learning.py --test-only
python run_federated_learning.py --preprocess-only
python run_federated_learning.py --visualize-only
```

The automated runner:
- ✅ Checks all prerequisites
- ✅ Runs system tests
- ✅ Starts server and clients automatically
- ✅ Monitors progress
- ✅ Generates visualizations
- ✅ Provides comprehensive results

## 🛠️ Configuration

Key configuration options in `config/settings.py`:
- Number of federated learning rounds
- Privacy parameters (epsilon, delta)
- Model hyperparameters
- Server and client network settings

## 📚 Documentation

- [Setup Guide](docs/setup.md) - Detailed installation instructions
- [API Reference](docs/api.md) - Code documentation
- [Deployment Guide](docs/deployment.md) - Cloud deployment instructions
- [Privacy Guide](docs/privacy.md) - Understanding differential privacy

## 🤝 Contributing

This is a student project demonstrating federated learning concepts. Feel free to fork and extend for educational purposes.

## 📄 License

MIT License - See LICENSE file for details

## 🎓 Academic Use

This project is designed for educational purposes and demonstrates:
- Federated learning implementation
- Privacy-preserving machine learning
- Healthcare data collaboration
- Cloud deployment strategies

Perfect for computer science, data science, or healthcare informatics coursework.
# federal-render
