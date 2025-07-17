# Setup Guide - Federated Learning for Diabetes Prediction

This guide provides detailed instructions for setting up and running the federated learning system locally and in production.

## Prerequisites

### System Requirements
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space
- Internet connection for package installation

### Required Software
- Git (for cloning the repository)
- Python pip (package manager)
- Optional: Docker (for containerized deployment)

## Local Development Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd fedral-main
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python test_system.py
```

You should see all tests pass with the message: "🎉 All tests passed! System is ready for federated learning."

### 5. Prepare Data

```bash
python scripts/preprocess.py
```

This will:
- Download the Pima Indians Diabetes Dataset (or create sample data)
- Clean and normalize the data
- Split data into 3 hospital partitions
- Save processed data to `data/` directory

## Running the Federated Learning System

### Method 1: Manual Start (Recommended for Learning)

1. **Start the Server** (Terminal 1):
```bash
python server.py
```

2. **Start Hospital Clients** (Separate terminals):
```bash
# Terminal 2
python client.py --hospital-id 0

# Terminal 3
python client.py --hospital-id 1

# Terminal 4
python client.py --hospital-id 2
```

3. **Monitor Progress**:
- Watch the terminal outputs for training progress
- Check `logs/` directory for detailed logs
- Metrics are saved to `logs/metrics.json`

4. **Visualize Results**:
```bash
python scripts/visualize.py
```

### Method 2: Automated Testing

For quick testing, you can run a simplified version:

```bash
python test_system.py
```

## Configuration

### Environment Variables

You can customize the system behavior using environment variables:

```bash
# Server configuration
export FL_SERVER_HOST=0.0.0.0
export FL_SERVER_PORT=8080
export FL_ROUNDS=10
export FL_MIN_CLIENTS=2

# Privacy configuration
export DP_EPSILON=1.0
export DP_DELTA=1e-5

# Logging
export LOG_LEVEL=INFO
```

### Configuration Files

Edit `config/settings.py` to modify:
- Number of federated learning rounds
- Privacy parameters (epsilon, delta)
- Model hyperparameters
- Data processing options

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'config'
   ```
   **Solution**: Ensure you're running scripts from the project root directory.

2. **Port Already in Use**
   ```
   OSError: [Errno 98] Address already in use
   ```
   **Solution**: Change the port in configuration or kill existing processes:
   ```bash
   lsof -ti:8080 | xargs kill -9
   ```

3. **Missing Data Files**
   ```
   FileNotFoundError: Hospital data not found
   ```
   **Solution**: Run the preprocessing script:
   ```bash
   python scripts/preprocess.py
   ```

4. **Memory Issues**
   **Solution**: Reduce the dataset size or increase system memory.

### Debugging

1. **Enable Verbose Logging**:
   ```bash
   python server.py --verbose
   python client.py --hospital-id 0 --verbose
   ```

2. **Check Log Files**:
   - Server logs: `logs/server.log`
   - Client logs: `logs/client_0.log`, `logs/client_1.log`, etc.
   - Metrics: `logs/metrics.json`

3. **Test Individual Components**:
   ```bash
   python test_system.py
   ```

## Performance Optimization

### For Better Performance

1. **Increase Batch Size**: Modify `MODEL_CONFIG` in `config/settings.py`
2. **Reduce Privacy Noise**: Increase epsilon value (reduces privacy)
3. **Use More Clients**: Add more hospital partitions
4. **Optimize Network**: Use faster network connection for distributed setup

### For Better Privacy

1. **Decrease Epsilon**: Lower values provide stronger privacy
2. **Increase Noise**: Modify noise multiplier in privacy config
3. **Reduce Rounds**: Fewer rounds consume less privacy budget
4. **Use Secure Communication**: Enable HTTPS in production

## Next Steps

After successful local setup:

1. **Experiment with Parameters**: Try different privacy settings
2. **Add More Hospitals**: Create additional data partitions
3. **Deploy to Cloud**: Follow the deployment guide
4. **Integrate Real Data**: Replace sample data with actual hospital data
5. **Enhance Security**: Add authentication and encryption

## Getting Help

- Check the troubleshooting section above
- Review log files for error details
- Run the test script to verify system integrity
- Consult the API documentation in `docs/api.md`

## Security Considerations

- Never share raw patient data between hospitals
- Monitor privacy budget consumption
- Use HTTPS in production environments
- Implement proper authentication for clients
- Regularly update dependencies for security patches
