# 🔧 Troubleshooting Guide

This guide helps you resolve common issues when running the federated learning system.

## 🚨 Common Errors and Solutions

### 1. ModuleNotFoundError: No module named 'flwr'

**Error:**
```
ModuleNotFoundError: No module named 'flwr'
```

**Solution:**
```bash
# Install the required packages
pip install flwr scikit-learn pandas numpy matplotlib seaborn flask requests

# Or install all at once (may have compilation issues on some systems)
pip install -r requirements.txt
```

**Why this happens:** The Flower federated learning library and other dependencies aren't installed.

### 2. Compilation Errors with scikit-learn

**Error:**
```
error: metadata-generation-failed
Cython.Compiler.Errors.CompileError: sklearn/linear_model/_cd_fast.pyx
```

**Solution:**
```bash
# Install packages individually to avoid compilation issues
pip install flwr
pip install scikit-learn
pip install pandas numpy matplotlib seaborn flask requests
```

**Why this happens:** Some systems have issues compiling scikit-learn from source.

### 3. FileNotFoundError: Hospital data not found

**Error:**
```
FileNotFoundError: Hospital data not found: data/hospital_0.csv
```

**Solution:**
```bash
# Run the data preprocessing script
python scripts/preprocess.py
```

**Why this happens:** The hospital data hasn't been generated yet.

### 4. Port Already in Use

**Error:**
```
OSError: [Errno 98] Address already in use
```

**Solution:**
```bash
# Kill processes using port 8080
lsof -ti:8080 | xargs kill -9

# Or use a different port
python server.py --port 8081
```

**Why this happens:** Another process is using the default port 8080.

### 5. Server Failed to Start

**Error:**
```
❌ Server failed to start
```

**Solution:**
```bash
# Check if all dependencies are installed
python test_system.py

# Try running the server manually to see the error
python server.py

# Check if data is prepared
ls data/hospital_*.csv
```

### 6. Client Connection Timeout

**Error:**
```
Error: Connection timeout
```

**Solution:**
```bash
# Make sure server is running first
python server.py

# In another terminal, start client
python client.py --hospital-id 0

# Check if server address is correct
python client.py --hospital-id 0 --server-address localhost:8080
```

## 🛠️ Step-by-Step Troubleshooting

### Step 1: Check System Requirements

```bash
# Check Python version (should be 3.8+)
python --version

# Check available disk space (need ~2GB)
df -h

# Check memory (need ~4GB)
free -h
```

### Step 2: Install Dependencies

```bash
# Method 1: Install individually (recommended)
pip install flwr scikit-learn pandas numpy matplotlib seaborn flask requests

# Method 2: Install from requirements (may fail on some systems)
pip install -r requirements.txt
```

### Step 3: Test the System

```bash
# Run comprehensive tests
python test_system.py

# Should show: "🎉 All tests passed! System is ready for federated learning."
```

### Step 4: Prepare Data

```bash
# Generate hospital data partitions
python scripts/preprocess.py

# Verify data files exist
ls data/hospital_*.csv
```

### Step 5: Run Quick Demo

```bash
# Run simplified demo (fastest way to test)
python quick_demo.py

# Should show successful federated learning simulation
```

### Step 6: Run Full System

```bash
# Option 1: Automated (recommended)
python run_federated_learning.py

# Option 2: Manual control
# Terminal 1:
python server.py

# Terminal 2:
python client.py --hospital-id 0

# Terminal 3:
python client.py --hospital-id 1

# Terminal 4:
python client.py --hospital-id 2
```

## 🔍 Diagnostic Commands

### Check Installation Status

```bash
# Check if key packages are installed
python -c "import flwr; print('✅ Flower installed')"
python -c "import sklearn; print('✅ scikit-learn installed')"
python -c "import pandas; print('✅ pandas installed')"
python -c "import numpy; print('✅ numpy installed')"
python -c "import matplotlib; print('✅ matplotlib installed')"
```

### Check Data Status

```bash
# Check if data files exist
ls -la data/

# Check data file contents
head data/hospital_0.csv
wc -l data/hospital_*.csv
```

### Check System Status

```bash
# Check running processes
ps aux | grep python

# Check port usage
netstat -tulpn | grep 8080

# Check logs
tail -f logs/server.log
tail -f logs/client_0.log
```

## 🚀 Quick Fixes

### Reset Everything

```bash
# Clean up processes
pkill -f "python.*server.py"
pkill -f "python.*client.py"

# Remove generated files
rm -rf logs/* results/* data/hospital_*.csv

# Reinstall dependencies
pip uninstall flwr scikit-learn -y
pip install flwr scikit-learn pandas numpy matplotlib seaborn flask requests

# Regenerate data
python scripts/preprocess.py

# Test system
python test_system.py
```

### Minimal Working Example

```bash
# Just run the quick demo
python quick_demo.py

# This should work if basic dependencies are installed
```

## 🆘 Getting Help

### If Nothing Works

1. **Check Python version**: Must be 3.8 or higher
2. **Try virtual environment**:
   ```bash
   python -m venv fl_env
   source fl_env/bin/activate  # On Windows: fl_env\Scripts\activate
   pip install flwr scikit-learn pandas numpy matplotlib seaborn flask requests
   python quick_demo.py
   ```

3. **Check system resources**: Ensure enough RAM and disk space
4. **Try on different system**: Some systems have compilation issues

### Error Reporting

If you encounter an error not covered here:

1. **Run diagnostics**:
   ```bash
   python test_system.py > test_output.txt 2>&1
   ```

2. **Check logs**:
   ```bash
   ls logs/
   cat logs/server.log
   ```

3. **Provide details**:
   - Operating system and version
   - Python version
   - Error message
   - Steps that led to the error

## ✅ Success Indicators

You know the system is working when:

- ✅ `python test_system.py` shows all tests pass
- ✅ `python quick_demo.py` completes successfully
- ✅ Data files exist in `data/hospital_*.csv`
- ✅ Server starts without errors
- ✅ Clients connect and train successfully
- ✅ Results are generated in `results/` directory

## 🎯 Alternative Approaches

### If Full System Doesn't Work

1. **Use Quick Demo**: `python quick_demo.py`
2. **Use Jupyter Notebook**: `jupyter notebook notebooks/demo.ipynb`
3. **Run Individual Components**: Test server and client separately
4. **Use Docker**: If available, use containerized version

### For Educational Purposes

Even if the full federated learning doesn't run, you can still:
- Study the code architecture
- Understand privacy mechanisms
- Learn about federated learning concepts
- Use the documentation and guides

The quick demo (`python quick_demo.py`) provides the core educational value without requiring the full Flower server/client setup.
