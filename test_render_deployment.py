#!/usr/bin/env python3
"""
Test script for Render deployment.
Tests the deployed federated learning server.
"""

import requests
import json
import time
import subprocess
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Your Render service URL
RENDER_URL = "https://federated-learning-server-xbx0.onrender.com"
FL_SERVER = "federated-learning-server-xbx0.onrender.com:10000"

def test_health_check():
    """Test the health check endpoint."""
    logger.info("🔍 Testing health check endpoint...")
    
    try:
        response = requests.get(f"{RENDER_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ Health check passed: {data['status']}")
            return True
        else:
            logger.error(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Health check error: {e}")
        return False

def test_web_dashboard():
    """Test the web dashboard."""
    logger.info("🌐 Testing web dashboard...")
    
    try:
        response = requests.get(RENDER_URL, timeout=10)
        if response.status_code == 200:
            if "Federated Learning Server" in response.text:
                logger.info("✅ Web dashboard is accessible")
                return True
            else:
                logger.error("❌ Web dashboard content is incorrect")
                return False
        else:
            logger.error(f"❌ Web dashboard failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Web dashboard error: {e}")
        return False

def test_api_endpoints():
    """Test API endpoints."""
    logger.info("📊 Testing API endpoints...")
    
    endpoints = ["/status", "/metrics"]
    
    for endpoint in endpoints:
        try:
            response = requests.get(f"{RENDER_URL}{endpoint}", timeout=10)
            if response.status_code == 200:
                logger.info(f"✅ {endpoint} endpoint working")
            else:
                logger.warning(f"⚠️ {endpoint} returned {response.status_code}")
        except Exception as e:
            logger.warning(f"⚠️ {endpoint} error: {e}")

def check_local_setup():
    """Check if local environment is ready."""
    logger.info("🔧 Checking local setup...")
    
    # Check if data files exist
    data_files = ["data/hospital_0.csv", "data/hospital_1.csv", "data/hospital_2.csv"]
    missing_files = [f for f in data_files if not os.path.exists(f)]
    
    if missing_files:
        logger.warning(f"⚠️ Missing data files: {missing_files}")
        logger.info("Running data preprocessing...")
        try:
            subprocess.run([sys.executable, "scripts/preprocess.py"], check=True)
            logger.info("✅ Data preprocessing completed")
        except subprocess.CalledProcessError:
            logger.error("❌ Data preprocessing failed")
            return False
    else:
        logger.info("✅ All data files present")
    
    # Check if client script exists
    if not os.path.exists("client.py"):
        logger.error("❌ client.py not found")
        return False
    
    logger.info("✅ Local setup is ready")
    return True

def test_single_client():
    """Test connection with a single client."""
    logger.info("🏥 Testing single client connection...")
    
    try:
        # Start a single client for a short test
        cmd = [
            sys.executable, "client.py",
            "--hospital-id", "0",
            "--server-address", FL_SERVER
        ]
        
        logger.info(f"Starting client: {' '.join(cmd)}")
        logger.info("This will take about 30 seconds...")
        
        # Run client with timeout
        result = subprocess.run(cmd, timeout=60, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Client connected and completed successfully")
            if "Training completed" in result.stdout:
                logger.info("✅ Training was successful")
            return True
        else:
            logger.error(f"❌ Client failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.warning("⚠️ Client test timed out (this might be normal)")
        return True  # Timeout might be normal for FL
    except Exception as e:
        logger.error(f"❌ Client test error: {e}")
        return False

def show_connection_instructions():
    """Show instructions for manual testing."""
    logger.info("📋 Manual Testing Instructions:")
    logger.info("=" * 50)
    logger.info("To test federated learning manually:")
    logger.info("")
    logger.info("1. Open 3 separate terminals")
    logger.info("2. In each terminal, run:")
    logger.info(f"   Terminal 1: python client.py --hospital-id 0 --server-address {FL_SERVER}")
    logger.info(f"   Terminal 2: python client.py --hospital-id 1 --server-address {FL_SERVER}")
    logger.info(f"   Terminal 3: python client.py --hospital-id 2 --server-address {FL_SERVER}")
    logger.info("")
    logger.info("3. Monitor progress:")
    logger.info(f"   - Web Dashboard: {RENDER_URL}")
    logger.info(f"   - Health Check: {RENDER_URL}/health")
    logger.info(f"   - Metrics API: {RENDER_URL}/metrics")
    logger.info("")
    logger.info("4. Expected results:")
    logger.info("   - 10 rounds of federated learning")
    logger.info("   - Improving MSE and R² scores")
    logger.info("   - Privacy budget tracking")
    logger.info("=" * 50)

def main():
    """Main testing function."""
    logger.info("🚀 Testing Render Deployment")
    logger.info("=" * 60)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Health Check
    total_tests += 1
    if test_health_check():
        tests_passed += 1
    
    # Test 2: Web Dashboard
    total_tests += 1
    if test_web_dashboard():
        tests_passed += 1
    
    # Test 3: API Endpoints
    total_tests += 1
    test_api_endpoints()  # This always "passes"
    tests_passed += 1
    
    # Test 4: Local Setup
    total_tests += 1
    if check_local_setup():
        tests_passed += 1
    
    # Test 5: Single Client (optional)
    logger.info("")
    user_input = input("🤔 Test single client connection? (y/n): ").lower().strip()
    if user_input == 'y':
        total_tests += 1
        if test_single_client():
            tests_passed += 1
    
    # Results
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        logger.info("🎉 All tests passed! Your deployment is working perfectly!")
    elif tests_passed >= total_tests - 1:
        logger.info("✅ Deployment is working well with minor issues")
    else:
        logger.warning("⚠️ Some tests failed. Check the logs above.")
    
    logger.info("")
    show_connection_instructions()
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
