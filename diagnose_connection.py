#!/usr/bin/env python3
"""
Diagnostic script to troubleshoot client connection issues.
"""

import socket
import requests
import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RENDER_HOST = "federated-learning-server-xbx0.onrender.com"
RENDER_URL = f"https://{RENDER_HOST}"

def test_dns_resolution():
    """Test DNS resolution."""
    logger.info("🔍 Testing DNS resolution...")
    try:
        import socket
        ip = socket.gethostbyname(RENDER_HOST)
        logger.info(f"✅ DNS resolved: {RENDER_HOST} -> {ip}")
        return True
    except Exception as e:
        logger.error(f"❌ DNS resolution failed: {e}")
        return False

def test_http_connection():
    """Test HTTP connection to web interface."""
    logger.info("🌐 Testing HTTP connection...")
    try:
        response = requests.get(f"{RENDER_URL}/health", timeout=10)
        if response.status_code == 200:
            logger.info("✅ HTTP connection successful")
            return True
        else:
            logger.error(f"❌ HTTP returned {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ HTTP connection failed: {e}")
        return False

def test_port_connectivity(port):
    """Test if a specific port is reachable."""
    logger.info(f"🔌 Testing port {port} connectivity...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        result = sock.connect_ex((RENDER_HOST, port))
        sock.close()
        
        if result == 0:
            logger.info(f"✅ Port {port} is reachable")
            return True
        else:
            logger.error(f"❌ Port {port} is not reachable (error code: {result})")
            return False
    except Exception as e:
        logger.error(f"❌ Port {port} test failed: {e}")
        return False

def check_render_service_info():
    """Get information about the Render service."""
    logger.info("📊 Checking Render service info...")
    try:
        response = requests.get(f"{RENDER_URL}/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ Service status: {data}")
            return True
        else:
            logger.warning(f"⚠️ Status endpoint returned {response.status_code}")
            return False
    except Exception as e:
        logger.warning(f"⚠️ Could not get service status: {e}")
        return False

def test_grpc_connection():
    """Test gRPC connection (what Flower uses)."""
    logger.info("🔗 Testing gRPC connection...")
    try:
        import grpc
        
        # Test different ports
        ports_to_test = [10000, 8080, 443, 80]
        
        for port in ports_to_test:
            try:
                channel = grpc.insecure_channel(f"{RENDER_HOST}:{port}")
                grpc.channel_ready_future(channel).result(timeout=5)
                logger.info(f"✅ gRPC connection successful on port {port}")
                channel.close()
                return True
            except Exception as e:
                logger.debug(f"Port {port} failed: {e}")
                continue
        
        logger.error("❌ No gRPC ports are accessible")
        return False
        
    except ImportError:
        logger.warning("⚠️ grpc not installed, skipping gRPC test")
        return False
    except Exception as e:
        logger.error(f"❌ gRPC test failed: {e}")
        return False

def suggest_solutions():
    """Suggest solutions based on test results."""
    logger.info("💡 Suggested Solutions:")
    logger.info("=" * 50)
    
    logger.info("🔧 Issue: Render only exposes one port (the web port)")
    logger.info("   Flower clients need gRPC access, but Render blocks additional ports")
    logger.info("")
    
    logger.info("✅ Solution Options:")
    logger.info("1. Use HTTP-based federated learning instead of gRPC")
    logger.info("2. Run clients locally and use ngrok to expose local server")
    logger.info("3. Use a different deployment platform (Railway, Fly.io)")
    logger.info("4. Run the demo locally with the quick_demo.py script")
    logger.info("")
    
    logger.info("🚀 Immediate workaround:")
    logger.info("   python quick_demo.py  # Simulates FL locally")
    logger.info("")
    
    logger.info("🌐 Alternative: Local server with ngrok")
    logger.info("   1. Run: python server.py")
    logger.info("   2. In another terminal: ngrok http 8080")
    logger.info("   3. Use ngrok URL for client connections")

def main():
    """Main diagnostic function."""
    logger.info("🔍 Diagnosing Federated Learning Connection Issues")
    logger.info("=" * 60)
    
    tests = [
        ("DNS Resolution", test_dns_resolution),
        ("HTTP Connection", test_http_connection),
        ("Port 10000", lambda: test_port_connectivity(10000)),
        ("Port 8080", lambda: test_port_connectivity(8080)),
        ("Port 443 (HTTPS)", lambda: test_port_connectivity(443)),
        ("Service Status", check_render_service_info),
        ("gRPC Connection", test_grpc_connection),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 DIAGNOSTIC SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name:20} {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed < total:
        logger.info("\n🔍 DIAGNOSIS:")
        if not any(name == "Port 10000" and result for name, result in results):
            logger.info("❌ Port 10000 is not accessible from external clients")
            logger.info("   This is the main issue - Render doesn't expose this port")
        
        if any(name == "HTTP Connection" and result for name, result in results):
            logger.info("✅ Web interface is working")
            logger.info("   The server is running, but gRPC port is blocked")
    
    logger.info("")
    suggest_solutions()
    
    return passed >= total - 2  # Allow some failures

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
