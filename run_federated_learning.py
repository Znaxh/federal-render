#!/usr/bin/env python3
"""
Complete federated learning system runner.
This script orchestrates the entire federated learning process.
"""

import os
import sys
import time
import signal
import subprocess
import threading
import logging
from typing import List, Optional
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from config.settings import get_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FederatedLearningRunner:
    """
    Orchestrates the complete federated learning process.
    """
    
    def __init__(self, num_clients: int = 3, rounds: int = 10):
        self.num_clients = num_clients
        self.rounds = rounds
        self.processes = []
        self.server_process = None
        
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met."""
        logger.info("Checking prerequisites...")
        
        # Check if data is preprocessed
        data_files = [f"data/hospital_{i}.csv" for i in range(self.num_clients)]
        missing_files = [f for f in data_files if not os.path.exists(f)]
        
        if missing_files:
            logger.error(f"Missing data files: {missing_files}")
            logger.info("Running data preprocessing...")
            try:
                subprocess.run([sys.executable, "scripts/preprocess.py"], check=True)
                logger.info("✅ Data preprocessing completed")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Data preprocessing failed: {e}")
                return False
        
        # Run system tests
        logger.info("Running system tests...")
        try:
            result = subprocess.run([sys.executable, "test_system.py"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("✅ All system tests passed")
            else:
                logger.error(f"❌ System tests failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to run system tests: {e}")
            return False
        
        return True
    
    def start_server(self) -> bool:
        """Start the federated learning server."""
        logger.info("Starting federated learning server...")
        
        try:
            server_cmd = [
                sys.executable, "server.py",
                "--rounds", str(self.rounds),
                "--host", "localhost",
                "--port", "8080"
            ]
            
            self.server_process = subprocess.Popen(
                server_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Give server time to start
            time.sleep(3)
            
            if self.server_process.poll() is None:
                logger.info("✅ Server started successfully")
                return True
            else:
                stdout, stderr = self.server_process.communicate()
                logger.error(f"❌ Server failed to start: {stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to start server: {e}")
            return False
    
    def start_clients(self) -> bool:
        """Start all federated learning clients."""
        logger.info(f"Starting {self.num_clients} federated learning clients...")
        
        # Wait a bit more for server to be ready
        time.sleep(2)
        
        for client_id in range(self.num_clients):
            try:
                client_cmd = [
                    sys.executable, "client.py",
                    "--hospital-id", str(client_id),
                    "--server-address", "localhost:8080"
                ]
                
                process = subprocess.Popen(
                    client_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                self.processes.append(process)
                logger.info(f"✅ Started client {client_id}")
                
                # Small delay between client starts
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Failed to start client {client_id}: {e}")
                return False
        
        return True
    
    def monitor_progress(self) -> None:
        """Monitor the federated learning progress."""
        logger.info("Monitoring federated learning progress...")
        
        # Monitor server process
        def monitor_server():
            if self.server_process:
                stdout, stderr = self.server_process.communicate()
                if stdout:
                    logger.info(f"Server output: {stdout}")
                if stderr and self.server_process.returncode != 0:
                    logger.error(f"Server error: {stderr}")
        
        # Start server monitoring in background
        server_thread = threading.Thread(target=monitor_server)
        server_thread.daemon = True
        server_thread.start()
        
        # Monitor clients
        active_clients = len(self.processes)
        while active_clients > 0:
            time.sleep(5)
            active_clients = sum(1 for p in self.processes if p.poll() is None)
            logger.info(f"Active clients: {active_clients}")
            
            # Check if server is still running
            if self.server_process and self.server_process.poll() is not None:
                logger.info("Server process completed")
                break
        
        logger.info("Federated learning process completed")
    
    def cleanup(self) -> None:
        """Clean up all processes."""
        logger.info("Cleaning up processes...")
        
        # Terminate clients
        for i, process in enumerate(self.processes):
            if process.poll() is None:
                logger.info(f"Terminating client {i}")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        
        # Terminate server
        if self.server_process and self.server_process.poll() is None:
            logger.info("Terminating server")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
        
        logger.info("✅ Cleanup completed")
    
    def generate_results(self) -> None:
        """Generate visualization and results."""
        logger.info("Generating results and visualizations...")
        
        try:
            # Generate visualizations
            subprocess.run([sys.executable, "scripts/visualize.py"], check=True)
            logger.info("✅ Visualizations generated")
            
            # Check if results exist
            results_dir = "results"
            if os.path.exists(results_dir):
                result_files = os.listdir(results_dir)
                logger.info(f"Generated files: {result_files}")
            
            # Display summary
            metrics_file = "logs/metrics.json"
            if os.path.exists(metrics_file):
                import json
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                if metrics:
                    final_metrics = metrics[-1]
                    logger.info("📊 FINAL RESULTS:")
                    logger.info(f"   Rounds completed: {len(metrics)}")
                    logger.info(f"   Final MSE: {final_metrics.get('mse', 'N/A'):.4f}")
                    logger.info(f"   Final R²: {final_metrics.get('r2', 'N/A'):.4f}")
                    logger.info(f"   Privacy level: {final_metrics.get('privacy_level', 'N/A')}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate results: {e}")
    
    def run(self) -> bool:
        """Run the complete federated learning process."""
        logger.info("🚀 Starting Federated Learning System")
        logger.info("=" * 60)
        
        try:
            # Check prerequisites
            if not self.check_prerequisites():
                return False
            
            # Start server
            if not self.start_server():
                return False
            
            # Start clients
            if not self.start_clients():
                return False
            
            # Monitor progress
            self.monitor_progress()
            
            # Generate results
            self.generate_results()
            
            logger.info("🎉 Federated learning completed successfully!")
            return True
            
        except KeyboardInterrupt:
            logger.info("⚠️ Process interrupted by user")
            return False
        except Exception as e:
            logger.error(f"❌ Federated learning failed: {e}")
            return False
        finally:
            self.cleanup()

def signal_handler(signum, frame):
    """Handle interrupt signals."""
    logger.info("Received interrupt signal, cleaning up...")
    sys.exit(0)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Federated Learning System Runner")
    parser.add_argument("--clients", type=int, default=3,
                       help="Number of client hospitals (default: 3)")
    parser.add_argument("--rounds", type=int, default=10,
                       help="Number of federated learning rounds (default: 10)")
    parser.add_argument("--test-only", action="store_true",
                       help="Run system tests only")
    parser.add_argument("--preprocess-only", action="store_true",
                       help="Run data preprocessing only")
    parser.add_argument("--visualize-only", action="store_true",
                       help="Generate visualizations only")
    
    args = parser.parse_args()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        if args.test_only:
            logger.info("Running system tests only...")
            result = subprocess.run([sys.executable, "test_system.py"])
            sys.exit(result.returncode)
        
        if args.preprocess_only:
            logger.info("Running data preprocessing only...")
            result = subprocess.run([sys.executable, "scripts/preprocess.py"])
            sys.exit(result.returncode)
        
        if args.visualize_only:
            logger.info("Generating visualizations only...")
            result = subprocess.run([sys.executable, "scripts/visualize.py"])
            sys.exit(result.returncode)
        
        # Run complete federated learning
        runner = FederatedLearningRunner(
            num_clients=args.clients,
            rounds=args.rounds
        )
        
        success = runner.run()
        
        if success:
            print("\n" + "=" * 60)
            print("🎉 FEDERATED LEARNING COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print("📁 Check the following directories for results:")
            print("   - logs/: Training logs and metrics")
            print("   - results/: Visualizations and plots")
            print("   - notebooks/demo.ipynb: Interactive demo")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("❌ FEDERATED LEARNING FAILED")
            print("=" * 60)
            print("🔍 Check the logs for error details")
            print("🧪 Run 'python test_system.py' to diagnose issues")
            print("=" * 60)
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
