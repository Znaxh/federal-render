"""
Federated Learning Server for Diabetes Prediction.
Implements Flower server with FedAvg strategy and performance monitoring.
"""

import argparse
import logging
import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import flwr as fl
from flwr.common import Metrics, FitRes, EvaluateRes
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
import threading
from flask import Flask, jsonify

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from models.linear_regression import FederatedAveraging
from config.settings import get_config
from config.privacy import dp_mechanism

# Setup logging
def setup_logging():
    """Setup logging for the server."""
    log_config = get_config("logging")
    log_file = log_config["server_log_file"]
    
    # Create logs directory
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_config["level"]),
        format=log_config["format"],
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class MetricsLogger:
    """
    Handles logging and storage of federated learning metrics.
    """
    
    def __init__(self):
        self.metrics_history = []
        self.log_config = get_config("logging")
        self.metrics_file = self.log_config["metrics_file"]
        
        # Create logs directory
        os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)
        
        self.logger = logging.getLogger("MetricsLogger")
    
    def log_round_metrics(self, round_num: int, metrics: Dict) -> None:
        """
        Log metrics for a federated learning round.
        
        Args:
            round_num: Round number
            metrics: Dictionary of metrics
        """
        # Add timestamp and round info
        metrics_entry = {
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        
        self.metrics_history.append(metrics_entry)
        
        # Save to file
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        # Log to console
        self.logger.info(f"Round {round_num} - "
                        f"Loss: {metrics.get('loss', 'N/A'):.4f}, "
                        f"MSE: {metrics.get('mse', 'N/A'):.4f}, "
                        f"R²: {metrics.get('r2', 'N/A'):.4f}")
    
    def get_metrics_history(self) -> List[Dict]:
        """Get the complete metrics history."""
        return self.metrics_history

class CustomFedAvg(FedAvg):
    """
    Custom FedAvg strategy with enhanced logging and privacy reporting.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = logging.getLogger("CustomFedAvg")
        self.metrics_logger = MetricsLogger()
        self.round_num = 0
        
        # Log strategy configuration
        self.logger.info(f"Initialized FedAvg strategy with {kwargs}")
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        """
        Aggregate fit results from clients.
        
        Args:
            server_round: Current round number
            results: Fit results from clients
            failures: Failed clients
            
        Returns:
            Aggregated parameters and metrics
        """
        self.round_num = server_round
        
        if failures:
            self.logger.warning(f"Round {server_round}: {len(failures)} client failures")
        
        if not results:
            self.logger.error(f"Round {server_round}: No successful client results")
            return None, {}
        
        # Log client participation
        self.logger.info(f"Round {server_round}: Aggregating results from "
                        f"{len(results)} clients")
        
        # Extract parameters and metrics
        client_parameters = []
        client_metrics = []
        
        for client_proxy, fit_res in results:
            # Convert parameters
            params_dict = {
                "coefficients": fit_res.parameters.tensors[0],
                "intercept": fit_res.parameters.tensors[1][0],
                "num_samples": fit_res.num_examples
            }
            client_parameters.append(params_dict)
            
            # Extract metrics
            if fit_res.metrics:
                client_metrics.append({
                    **fit_res.metrics,
                    "num_samples": fit_res.num_examples
                })
        
        # Aggregate parameters using FedAvg
        try:
            aggregated_params = FederatedAveraging.aggregate_parameters(client_parameters)
            
            # Convert back to Flower format
            parameters_aggregated = fl.common.ndarrays_to_parameters([
                aggregated_params["coefficients"],
                np.array([aggregated_params["intercept"]])
            ])
            
            # Aggregate metrics
            aggregated_metrics = FederatedAveraging.aggregate_metrics(client_metrics)
            
            # Add privacy information
            privacy_report = dp_mechanism.get_privacy_report(server_round)
            aggregated_metrics.update({
                "privacy_epsilon": privacy_report["total_epsilon"],
                "privacy_delta": privacy_report["total_delta"],
                "privacy_level": privacy_report["privacy_level"]
            })
            
            # Log aggregated metrics
            self.metrics_logger.log_round_metrics(server_round, aggregated_metrics)
            
            self.logger.info(f"Round {server_round} aggregation completed successfully")
            
            return parameters_aggregated, aggregated_metrics
            
        except Exception as e:
            self.logger.error(f"Round {server_round} aggregation failed: {e}")
            return None, {}
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, fl.common.Scalar]]:
        """
        Aggregate evaluation results from clients.
        
        Args:
            server_round: Current round number
            results: Evaluation results from clients
            failures: Failed clients
            
        Returns:
            Aggregated loss and metrics
        """
        if failures:
            self.logger.warning(f"Round {server_round}: {len(failures)} evaluation failures")
        
        if not results:
            self.logger.warning(f"Round {server_round}: No evaluation results")
            return None, {}
        
        # Extract metrics
        client_metrics = []
        total_samples = 0
        
        for client_proxy, eval_res in results:
            if eval_res.metrics:
                client_metrics.append({
                    **eval_res.metrics,
                    "num_samples": eval_res.num_examples,
                    "loss": eval_res.loss
                })
                total_samples += eval_res.num_examples
        
        if not client_metrics:
            return None, {}
        
        # Aggregate evaluation metrics
        aggregated_metrics = FederatedAveraging.aggregate_metrics(client_metrics)
        
        # Calculate weighted average loss
        weighted_loss = sum(
            metrics["loss"] * metrics["num_samples"] 
            for metrics in client_metrics
        ) / total_samples
        
        self.logger.info(f"Round {server_round} evaluation - "
                        f"Loss: {weighted_loss:.4f}, "
                        f"MSE: {aggregated_metrics.get('mse', 'N/A'):.4f}")
        
        return weighted_loss, aggregated_metrics

def create_health_check_server(port: int) -> None:
    """
    Create a simple health check server for deployment platforms.
    """
    app = Flask(__name__)

    @app.route('/health')
    def health_check():
        return jsonify({
            "status": "healthy",
            "service": "federated-learning-server",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        })

    @app.route('/')
    def root():
        return jsonify({
            "message": "Federated Learning Server",
            "status": "running",
            "endpoints": ["/health"]
        })

    # Run health check server in background
    app.run(host='0.0.0.0', port=port + 1, debug=False, use_reloader=False)

def create_strategy() -> CustomFedAvg:
    """
    Create the federated learning strategy.
    
    Returns:
        CustomFedAvg strategy instance
    """
    server_config = get_config("server")
    model_config = get_config("model")
    
    # Initialize parameters (random initialization)
    num_features = len(model_config["features"])
    initial_parameters = fl.common.ndarrays_to_parameters([
        np.random.normal(0, 0.01, num_features),  # coefficients
        np.array([0.0])  # intercept
    ])
    
    strategy = CustomFedAvg(
        fraction_fit=1.0,  # Use all available clients for training
        fraction_evaluate=1.0,  # Use all available clients for evaluation
        min_fit_clients=server_config["min_fit_clients"],
        min_evaluate_clients=server_config["min_evaluate_clients"],
        min_available_clients=server_config["min_available_clients"],
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=lambda metrics: {
            "accuracy": sum(m["r2"] * m["num_samples"] for m in metrics) / 
                       sum(m["num_samples"] for m in metrics)
        }
    )
    
    return strategy

def main():
    """Main function to run the server."""
    parser = argparse.ArgumentParser(description="Federated Learning Server")
    parser.add_argument("--host", type=str, 
                       default=None,
                       help="Server host (default from config)")
    parser.add_argument("--port", type=int,
                       default=None,
                       help="Server port (default from config)")
    parser.add_argument("--rounds", type=int,
                       default=None,
                       help="Number of FL rounds (default from config)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger = setup_logging()
    
    try:
        # Get configuration
        server_config = get_config("server")
        
        host = args.host or server_config["host"]
        port = args.port or server_config["port"]
        rounds = args.rounds or server_config["rounds"]
        
        logger.info(f"Starting Federated Learning Server")
        logger.info(f"Host: {host}, Port: {port}, Rounds: {rounds}")
        
        # Generate privacy report
        privacy_report = dp_mechanism.get_privacy_report(rounds)
        logger.info(f"Privacy Configuration:")
        logger.info(f"  - Privacy Level: {privacy_report['privacy_level']}")
        logger.info(f"  - Per-round ε: {privacy_report['per_round_epsilon']:.4f}")
        logger.info(f"  - Total ε: {privacy_report['total_epsilon']:.4f}")
        logger.info(f"  - Recommendations: {privacy_report['recommendations']}")
        
        # Create strategy
        strategy = create_strategy()

        # Start health check server in background
        health_thread = threading.Thread(
            target=create_health_check_server,
            args=(port,),
            daemon=True
        )
        health_thread.start()
        logger.info(f"Health check server started on port {port + 1}")

        # Configure server
        config = fl.server.ServerConfig(num_rounds=rounds)

        # Start server
        logger.info("Server starting...")
        fl.server.start_server(
            server_address=f"{host}:{port}",
            config=config,
            strategy=strategy
        )
        
        logger.info("Federated learning completed successfully!")
        
        # Print final summary
        metrics_history = strategy.metrics_logger.get_metrics_history()
        if metrics_history:
            final_metrics = metrics_history[-1]
            print("\n" + "="*60)
            print("FEDERATED LEARNING SUMMARY")
            print("="*60)
            print(f"Total Rounds: {len(metrics_history)}")
            print(f"Final MSE: {final_metrics.get('mse', 'N/A'):.4f}")
            print(f"Final R²: {final_metrics.get('r2', 'N/A'):.4f}")
            print(f"Privacy Level: {final_metrics.get('privacy_level', 'N/A')}")
            print(f"Total Privacy Budget: ε={final_metrics.get('privacy_epsilon', 'N/A'):.4f}")
            print("="*60)
        
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
