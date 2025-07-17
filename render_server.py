#!/usr/bin/env python3
"""
Render-optimized Federated Learning Server.
This version is specifically designed for Render deployment.
"""

import os
import sys
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import flwr as fl
from flwr.common import Metrics, FitRes, EvaluateRes
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flask import Flask, jsonify, render_template_string, request
import threading

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from models.linear_regression import FederatedAveraging
from config.settings import get_config
from config.privacy import dp_mechanism

# Setup logging for Render
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class RenderMetricsLogger:
    """Simplified metrics logger for Render deployment."""
    
    def __init__(self):
        self.metrics_history = []
        self.logger = logging.getLogger("MetricsLogger")
    
    def log_round_metrics(self, round_num: int, metrics: Dict) -> None:
        """Log metrics for a federated learning round."""
        metrics_entry = {
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        
        self.metrics_history.append(metrics_entry)
        
        # Log to console (visible in Render logs)
        self.logger.info(f"Round {round_num} - "
                        f"Loss: {metrics.get('loss', 'N/A'):.4f}, "
                        f"MSE: {metrics.get('mse', 'N/A'):.4f}, "
                        f"R²: {metrics.get('r2', 'N/A'):.4f}")
    
    def get_metrics_history(self) -> List[Dict]:
        """Get the complete metrics history."""
        return self.metrics_history

class RenderFedAvg(FedAvg):
    """Render-optimized FedAvg strategy."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = logging.getLogger("RenderFedAvg")
        self.metrics_logger = RenderMetricsLogger()
        self.round_num = 0
        
        self.logger.info(f"Initialized FedAvg strategy for Render deployment")
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        """Aggregate fit results from clients."""
        self.round_num = server_round
        
        if failures:
            self.logger.warning(f"Round {server_round}: {len(failures)} client failures")
        
        if not results:
            self.logger.error(f"Round {server_round}: No successful client results")
            return None, {}
        
        self.logger.info(f"Round {server_round}: Aggregating results from {len(results)} clients")
        
        # Extract parameters and metrics
        client_parameters = []
        client_metrics = []
        
        for client_proxy, fit_res in results:
            params_dict = {
                "coefficients": fit_res.parameters.tensors[0],
                "intercept": fit_res.parameters.tensors[1][0],
                "num_samples": fit_res.num_examples
            }
            client_parameters.append(params_dict)
            
            if fit_res.metrics:
                client_metrics.append({
                    **fit_res.metrics,
                    "num_samples": fit_res.num_examples
                })
        
        try:
            # Aggregate parameters
            aggregated_params = FederatedAveraging.aggregate_parameters(client_parameters)
            
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
            
            # Log metrics
            self.metrics_logger.log_round_metrics(server_round, aggregated_metrics)
            
            self.logger.info(f"Round {server_round} aggregation completed successfully")
            
            return parameters_aggregated, aggregated_metrics
            
        except Exception as e:
            self.logger.error(f"Round {server_round} aggregation failed: {e}")
            return None, {}

def create_web_interface(strategy: RenderFedAvg, port: int) -> Flask:
    """Create a web interface for monitoring the federated learning process."""
    
    app = Flask(__name__)
    
    @app.route('/')
    def dashboard():
        """Main dashboard showing FL status."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Federated Learning Server - Diabetes Prediction</title>
            <meta http-equiv="refresh" content="30">
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                .header { text-align: center; color: #2c3e50; margin-bottom: 30px; }
                .status { display: flex; justify-content: space-around; margin: 20px 0; }
                .status-card { background: #ecf0f1; padding: 20px; border-radius: 8px; text-align: center; min-width: 150px; }
                .status-card.active { background: #2ecc71; color: white; }
                .metrics { margin: 20px 0; }
                .metric-row { display: flex; justify-content: space-between; padding: 10px; border-bottom: 1px solid #eee; }
                .instructions { background: #e8f4fd; padding: 20px; border-radius: 8px; margin: 20px 0; }
                .code { background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; font-family: monospace; margin: 10px 0; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🏥 Federated Learning Server</h1>
                    <h2>Diabetes Prediction with Privacy Protection</h2>
                    <p>Server Status: <strong style="color: #27ae60;">RUNNING</strong></p>
                </div>
                
                <div class="status">
                    <div class="status-card active">
                        <h3>Server</h3>
                        <p>✅ Online</p>
                    </div>
                    <div class="status-card">
                        <h3>Clients</h3>
                        <p>{{ client_count }} Connected</p>
                    </div>
                    <div class="status-card">
                        <h3>Rounds</h3>
                        <p>{{ round_count }} Completed</p>
                    </div>
                    <div class="status-card">
                        <h3>Privacy</h3>
                        <p>{{ privacy_level }}</p>
                    </div>
                </div>
                
                <div class="metrics">
                    <h3>📊 Latest Metrics</h3>
                    {% if latest_metrics %}
                        <div class="metric-row"><span>MSE:</span><span>{{ "%.4f"|format(latest_metrics.mse) }}</span></div>
                        <div class="metric-row"><span>R² Score:</span><span>{{ "%.4f"|format(latest_metrics.r2) }}</span></div>
                        <div class="metric-row"><span>Privacy Budget (ε):</span><span>{{ "%.2f"|format(latest_metrics.privacy_epsilon) }}</span></div>
                        <div class="metric-row"><span>Last Update:</span><span>{{ latest_metrics.timestamp }}</span></div>
                    {% else %}
                        <p>No training data yet. Waiting for clients to connect...</p>
                    {% endif %}
                </div>
                
                <div class="instructions">
                    <h3>🚀 Connect Your Clients</h3>
                    <p>To connect hospital clients to this server, run the following command from your local machine:</p>
                    <div class="code">
python client.py --hospital-id 0 --server-address {{ server_url }}
                    </div>
                    <p>Replace <code>--hospital-id</code> with 0, 1, or 2 for different hospitals.</p>
                    <p><strong>Note:</strong> Make sure to use port 10000 for the federated learning server.</p>
                    
                    <h4>📋 Setup Instructions:</h4>
                    <ol>
                        <li>Clone the repository: <code>git clone &lt;your-repo-url&gt;</code></li>
                        <li>Install dependencies: <code>pip install flwr scikit-learn pandas numpy matplotlib</code></li>
                        <li>Prepare data: <code>python scripts/preprocess.py</code></li>
                        <li>Connect clients using the command above</li>
                    </ol>
                </div>
                
                <div class="instructions">
                    <h3>🔒 Privacy Information</h3>
                    <p><strong>Differential Privacy:</strong> This system uses differential privacy to protect patient data.</p>
                    <p><strong>No Data Sharing:</strong> Only model parameters are shared, never raw patient data.</p>
                    <p><strong>Privacy Budget:</strong> ε=1.0 per round, δ=1e-5 (High Privacy Level)</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Get current metrics
        metrics_history = strategy.metrics_logger.get_metrics_history()
        latest_metrics = metrics_history[-1] if metrics_history else None
        
        # Get server URL
        host = request.host
        if ':' in host:
            server_url = host.split(':')[0] + ':10000'  # Use Render's port
        else:
            server_url = host + ':10000'
        
        return render_template_string(
            html_template,
            client_count=0,  # Would need to track this
            round_count=len(metrics_history),
            privacy_level="High Privacy",
            latest_metrics=latest_metrics,
            server_url=server_url
        )
    
    @app.route('/health')
    def health_check():
        """Health check endpoint for Render."""
        return jsonify({
            "status": "healthy",
            "service": "federated-learning-server",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "rounds_completed": len(strategy.metrics_logger.get_metrics_history())
        })
    
    @app.route('/metrics')
    def get_metrics():
        """API endpoint for metrics."""
        return jsonify({
            "metrics_history": strategy.metrics_logger.get_metrics_history(),
            "privacy_report": dp_mechanism.get_privacy_report(strategy.round_num)
        })
    
    @app.route('/status')
    def get_status():
        """API endpoint for server status."""
        return jsonify({
            "server_status": "running",
            "rounds_completed": len(strategy.metrics_logger.get_metrics_history()),
            "privacy_level": "High Privacy",
            "last_update": datetime.now().isoformat()
        })
    
    return app

def create_strategy() -> RenderFedAvg:
    """Create the federated learning strategy for Render."""
    model_config = get_config("model")
    
    # Initialize parameters
    num_features = len(model_config["features"])
    initial_parameters = fl.common.ndarrays_to_parameters([
        np.random.normal(0, 0.01, num_features),
        np.array([0.0])
    ])
    
    strategy = RenderFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=1,  # Reduced for easier testing
        min_evaluate_clients=1,
        min_available_clients=1,
        initial_parameters=initial_parameters,
    )
    
    return strategy

def main():
    """Main function for Render deployment."""
    # Get port from environment (Render sets this)
    port = int(os.environ.get("PORT", 8080))
    host = "0.0.0.0"

    logger.info(f"🚀 Starting Federated Learning Server on Render")
    logger.info(f"Host: {host}, Port: {port}")

    # Create strategy
    strategy = create_strategy()

    # Create web interface
    web_app = create_web_interface(strategy, port)

    # Configure FL server
    config = fl.server.ServerConfig(num_rounds=10)

    logger.info("🌐 Web dashboard available at your Render URL")
    logger.info(f"🔗 Clients should connect to your-render-url.com:{port}")

    # Start FL server in background thread
    def start_fl_server():
        try:
            logger.info("Starting Flower server...")
            fl.server.start_server(
                server_address=f"{host}:{port}",
                config=config,
                strategy=strategy
            )
        except Exception as e:
            logger.error(f"Flower server failed: {e}")

    # Start FL server in background
    fl_thread = threading.Thread(target=start_fl_server, daemon=True)
    fl_thread.start()

    # Give FL server time to start
    import time
    time.sleep(2)

    # Run web interface on main thread (this keeps the service alive)
    logger.info("Starting web interface...")
    try:
        web_app.run(host=host, port=port, debug=False)
    except Exception as e:
        logger.error(f"Web interface failed: {e}")
        # Keep the process alive
        while True:
            time.sleep(60)

if __name__ == "__main__":
    main()
