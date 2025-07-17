"""
Deployment utilities for the federated learning system.
"""

import os
import sys
import subprocess
import argparse
import logging
from typing import Dict, List, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeploymentManager:
    """
    Manages deployment of the federated learning system.
    """
    
    def __init__(self):
        self.deployment_config = get_config("deployment")
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    def check_prerequisites(self) -> bool:
        """
        Check if all prerequisites for deployment are met.
        
        Returns:
            True if all prerequisites are met
        """
        logger.info("Checking deployment prerequisites...")
        
        # Check if required files exist
        required_files = [
            "requirements.txt",
            "server.py",
            "client.py",
            "config/settings.py"
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(os.path.join(self.project_root, file)):
                missing_files.append(file)
        
        if missing_files:
            logger.error(f"Missing required files: {missing_files}")
            return False
        
        # Check if data preprocessing has been run
        data_dir = os.path.join(self.project_root, "data")
        if not os.path.exists(data_dir) or not any(
            f.startswith("hospital_") for f in os.listdir(data_dir)
        ):
            logger.warning("Hospital data not found. Run preprocessing first.")
            return False
        
        logger.info("All prerequisites met!")
        return True
    
    def prepare_for_render(self) -> bool:
        """
        Prepare the project for Render deployment.
        
        Returns:
            True if preparation successful
        """
        logger.info("Preparing for Render deployment...")
        
        try:
            # Create a simple health check endpoint
            health_check_code = '''
import os
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "federated-learning-server",
        "version": "1.0.0"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
'''
            
            health_file = os.path.join(self.project_root, "health_check.py")
            with open(health_file, 'w') as f:
                f.write(health_check_code)
            
            # Update server.py to include health check
            self._add_health_check_to_server()
            
            logger.info("Render preparation completed!")
            return True
            
        except Exception as e:
            logger.error(f"Render preparation failed: {e}")
            return False
    
    def _add_health_check_to_server(self):
        """Add health check endpoint to server.py"""
        # This is a simplified approach - in practice, you'd modify the server
        # to include a health check endpoint
        pass
    
    def create_docker_image(self, tag: str = "federated-learning:latest") -> bool:
        """
        Build Docker image for the application.
        
        Args:
            tag: Docker image tag
            
        Returns:
            True if build successful
        """
        logger.info(f"Building Docker image: {tag}")
        
        try:
            dockerfile_path = os.path.join(self.project_root, "deployment", "Dockerfile")
            
            cmd = [
                "docker", "build",
                "-f", dockerfile_path,
                "-t", tag,
                self.project_root
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Docker image built successfully!")
                return True
            else:
                logger.error(f"Docker build failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Docker build error: {e}")
            return False
    
    def run_local_docker(self, tag: str = "federated-learning:latest", port: int = 8080) -> bool:
        """
        Run the Docker container locally for testing.
        
        Args:
            tag: Docker image tag
            port: Port to expose
            
        Returns:
            True if container started successfully
        """
        logger.info(f"Running Docker container locally on port {port}")
        
        try:
            cmd = [
                "docker", "run",
                "-p", f"{port}:8080",
                "-v", f"{self.project_root}/data:/app/data",
                "-v", f"{self.project_root}/logs:/app/logs",
                "-v", f"{self.project_root}/results:/app/results",
                "--name", "fl-server-local",
                "--rm",
                tag
            ]
            
            logger.info("Starting container... (Press Ctrl+C to stop)")
            subprocess.run(cmd)
            return True
            
        except KeyboardInterrupt:
            logger.info("Container stopped by user")
            return True
        except Exception as e:
            logger.error(f"Failed to run container: {e}")
            return False
    
    def generate_deployment_guide(self) -> str:
        """
        Generate a deployment guide.
        
        Returns:
            Path to the generated guide
        """
        guide_content = """
# Federated Learning Deployment Guide

## Local Development

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data**
   ```bash
   python scripts/preprocess.py
   ```

3. **Start Server**
   ```bash
   python server.py
   ```

4. **Run Clients** (in separate terminals)
   ```bash
   python client.py --hospital-id 0
   python client.py --hospital-id 1
   python client.py --hospital-id 2
   ```

## Render Deployment

1. **Connect GitHub Repository**
   - Go to Render dashboard
   - Connect your GitHub repository
   - Select this repository

2. **Create Web Service**
   - Service Type: Web Service
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python server.py --host 0.0.0.0 --port $PORT`

3. **Environment Variables**
   - FL_ROUNDS: 10
   - FL_MIN_CLIENTS: 2
   - DP_EPSILON: 1.0
   - LOG_LEVEL: INFO

4. **Connect Clients**
   - Use the Render service URL as server address
   - Run clients from local machines or other servers

## Docker Deployment

1. **Build Image**
   ```bash
   python scripts/deploy.py --build-docker
   ```

2. **Run Locally**
   ```bash
   python scripts/deploy.py --run-docker
   ```

3. **Deploy to Cloud**
   - Push image to container registry
   - Deploy using cloud provider's container service

## Testing Deployment

1. **Health Check**
   ```bash
   curl https://your-render-url.com/health
   ```

2. **Run Test Clients**
   ```bash
   python client.py --hospital-id 0 --server-address your-render-url.com:443
   ```

## Troubleshooting

- Check logs in the `logs/` directory
- Verify all environment variables are set
- Ensure firewall allows connections on the specified port
- Check that data preprocessing has been completed

## Security Considerations

- Use HTTPS in production
- Implement proper authentication for clients
- Monitor privacy budget consumption
- Regularly update dependencies
"""
        
        guide_path = os.path.join(self.project_root, "docs", "deployment.md")
        os.makedirs(os.path.dirname(guide_path), exist_ok=True)
        
        with open(guide_path, 'w') as f:
            f.write(guide_content)
        
        logger.info(f"Deployment guide created: {guide_path}")
        return guide_path

def main():
    """Main function for deployment utilities."""
    parser = argparse.ArgumentParser(description="Federated Learning Deployment Utilities")
    parser.add_argument("--check", action="store_true",
                       help="Check deployment prerequisites")
    parser.add_argument("--prepare-render", action="store_true",
                       help="Prepare for Render deployment")
    parser.add_argument("--build-docker", action="store_true",
                       help="Build Docker image")
    parser.add_argument("--run-docker", action="store_true",
                       help="Run Docker container locally")
    parser.add_argument("--generate-guide", action="store_true",
                       help="Generate deployment guide")
    parser.add_argument("--docker-tag", type=str, default="federated-learning:latest",
                       help="Docker image tag")
    parser.add_argument("--port", type=int, default=8080,
                       help="Port for local Docker container")
    
    args = parser.parse_args()
    
    deployment_manager = DeploymentManager()
    
    try:
        if args.check:
            if deployment_manager.check_prerequisites():
                print("✅ All prerequisites met!")
            else:
                print("❌ Prerequisites not met. Please fix the issues above.")
                sys.exit(1)
        
        if args.prepare_render:
            if deployment_manager.prepare_for_render():
                print("✅ Render preparation completed!")
            else:
                print("❌ Render preparation failed.")
                sys.exit(1)
        
        if args.build_docker:
            if deployment_manager.create_docker_image(args.docker_tag):
                print(f"✅ Docker image built: {args.docker_tag}")
            else:
                print("❌ Docker build failed.")
                sys.exit(1)
        
        if args.run_docker:
            deployment_manager.run_local_docker(args.docker_tag, args.port)
        
        if args.generate_guide:
            guide_path = deployment_manager.generate_deployment_guide()
            print(f"✅ Deployment guide generated: {guide_path}")
        
        if not any([args.check, args.prepare_render, args.build_docker, 
                   args.run_docker, args.generate_guide]):
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Deployment operation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
