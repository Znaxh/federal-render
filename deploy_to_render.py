#!/usr/bin/env python3
"""
Automated deployment script for Render.
Prepares the repository for Render deployment.
"""

import os
import sys
import json
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_for_render():
    """Prepare the repository for Render deployment."""
    
    logger.info("🚀 Preparing repository for Render deployment...")
    
    # Create render.yaml for automatic deployment
    render_config = {
        "services": [
            {
                "type": "web",
                "name": "federated-learning-server",
                "env": "python",
                "plan": "free",
                "buildCommand": "pip install -r render_requirements.txt",
                "startCommand": "python render_server.py",
                "envVars": [
                    {"key": "FL_ROUNDS", "value": "10"},
                    {"key": "FL_MIN_CLIENTS", "value": "1"},
                    {"key": "FL_MIN_AVAILABLE_CLIENTS", "value": "1"},
                    {"key": "DP_EPSILON", "value": "1.0"},
                    {"key": "DP_DELTA", "value": "1e-5"},
                    {"key": "LOG_LEVEL", "value": "INFO"}
                ],
                "healthCheckPath": "/health"
            }
        ]
    }
    
    # Save render.yaml
    with open("render.yaml", "w") as f:
        import yaml
        yaml.dump(render_config, f, default_flow_style=False)
    
    logger.info("✅ Created render.yaml")
    
    # Create .gitignore for deployment
    gitignore_content = """
# Deployment specific
logs/
results/
data/hospital_*.csv
*.log

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content.strip())
    
    logger.info("✅ Updated .gitignore")
    
    # Create deployment README
    deployment_readme = """
# Render Deployment

This repository is configured for automatic deployment to Render.

## Quick Deploy

1. Connect this repository to Render
2. Render will automatically use the `render.yaml` configuration
3. Your service will be available at the provided URL

## Manual Deploy

1. Create new Web Service on Render
2. Build Command: `pip install -r render_requirements.txt`
3. Start Command: `python render_server.py`
4. Set environment variables as specified in render.yaml

## Connecting Clients

```bash
python client.py --hospital-id 0 --server-address your-service.onrender.com:8080
```
"""
    
    with open("RENDER_DEPLOY.md", "w") as f:
        f.write(deployment_readme.strip())
    
    logger.info("✅ Created RENDER_DEPLOY.md")
    
    return True

def check_git_status():
    """Check if repository is ready for deployment."""
    
    logger.info("🔍 Checking git status...")
    
    try:
        # Check if git is initialized
        result = subprocess.run(["git", "status"], capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning("Git not initialized. Run: git init")
            return False
        
        # Check for uncommitted changes
        if "nothing to commit" not in result.stdout:
            logger.warning("You have uncommitted changes. Commit them before deploying.")
            return False
        
        # Check for remote
        result = subprocess.run(["git", "remote", "-v"], capture_output=True, text=True)
        if not result.stdout:
            logger.warning("No git remote configured. Add your GitHub repository:")
            logger.warning("git remote add origin https://github.com/username/repo.git")
            return False
        
        logger.info("✅ Git repository is ready")
        return True
        
    except FileNotFoundError:
        logger.error("Git not found. Please install git.")
        return False

def create_deployment_checklist():
    """Create a deployment checklist."""
    
    checklist = """
# 🚀 Render Deployment Checklist

## Pre-deployment
- [ ] Code is committed to git
- [ ] Repository is pushed to GitHub
- [ ] All tests pass: `python test_system.py`

## Render Setup
- [ ] Created Render account
- [ ] Connected GitHub repository
- [ ] Created Web Service
- [ ] Set environment variables
- [ ] Deployment successful

## Testing
- [ ] Health check works: `curl https://your-service.onrender.com/health`
- [ ] Web dashboard accessible
- [ ] Clients can connect from local machine

## Client Connection
```bash
# Install dependencies locally
pip install flwr scikit-learn pandas numpy matplotlib

# Prepare data
python scripts/preprocess.py

# Connect clients
python client.py --hospital-id 0 --server-address your-service.onrender.com:8080
```

## Troubleshooting
- Check Render logs for errors
- Verify environment variables are set
- Ensure repository has latest changes
- Test locally first: `python render_server.py`
"""
    
    with open("DEPLOYMENT_CHECKLIST.md", "w") as f:
        f.write(checklist.strip())
    
    logger.info("✅ Created DEPLOYMENT_CHECKLIST.md")

def main():
    """Main deployment preparation function."""
    
    logger.info("🚀 Render Deployment Preparation")
    logger.info("=" * 50)
    
    try:
        # Prepare files for Render
        prepare_for_render()
        
        # Check git status
        git_ready = check_git_status()
        
        # Create checklist
        create_deployment_checklist()
        
        logger.info("=" * 50)
        logger.info("✅ Render deployment preparation complete!")
        logger.info("")
        logger.info("📋 Next steps:")
        logger.info("1. Review DEPLOYMENT_CHECKLIST.md")
        logger.info("2. Commit and push changes to GitHub")
        logger.info("3. Go to render.com and create a new Web Service")
        logger.info("4. Connect your GitHub repository")
        logger.info("5. Use the configuration from render.yaml")
        logger.info("")
        logger.info("🌐 Your service will be available at:")
        logger.info("   https://your-service-name.onrender.com")
        
        if not git_ready:
            logger.warning("")
            logger.warning("⚠️  Git repository needs attention before deploying")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Deployment preparation failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
