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