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