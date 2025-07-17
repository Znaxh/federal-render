# Deployment Guide - Federated Learning System

This guide provides step-by-step instructions for deploying the federated learning system to various platforms.

## 🚀 Render Deployment (Recommended for Students)

Render offers free hosting with HTTPS, making it perfect for student projects.

### Prerequisites
- GitHub account
- Render account (free)
- Project pushed to GitHub repository

### Step 1: Prepare Repository

1. **Push to GitHub**:
```bash
git add .
git commit -m "Initial federated learning system"
git push origin main
```

2. **Verify Files**:
Ensure these files are in your repository:
- `requirements.txt`
- `server.py`
- `deployment/render.yaml`
- `config/settings.py`

### Step 2: Create Render Service

1. **Login to Render**:
   - Go to [render.com](https://render.com)
   - Sign up/login with GitHub

2. **Create New Web Service**:
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the federated learning repository

3. **Configure Service**:
   ```
   Name: federated-learning-server
   Environment: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: python server.py --host 0.0.0.0 --port $PORT
   ```

### Step 3: Environment Variables

Add these environment variables in Render dashboard:

```
FL_SERVER_HOST=0.0.0.0
FL_ROUNDS=10
FL_MIN_CLIENTS=2
FL_MIN_AVAILABLE_CLIENTS=3
DP_EPSILON=1.0
DP_DELTA=1e-5
LOG_LEVEL=INFO
```

### Step 4: Deploy

1. Click "Create Web Service"
2. Wait for deployment (5-10 minutes)
3. Note your service URL: `https://your-service-name.onrender.com`

### Step 5: Connect Clients

Update client configuration to use your Render URL:

```bash
python client.py --hospital-id 0 --server-address your-service-name.onrender.com:443
```

## 🐳 Docker Deployment

### Build Docker Image

```bash
# Build the image
python scripts/deploy.py --build-docker

# Or manually:
docker build -f deployment/Dockerfile -t federated-learning .
```

### Run Locally

```bash
# Run with Docker
python scripts/deploy.py --run-docker

# Or manually:
docker run -p 8080:8080 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/results:/app/results \
  federated-learning
```

### Deploy to Cloud

1. **Push to Registry**:
```bash
# Tag for registry
docker tag federated-learning your-registry/federated-learning:latest

# Push to registry
docker push your-registry/federated-learning:latest
```

2. **Deploy to Cloud Provider**:
   - **Google Cloud Run**: Use the pushed image
   - **AWS ECS**: Create task definition with the image
   - **Azure Container Instances**: Deploy from registry

## ☁️ Other Cloud Platforms

### Heroku

1. **Install Heroku CLI**
2. **Create Heroku App**:
```bash
heroku create your-fl-app
```

3. **Add Buildpack**:
```bash
heroku buildpacks:set heroku/python
```

4. **Create Procfile**:
```
web: python server.py --host 0.0.0.0 --port $PORT
```

5. **Deploy**:
```bash
git push heroku main
```

### Railway

1. **Connect GitHub repository**
2. **Configure build**:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python server.py --host 0.0.0.0 --port $PORT`

### Fly.io

1. **Install flyctl**
2. **Initialize app**:
```bash
fly launch
```

3. **Deploy**:
```bash
fly deploy
```

## 🔧 Production Considerations

### Security

1. **HTTPS Only**:
   - All cloud platforms provide HTTPS by default
   - Never use HTTP for healthcare data

2. **Authentication**:
```python
# Add to server.py
def authenticate_client(client_id, token):
    # Implement your authentication logic
    return verify_token(client_id, token)
```

3. **Network Security**:
   - Use VPN for client connections
   - Implement IP whitelisting
   - Use secure communication protocols

### Monitoring

1. **Health Checks**:
```python
# Add to server.py
@app.route('/health')
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}
```

2. **Logging**:
   - Use structured logging
   - Implement log aggregation
   - Monitor privacy budget consumption

3. **Metrics**:
   - Track model performance
   - Monitor client participation
   - Alert on anomalies

### Scalability

1. **Horizontal Scaling**:
   - Use load balancers for multiple server instances
   - Implement client discovery mechanisms

2. **Database Integration**:
```python
# Example: Store metrics in database
def store_metrics(round_num, metrics):
    db.insert('fl_metrics', {
        'round': round_num,
        'metrics': json.dumps(metrics),
        'timestamp': datetime.now()
    })
```

3. **Caching**:
   - Cache model parameters
   - Use Redis for session management

## 🧪 Testing Deployment

### Local Testing

1. **Test with ngrok**:
```bash
# Install ngrok
npm install -g ngrok

# Start server locally
python server.py

# In another terminal, expose to internet
ngrok http 8080

# Use ngrok URL for remote clients
python client.py --hospital-id 0 --server-address abc123.ngrok.io:80
```

### Production Testing

1. **Health Check**:
```bash
curl https://your-deployment-url.com/health
```

2. **Client Connection Test**:
```bash
python client.py --hospital-id 0 --server-address your-deployment-url.com:443
```

3. **Load Testing**:
```bash
# Install artillery
npm install -g artillery

# Create load test config
artillery quick --count 10 --num 5 https://your-deployment-url.com/health
```

## 🔍 Troubleshooting

### Common Issues

1. **Port Binding Error**:
   ```
   Error: Address already in use
   ```
   **Solution**: Use environment variable `$PORT` in cloud deployments

2. **Module Import Error**:
   ```
   ModuleNotFoundError: No module named 'config'
   ```
   **Solution**: Ensure all files are included in deployment

3. **Memory Limits**:
   ```
   Error: Process killed (out of memory)
   ```
   **Solution**: Optimize model size or upgrade instance

4. **Client Connection Timeout**:
   ```
   Error: Connection timeout
   ```
   **Solution**: Check firewall settings and server status

### Debugging

1. **Check Logs**:
```bash
# Render logs
render logs --service your-service-name

# Docker logs
docker logs container-name

# Local logs
tail -f logs/server.log
```

2. **Test Components**:
```bash
# Test system locally
python test_system.py

# Test specific components
python -c "from config.settings import get_config; print(get_config('server'))"
```

## 📊 Monitoring and Maintenance

### Performance Monitoring

1. **Server Metrics**:
   - CPU usage
   - Memory consumption
   - Network I/O
   - Response times

2. **FL Metrics**:
   - Model convergence
   - Client participation rates
   - Privacy budget consumption
   - Training accuracy

### Maintenance Tasks

1. **Regular Updates**:
   - Update dependencies
   - Security patches
   - Model improvements

2. **Data Management**:
   - Log rotation
   - Metrics archival
   - Backup strategies

3. **Privacy Audits**:
   - Review privacy parameters
   - Audit data access logs
   - Validate compliance

## 🎯 Best Practices

1. **Environment Separation**:
   - Development
   - Staging
   - Production

2. **Configuration Management**:
   - Use environment variables
   - Version control configurations
   - Document all settings

3. **Backup and Recovery**:
   - Regular backups
   - Disaster recovery plans
   - Data retention policies

4. **Documentation**:
   - Deployment procedures
   - Troubleshooting guides
   - API documentation

## 📞 Support

For deployment issues:
1. Check the troubleshooting section
2. Review platform-specific documentation
3. Test locally first
4. Check logs for error details

Remember: This is an educational project. For production healthcare applications, additional security, compliance, and validation measures are required.
