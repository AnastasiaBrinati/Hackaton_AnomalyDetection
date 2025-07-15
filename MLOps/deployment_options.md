# 🌐 Deployment Online della Dashboard MLOps

## ❌ Limitazioni di GitHub
- GitHub Pages: Solo siti statici
- GitHub Actions: CI/CD, non hosting continuo
- GitHub non supporta servizi backend persistenti

## ✅ Soluzioni Alternative

### 1. 🆓 **Railway** (Raccomandato - Free tier)
```bash
# Installa Railway CLI
npm install -g @railway/cli

# Login e deploy
railway login
railway init
railway up
```

**Vantaggi:**
- Free tier generoso
- Deploy automatico da GitHub
- Supporto Docker
- Database inclusi

### 2. 🌊 **Render** (Free tier)
```yaml
# render.yaml
services:
  - type: web
    name: mlops-app
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python app.py
    
  - type: web
    name: grafana
    env: docker
    dockerfilePath: ./Dockerfile.grafana
```

### 3. ☁️ **Google Cloud Run** (Pay-per-use)
```bash
# Build e deploy
gcloud builds submit --tag gcr.io/PROJECT-ID/mlops-app
gcloud run deploy --image gcr.io/PROJECT-ID/mlops-app
```

### 4. 🔷 **Azure Container Instances** (Pay-per-use)
```bash
# Deploy container group
az container create \
  --resource-group myResourceGroup \
  --name mlops-dashboard \
  --image your-registry/mlops-app
```

### 5. 📦 **Heroku** (Hobby plan: $7/mese)
```bash
# Deploy
heroku create mlops-dashboard
git push heroku main
```

### 6. 🎯 **Grafana Cloud** (Servizio gestito)
- Grafana hostato gratuitamente
- Connettiti al tuo Prometheus
- 10k metriche gratuite/mese

## 🚀 Soluzione Rapida: Railway

### Step 1: Prepara il progetto
```bash
# Crea Dockerfile
cat > Dockerfile << EOF
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "app.py"]
EOF
```

### Step 2: Configura Railway
```bash
# Installa Railway CLI
npm install -g @railway/cli

# Login
railway login

# Inizializza progetto
railway init

# Deploy
railway up
```

### Step 3: Configura variabili ambiente
```bash
# Imposta porta per Railway
railway variables set PORT=5000
```

## 🐳 Deployment Docker Completo

### Multi-container setup
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus:/etc/prometheus
      
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    depends_on:
      - prometheus
```

## 📊 Dashboard Grafana Cloud

### Opzione 1: Grafana Cloud gratuito
1. Registrati su https://grafana.com
2. Crea workspace gratuito
3. Configura Prometheus remoto:

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

remote_write:
  - url: https://prometheus-prod-10-prod-us-central-0.grafana.net/api/prom/push
    basic_auth:
      username: USER_ID
      password: API_KEY

scrape_configs:
  - job_name: 'ml-model-app'
    static_configs:
      - targets: ['YOUR_APP_URL:5000']
```

### Opzione 2: Prometheus remoto
```python
# app.py - Aggiungi configurazione per remote write
import os
from prometheus_client import CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client.gateway import push_to_gateway

# Configura push gateway
PUSH_GATEWAY_URL = os.environ.get('PUSH_GATEWAY_URL', 'localhost:9091')

def push_metrics():
    push_to_gateway(PUSH_GATEWAY_URL, job='ml-model-app', registry=REGISTRY)
```

## 🎯 Raccomandazione

### Per iniziare velocemente:
1. **Railway** - Deploy in 5 minuti, free tier
2. **Grafana Cloud** - Dashboard gestita gratuitamente
3. **Connetti** i due servizi

### Per produzione:
1. **Google Cloud Run** - Scalabilità automatica
2. **Azure Container Instances** - Semplice e affidabile
3. **AWS ECS/Fargate** - Controllo completo

## 🔧 Setup Automatico Railway

```bash
# Script di deploy rapido
#!/bin/bash
echo "🚀 Deploy MLOps Dashboard su Railway..."

# Crea Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

ENV FLASK_ENV=production
CMD ["python", "app.py"]
EOF

# Crea railway.json
cat > railway.json << 'EOF'
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE"
  },
  "deploy": {
    "startCommand": "python app.py",
    "healthcheckPath": "/health",
    "healthcheckTimeout": 100,
    "restartPolicyType": "ON_FAILURE"
  }
}
EOF

# Deploy
railway login
railway init
railway up

echo "✅ Deploy completato!"
echo "🌐 La tua dashboard sarà disponibile su Railway URL"
```

## 📈 Monitoring in Produzione

### Aggiungi logging
```python
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mlops.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
```

### Health checks
```python
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'model_loaded': MODEL_LOADED,
        'uptime': time.time() - start_time
    })
```

---

🎯 **Pronto per il deploy?** Railway è la scelta più veloce per iniziare! 