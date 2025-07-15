#!/bin/bash

echo "🚀 Deploy MLOps Dashboard su Railway..."

# Verifica se Railway CLI è installato
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI non trovato. Installando..."
    npm install -g @railway/cli
fi

# Verifica se siamo in una directory git
if [ ! -d ".git" ]; then
    echo "📦 Inizializzo repository Git..."
    git init
    git add .
    git commit -m "Initial commit - MLOps Dashboard"
fi

# Crea file di configurazione Railway
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

# Crea .dockerignore
cat > .dockerignore << 'EOF'
.git
.gitignore
README.md
Dockerfile
.docker
node_modules
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env
pip-log.txt
pip-delete-this-directory.txt
.tox
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.DS_Store
.vscode
deployment_options.md
SETUP_GUIDE.md
start_mlops.sh
deploy_railway.sh
EOF

echo "🔐 Fai login su Railway (si aprirà il browser)..."
railway login

echo "🎯 Inizializzazione progetto Railway..."
railway init

echo "🚀 Deploy in corso..."
railway up

echo ""
echo "✅ Deploy completato!"
echo "🌐 La tua dashboard MLOps è online!"
echo "📊 Controlla lo stato su: https://railway.app/dashboard"
echo ""
echo "🎯 Prossimi passi:"
echo "1. Copia l'URL della tua app da Railway dashboard"
echo "2. Testa gli endpoint:"
echo "   - GET /health - Health check"
echo "   - GET /metrics - Metriche Prometheus"
echo "   - POST /predict - Predizioni"
echo "3. Configura Grafana Cloud per visualizzare le metriche"
echo ""
echo "🔧 Per aggiornare: git push e railway up" 